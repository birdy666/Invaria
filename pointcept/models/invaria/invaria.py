import logging
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_cluster

from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point
from pointcept.models.utils.misc import offset2batch, batch2offset
from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    Block as PTv3Block,
    SerializedPooling,
    SerializedUnpooling,
)

logger = logging.getLogger(__name__)


def _make_norm_factory(kind: str):
    """Return a callable(channels) -> norm module for kind in {"bn", "ln"}."""
    kind = kind.lower()
    if kind == "bn":
        return partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
    if kind == "ln":
        return nn.LayerNorm
    raise ValueError(f"Unknown norm kind: {kind!r} (expected 'bn' or 'ln')")


class PredictorAttn(nn.Module):
    """Cross-attention upsampling: dense points query sparse backbone features."""

    def __init__(self, in_channels, channels, num_heads=4, k=16, norm_layer=None):
        super().__init__()
        self.k = k
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.kv_proj = nn.Linear(channels, channels * 2)
        self.rel_pos_mlp = nn.Sequential(
            nn.Linear(3, channels),
            nn.GELU(),
            nn.Linear(channels, num_heads),
        )
        self.out_proj = nn.Linear(channels, channels)
        self.norm = (norm_layer or nn.LayerNorm)(channels)

    def forward(self, dense_coord, dense_batch, dense_input_feat,
                sparse_coord, sparse_batch, sparse_feat):
        N = dense_coord.shape[0]
        k = min(self.k, sparse_feat.shape[0])
        H = self.num_heads
        D = self.head_dim

        knn_result = torch_cluster.knn(
            x=sparse_coord, y=dense_coord,
            batch_x=sparse_batch, batch_y=dense_batch,
            k=k,
        )
        row, col = knn_result

        q = self.q_proj(dense_input_feat).view(N, H, D)

        kv = self.kv_proj(sparse_feat)
        k_feat, v_feat = kv.chunk(2, dim=-1)

        k_gathered = k_feat[col].view(N, k, H, D)
        v_gathered = v_feat[col].view(N, k, H, D)

        attn = torch.einsum('nhd,nkhd->nkh', q, k_gathered) * self.scale
        rel_pos = (dense_coord[row] - sparse_coord[col]).view(N, k, 3)
        attn = attn + self.rel_pos_mlp(rel_pos)
        attn = attn.softmax(dim=1)

        out = torch.einsum('nkh,nkhd->nhd', attn, v_gathered)
        out = out.reshape(N, H * D)
        out = self.out_proj(out)
        out = self.norm(out)
        return out


@MODELS.register_module("Invaria")
class Invaria(nn.Module):
    """Sample n0 points, run the PTv3 backbone, predict features back on all input points."""

    def __init__(
        self,
        in_channels: int = 6,
        n0: int = 98304,
        predictor_k: int = 16,
        predictor_num_heads: int = 4,
        order: Tuple[str, ...] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        stride: Tuple[int, ...] = (2, 2, 2, 2),
        enc_depths: Tuple[int, ...] = (2, 2, 2, 6, 2),
        enc_channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
        enc_num_head: Tuple[int, ...] = (2, 4, 8, 16, 32),
        enc_patch_size: Tuple[int, ...] = (1024, 1024, 1024, 1024, 1024),
        dec_depths: Tuple[int, ...] = (2, 2, 2, 2),
        dec_channels: Tuple[int, ...] = (64, 64, 128, 256),
        dec_num_head: Tuple[int, ...] = (4, 4, 8, 16),
        dec_patch_size: Tuple[int, ...] = (1024, 1024, 1024, 1024),
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        shuffle_orders: bool = True,
        pre_norm: bool = True,
        enable_rpe: bool = False,
        enable_flash: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        grid_size_samples: int = 50,
        grid_size_scale: float = 1.5,
        grid_size_reduce: str = "mean",  # "mean" | "min" | "p<int>" (e.g. "p10")
        predictor_mode: str = "attn",
        predictor_nn_blocks: int = 3,
        predictor_nn_kernel_size: int = 3,
        attn_norm: str = "ln",
        other_norm: str = "bn",
        verbose: bool = False,
    ):
        super().__init__()
        self.n0 = n0
        self.grid_size_samples = grid_size_samples
        self.grid_size_scale = grid_size_scale
        self.grid_size_reduce = grid_size_reduce
        self.predictor_mode = predictor_mode
        self.verbose = verbose
        self.predictor_k = predictor_k
        self.predictor_num_heads = predictor_num_heads
        self.order = [order] if isinstance(order, str) else list(order)
        self.shuffle_orders = shuffle_orders

        num_stages = len(enc_depths)
        assert num_stages == len(stride) + 1
        assert num_stages == len(enc_channels)
        assert len(dec_depths) == num_stages - 1

        # other_norm: conv/pool/unpool; attn_norm: attention blocks and predictor
        bn_layer = _make_norm_factory(other_norm)
        ln_layer = _make_norm_factory(attn_norm)
        act_layer = nn.GELU

        self.embedding = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                enc_channels[0],
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            ),
            norm=bn_layer(enc_channels[0]),
            act=act_layer(),
        )

        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(num_stages):
            enc_drop_path_s = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    PTv3Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_s[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"enc{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            self.enc.add(module=enc, name=f"enc{s}")

        dec_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
        ]
        self.dec = PointSequential()
        dec_channels_full = list(dec_channels) + [enc_channels[-1]]
        for s in reversed(range(num_stages - 1)):
            dec_drop_path_s = dec_drop_path[
                sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
            ]
            dec_drop_path_s.reverse()
            dec = PointSequential()
            dec.add(
                SerializedUnpooling(
                    in_channels=dec_channels_full[s + 1],
                    skip_channels=enc_channels[s],
                    out_channels=dec_channels_full[s],
                    norm_layer=bn_layer,
                    act_layer=act_layer,
                ),
                name="up",
            )
            for i in range(dec_depths[s]):
                dec.add(
                    PTv3Block(
                        channels=dec_channels_full[s],
                        num_heads=dec_num_head[s],
                        patch_size=dec_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=dec_drop_path_s[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"enc{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            self.dec.add(module=dec, name=f"dec{s}")

        if predictor_mode == "attn":
            self.predictor = PredictorAttn(
                in_channels=in_channels,
                channels=dec_channels[0],
                num_heads=predictor_num_heads,
                k=predictor_k,
                norm_layer=ln_layer,
            )
        elif predictor_mode == "nn":
            self.predictor_blocks = nn.ModuleList()
            C = dec_channels[0]
            for i in range(predictor_nn_blocks):
                self.predictor_blocks.append(nn.ModuleDict(dict(
                    conv=spconv.SubMConv3d(C, C, kernel_size=predictor_nn_kernel_size, bias=True, indice_key="predictor_nn"),
                    norm1=bn_layer(C),
                    mlp=nn.Sequential(
                        nn.Linear(C, C * mlp_ratio),
                        nn.GELU(),
                        nn.Linear(C * mlp_ratio, C),
                    ),
                    norm2=bn_layer(C),
                )))

    def _reduce_nn_dists(self, nn_dists):
        mode = self.grid_size_reduce
        if mode == "mean":
            return nn_dists.mean()
        if mode == "min":
            return nn_dists.min()
        if mode.startswith("p"):
            q = float(mode[1:]) / 100.0
            return torch.quantile(nn_dists, q)
        raise ValueError(f"Unknown grid_size_reduce: {mode}")

    @staticmethod
    def _curve_neighbors(target_coord, target_batch, source_coord, source_batch, B, k=3):
        """Indices of the k source points nearest to each target along a Morton curve."""
        device = target_coord.device
        N = target_coord.shape[0]
        half = (k - 1) // 2
        offsets = torch.arange(-half, k - half, device=device)
        nearest = torch.empty((N, k), dtype=torch.long, device=device)
        for b in range(B):
            t_mask = (target_batch == b)
            s_mask = (source_batch == b)
            if not t_mask.any():
                continue
            t_coord_b = target_coord[t_mask]
            s_coord_b = source_coord[s_mask]
            n_s = s_coord_b.shape[0]
            if n_s == 0:
                nearest[t_mask] = 0
                continue
            all_coord = torch.cat([t_coord_b, s_coord_b], dim=0)
            mn = all_coord.min(0).values
            rng = (all_coord.max(0).values - mn).clamp(min=1e-6)
            t_q = ((t_coord_b - mn) / rng * 1023).long().clamp(0, 1023)
            s_q = ((s_coord_b - mn) / rng * 1023).long().clamp(0, 1023)
            t_morton = t_q[:, 0] * (1024 * 1024) + t_q[:, 1] * 1024 + t_q[:, 2]
            s_morton = s_q[:, 0] * (1024 * 1024) + s_q[:, 1] * 1024 + s_q[:, 2]
            s_sort = s_morton.argsort()
            s_morton_sorted = s_morton[s_sort]
            idx = torch.searchsorted(s_morton_sorted, t_morton).clamp(max=n_s - 1)
            cand_curve = (idx.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, n_s - 1)
            cand_local = s_sort[cand_curve]
            s_global = s_mask.nonzero(as_tuple=True)[0]
            nearest[t_mask] = s_global[cand_local]
        return nearest

    def forward(self, point: Point) -> Point:
        N = point.feat.shape[0]
        batch = (
            point.batch
            if "batch" in point.keys() and point.batch is not None
            else offset2batch(point.offset)
        )
        device = point.feat.device
        B = batch.max().item() + 1

        n0 = int(point.get("n0", self.n0) if hasattr(point, "get") else self.n0)
        supervise_n = int(point.get("supervise_n", 0) if hasattr(point, "get") else 0)

        backbone_indices_in = (
            point.get("backbone_indices", None) if hasattr(point, "get") else None
        )
        supervise_indices_in = (
            point.get("supervise_indices", None) if hasattr(point, "get") else None
        )

        full_coord = point.coord
        full_feat = point.feat
        full_batch = batch
        full_offset = (
            point.offset if "offset" in point.keys() else batch2offset(batch)
        )

        if backbone_indices_in is not None:
            if supervise_indices_in is not None and supervise_indices_in.shape[0] < N:
                target_mask = torch.zeros(N, dtype=torch.bool, device=device)
                target_mask[supervise_indices_in] = True
                target_indices = supervise_indices_in
            else:
                target_mask = None
                target_indices = torch.arange(N, device=device)

            sub_point = Point(
                dict(
                    feat=full_feat[backbone_indices_in],
                    coord=full_coord[backbone_indices_in],
                    batch=full_batch[backbone_indices_in],
                )
            )
            del point
        else:
            counts = torch.zeros(B, dtype=torch.long, device=device)
            counts.scatter_add_(0, batch, torch.ones(N, dtype=torch.long, device=device))
            scene_start = torch.zeros(B, dtype=torch.long, device=device)
            scene_start[1:] = counts[:-1].cumsum(0)

            if supervise_n > 0 and supervise_n < N:
                target_mask = torch.zeros(N, dtype=torch.bool, device=device)
                for b in range(B):
                    start = scene_start[b].item()
                    n_scene = counts[b].item()
                    pick = min(n_scene, supervise_n)
                    perm = torch.randperm(n_scene, device=device)[:pick]
                    target_mask[start + perm] = True
                target_indices = target_mask.nonzero(as_tuple=True)[0]
            else:
                target_mask = None
                target_indices = torch.arange(N, device=device)

            n_target = target_indices.shape[0]
            target_coord_sel = full_coord[target_indices]
            target_batch_sel = full_batch[target_indices]

            target_counts = torch.zeros(B, dtype=torch.long, device=device)
            target_counts.scatter_add_(
                0, target_batch_sel,
                torch.ones(n_target, dtype=torch.long, device=device),
            )
            target_scene_start = torch.zeros(B, dtype=torch.long, device=device)
            target_scene_start[1:] = target_counts[:-1].cumsum(0)

            needs_subsample = target_counts.max().item() > n0
            if needs_subsample:
                backbone_sel = torch.zeros(n_target, dtype=torch.bool, device=device)
                for b in range(B):
                    ts = target_scene_start[b].item()
                    tc = target_counts[b].item()
                    pick = min(tc, n0)
                    if pick >= tc:
                        backbone_sel[ts:ts + tc] = True
                        continue
                    sc = target_coord_sel[ts:ts + tc]
                    mn = sc.min(0)[0]
                    rng = (sc.max(0)[0] - mn).clamp(min=1e-6)
                    q = ((sc - mn) / rng * 1023).long().clamp(0, 1023)
                    morton = q[:, 0] * (1024 * 1024) + q[:, 1] * 1024 + q[:, 2]
                    sorted_idx = morton.argsort()
                    positions = torch.linspace(0, tc - 1, pick, device=device).long()
                    backbone_sel[ts + sorted_idx[positions]] = True
                backbone_indices = target_indices[backbone_sel]
                sub_point = Point(
                    dict(
                        feat=full_feat[backbone_indices],
                        coord=full_coord[backbone_indices],
                        batch=full_batch[backbone_indices],
                    )
                )
                del point
            else:
                if target_mask is not None:
                    sub_point = Point(
                        dict(
                            feat=full_feat[target_indices],
                            coord=full_coord[target_indices],
                            batch=full_batch[target_indices],
                        )
                    )
                    del point
                else:
                    sub_point = point

        with torch.no_grad():
            n_sample = min(self.grid_size_samples, sub_point.coord.shape[0])
            idx = torch.randperm(sub_point.coord.shape[0], device=device)[:n_sample]
            sample_coord = sub_point.coord[idx]
            all_coord = sub_point.coord
            dists = torch.cdist(sample_coord.unsqueeze(0), all_coord.unsqueeze(0)).squeeze(0)
            dists[torch.arange(n_sample, device=device), idx] = float('inf')
            grid_size = self._reduce_nn_dists(dists.min(dim=1).values) * self.grid_size_scale
        if self.verbose:
            with torch.no_grad():
                full_mask_s0 = full_batch == 0
                sub_mask_s0 = sub_point.batch == 0
                n_full_s0 = int(full_mask_s0.sum().item())
                n_points_s0 = int(sub_mask_s0.sum().item())
                if n_points_s0 > 1:
                    coord_s0 = sub_point.coord[sub_mask_s0]
                    n_sample_s0 = min(self.grid_size_samples, n_points_s0)
                    idx_s0 = torch.randperm(n_points_s0, device=device)[:n_sample_s0]
                    sample_s0 = coord_s0[idx_s0]
                    d_s0 = torch.cdist(sample_s0.unsqueeze(0), coord_s0.unsqueeze(0)).squeeze(0)
                    d_s0[torch.arange(n_sample_s0, device=device), idx_s0] = float("inf")
                    grid_size_s0 = self._reduce_nn_dists(d_s0.min(dim=1).values).item() * self.grid_size_scale
                else:
                    grid_size_s0 = float("nan")
            print(f"[Invaria scene0] n0={n0}, n_full={n_full_s0}, n_points={n_points_s0}, grid_size={grid_size_s0:.6f}")
        sub_point["grid_size"] = grid_size

        sub_point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        sub_point.sparsify()
        sub_point = self.embedding(sub_point)

        sub_point = self.enc(sub_point)

        bottleneck_feat = sub_point.feat.clone()
        bottleneck_coord = sub_point.coord.clone()
        bottleneck_batch = sub_point.batch.clone()

        sub_point = self.dec(sub_point)

        sub_point["bottleneck_feat"] = bottleneck_feat
        sub_point["bottleneck_coord"] = bottleneck_coord
        sub_point["bottleneck_batch"] = bottleneck_batch

        C = sub_point.feat.shape[1]

        if target_mask is not None:
            target_coord = full_coord[target_mask]
            target_batch = full_batch[target_mask]
            target_input_feat = full_feat[target_mask]
        else:
            target_coord = full_coord
            target_batch = full_batch
            target_input_feat = full_feat

        if self.predictor_mode == "attn":
            target_feat = self.predictor(
                dense_coord=target_coord,
                dense_batch=target_batch,
                dense_input_feat=target_input_feat,
                sparse_coord=sub_point.coord,
                sparse_batch=sub_point.batch,
                sparse_feat=sub_point.feat,
            )
        else:
            k_interp = 3
            nearest_k = self._curve_neighbors(
                target_coord, target_batch,
                sub_point.coord, sub_point.batch, B, k=k_interp,
            )
            neigh_coord = sub_point.coord[nearest_k]
            d = (target_coord.unsqueeze(1) - neigh_coord).norm(dim=-1)
            w = 1.0 / (d + 1e-8)
            w = w / w.sum(dim=1, keepdim=True)
            neighbor_feat = sub_point.feat[nearest_k]
            target_feat = (neighbor_feat * w.unsqueeze(-1)).sum(dim=1)

            with torch.no_grad():
                # estimate spacing within scene 0 only
                scene_mask = (target_batch == 0)
                scene_coord = target_coord[scene_mask]
                n_sample = min(self.grid_size_samples, scene_coord.shape[0])
                s_idx = torch.randperm(scene_coord.shape[0], device=device)[:n_sample]
                s_sample = scene_coord[s_idx]
                s_dists = torch.cdist(s_sample.unsqueeze(0), scene_coord.unsqueeze(0)).squeeze(0)
                s_dists[torch.arange(n_sample, device=device), s_idx] = float('inf')
                predictor_grid_size = s_dists.min(dim=1).values.mean() * self.grid_size_scale

            grid_coord = torch.floor(target_coord / predictor_grid_size).int()
            grid_coord -= grid_coord.min(dim=0)[0]
            spatial_shape = (grid_coord.max(dim=0)[0] + 1).tolist()
            spatial_shape = [max(s, 3) for s in spatial_shape]
            indices = torch.cat([target_batch.unsqueeze(1).int(), grid_coord], dim=1)

            for blk in self.predictor_blocks:
                st = spconv.SparseConvTensor(
                    features=target_feat,
                    indices=indices,
                    spatial_shape=spatial_shape,
                    batch_size=B,
                )
                target_feat = target_feat + blk["norm1"](blk["conv"](st).features)
                target_feat = target_feat + blk["norm2"](blk["mlp"](target_feat))

        output_feat = torch.zeros(N, C, device=device, dtype=target_feat.dtype)
        if target_mask is not None:
            output_feat[target_mask] = target_feat
            unsupervised_mask = ~target_mask
        else:
            output_feat = target_feat
            unsupervised_mask = None

        out_point = Point(
            dict(
                feat=output_feat,
                coord=full_coord,
                batch=full_batch,
                offset=full_offset,
            )
        )
        if unsupervised_mask is not None:
            out_point["points_masked"] = unsupervised_mask.nonzero(as_tuple=True)[0]
        out_point["bottleneck_feat"] = bottleneck_feat
        out_point["bottleneck_coord"] = bottleneck_coord
        out_point["bottleneck_batch"] = bottleneck_batch
        return out_point
