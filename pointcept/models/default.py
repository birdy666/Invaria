import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import torch_cluster
from peft import LoraConfig, get_peft_model
from collections import OrderedDict

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        ignore_index=-1,
        level_loss_enabled=True,
        level_channels=None,
        level_loss_weights=None,
        # Multi-pass training tiers derived geometrically: tiers = [n0 * n0_scale**i for i in range(num_passes)].
        # loss_weights gives one weight per tier (length must equal num_passes). num_passes <= 1 disables multi-pass.
        n0=None,
        n0_scale=4,
        num_passes=1,
        loss_weights=None,
        # "adaptive" picks n0 from the derived tier list; dict {"n0": int} forces a fixed n0; None disables.
        eval_pass=None,
        bottleneck_consistency_weight=0.0,
    ):
        super().__init__()
        self.bottleneck_consistency_weight = bottleneck_consistency_weight
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.level_loss_enabled = level_loss_enabled
        self.level_loss_weights = level_loss_weights
        self.backbone_out_channels = backbone_out_channels

        if n0 is not None and num_passes is not None and num_passes >= 1:
            self.n0_tiers = [int(n0 * (n0_scale ** i)) for i in range(num_passes)]
            if loss_weights is None:
                loss_weights = [1.0] * num_passes
            assert len(loss_weights) == num_passes, (
                f"loss_weights length {len(loss_weights)} must match num_passes {num_passes}"
            )
            self.multi_pass = (
                [dict(n0=tier, loss_weight=w) for tier, w in zip(self.n0_tiers, loss_weights)]
                if num_passes > 1
                else None
            )
        else:
            self.n0_tiers = None
            self.multi_pass = None
        self.eval_pass = eval_pass
        self.eval_n0_tiers = self.n0_tiers if self.n0_tiers is not None else [24576, 98304, 98304 * 4]

        if level_loss_enabled and level_channels is not None:
            self.level_projs = nn.ModuleList([
                nn.Linear(ch, backbone_out_channels) if ch != backbone_out_channels else nn.Identity()
                for ch in level_channels
            ])
        else:
            self.level_projs = None

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @staticmethod
    def _nested_sample(batch, N, device, tier_sizes):
        """Per-scene index subsets where each tier ⊂ the next-larger tier."""
        B = int(batch.max().item()) + 1
        counts = torch.zeros(B, dtype=torch.long, device=device)
        counts.scatter_add_(0, batch, torch.ones(N, dtype=torch.long, device=device))
        starts = torch.zeros(B, dtype=torch.long, device=device)
        starts[1:] = counts[:-1].cumsum(0)

        parent_per_scene = [
            torch.arange(
                int(starts[b].item()),
                int(starts[b].item()) + int(counts[b].item()),
                device=device,
            )
            for b in range(B)
        ]
        nested = {}
        for tier in sorted(tier_sizes, reverse=True):
            new_parent = []
            for b in range(B):
                p = parent_per_scene[b]
                pick = min(p.shape[0], tier)
                sel = p[torch.randperm(p.shape[0], device=device)[:pick]]
                new_parent.append(sel)
            parent_per_scene = new_parent
            nested[tier] = torch.cat(new_parent, dim=0)
        return nested

    def _run_backbone_and_head(self, point):
        point = self.backbone(point)
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return feat, seg_logits, point

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)

        if self.training and self.multi_pass is not None:
            return_dict = dict(loss=torch.tensor(0.0, device=point.feat.device))
            bottleneck_infos = []

            N_full = input_dict["feat"].shape[0]
            device_full = input_dict["feat"].device
            batch_full = (
                input_dict["batch"]
                if "batch" in input_dict
                else offset2batch(input_dict["offset"])
            )
            tier_sizes = [pc["n0"] for pc in self.multi_pass]
            nested_indices = self._nested_sample(
                batch_full, N_full, device_full, tier_sizes
            )
            full_idx = torch.arange(N_full, device=device_full)

            for i, pass_cfg in enumerate(self.multi_pass):
                n0 = pass_cfg["n0"]
                loss_weight = pass_cfg.get("loss_weight", 1.0)

                backbone_idx = nested_indices[n0]
                if i < len(self.multi_pass) - 1:
                    next_tier = nested_indices[self.multi_pass[i + 1]["n0"]]
                    in_backbone = torch.zeros(N_full, dtype=torch.bool, device=device_full)
                    in_backbone[backbone_idx] = True
                    extra_pool = next_tier[~in_backbone[next_tier]]
                    max_extra = int(extra_pool.shape[0])
                    k_extra = int(torch.randint(0, max_extra + 1, (1,)).item()) if max_extra > 0 else 0
                    if k_extra > 0:
                        perm = torch.randperm(max_extra, device=device_full)[:k_extra]
                        supervise_idx = torch.cat([backbone_idx, extra_pool[perm]], dim=0)
                    else:
                        supervise_idx = backbone_idx
                else:
                    supervise_idx = full_idx

                sub_point = Point({
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in input_dict.items()
                })
                sub_point["backbone_indices"] = backbone_idx
                sub_point["supervise_indices"] = supervise_idx
                sub_point["n0"] = n0
                if "skip_inner" in pass_cfg:
                    sub_point["skip_inner"] = pass_cfg["skip_inner"]

                _, seg_logits, out_point = self._run_backbone_and_head(sub_point)

                # Backbone subsample marks uncovered points so they don't contribute to the loss.
                segment = input_dict["segment"].clone()
                if "points_masked" in out_point.keys() and out_point.points_masked is not None:
                    segment[out_point.points_masked] = self.ignore_index
                pass_loss = self.criteria(seg_logits, segment)
                return_dict["loss"] = return_dict["loss"] + loss_weight * pass_loss
                return_dict[f"loss_pass{i}"] = pass_loss

                if (self.bottleneck_consistency_weight > 0
                        and "bottleneck_feat" in out_point.keys()):
                    bottleneck_infos.append(dict(
                        feat=out_point.bottleneck_feat,
                        coord=out_point.bottleneck_coord,
                        batch=out_point.bottleneck_batch,
                    ))

            if self.bottleneck_consistency_weight > 0 and len(bottleneck_infos) >= 2:
                consistency_loss = torch.tensor(0.0, device=point.feat.device)
                n_pairs = 0
                for i in range(len(bottleneck_infos)):
                    for j in range(i + 1, len(bottleneck_infos)):
                        a = bottleneck_infos[i]
                        b = bottleneck_infos[j]
                        if a["feat"].shape[0] <= b["feat"].shape[0]:
                            small, large = a, b
                        else:
                            small, large = b, a
                        assign = torch_cluster.knn(
                            x=large["coord"],
                            y=small["coord"],
                            batch_x=large["batch"],
                            batch_y=small["batch"],
                            k=1,
                        )
                        matched_large = large["feat"][assign[1]]
                        pair_loss = 1.0 - F.cosine_similarity(
                            small["feat"], matched_large, dim=-1
                        ).mean()
                        consistency_loss = consistency_loss + pair_loss
                        n_pairs += 1
                consistency_loss = consistency_loss / max(n_pairs, 1)
                return_dict["loss"] = return_dict["loss"] + self.bottleneck_consistency_weight * consistency_loss
                return_dict["loss_bottleneck_consistency"] = consistency_loss

            return return_dict

        if not self.training:
            if self.eval_pass == "adaptive":
                N = point.feat.shape[0]
                # Floor to largest tier <= N, fallback to smallest tier.
                n0_tiers = sorted(self.eval_n0_tiers)
                point["n0"] = n0_tiers[0]
                for t in n0_tiers:
                    if t <= N:
                        point["n0"] = t
                    else:
                        break
            elif self.eval_pass is not None:
                if "n0" in self.eval_pass:
                    point["n0"] = self.eval_pass["n0"]
                if "skip_inner" in self.eval_pass:
                    point["skip_inner"] = self.eval_pass["skip_inner"]
        feat, seg_logits, point = self._run_backbone_and_head(point)
        return_dict = dict()
        if return_point:
            return_dict["point"] = point
        if self.training:
            segment = input_dict["segment"].clone()
            if "points_masked" in point.keys() and point.points_masked is not None:
                segment[point.points_masked] = self.ignore_index
            loss = self.criteria(seg_logits, segment)
            return_dict["loss"] = loss

            if (self.level_loss_enabled and
                self.level_projs is not None and
                "level_features" in point.keys() and
                "level_point_to_node" in point.keys()):

                level_features = point.level_features
                level_point_to_node = point.level_point_to_node
                num_levels = point.num_levels

                for level in range(num_levels):
                    if level not in level_features:
                        dummy_feat = torch.zeros(1, self.level_projs[level].in_features if hasattr(self.level_projs[level], 'in_features') else self.backbone_out_channels,
                                                  device=feat.device, dtype=feat.dtype)
                        dummy_out = self.seg_head(self.level_projs[level](dummy_feat))
                        return_dict["loss"] = return_dict["loss"] + 0 * dummy_out.sum()
                        continue

                    node_feat = level_features[level]
                    point_to_node = level_point_to_node[level]
                    valid_points = point_to_node >= 0
                    projected_feat = self.level_projs[level](node_feat)
                    node_logits = self.seg_head(projected_feat)

                    if not valid_points.any():
                        return_dict["loss"] = return_dict["loss"] + 0 * node_logits.sum()
                        continue

                    safe_indices = point_to_node.clamp(min=0)
                    point_logits = node_logits[safe_indices]
                    level_segment = segment.clone()
                    level_segment[~valid_points] = self.ignore_index
                    level_loss = self.criteria(point_logits, level_segment)
                    return_dict[f"loss_level_{level}"] = level_loss

                    if self.level_loss_weights is not None and level < len(self.level_loss_weights):
                        weight = self.level_loss_weights[level]
                        return_dict["loss"] = return_dict["loss"] + weight * level_loss

        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultLORASegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        backbone_path=None,
        keywords=None,
        replacements=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.keywords = keywords
        self.replacements = replacements
        self.backbone = build_model(backbone)
        backbone_weight = torch.load(
            backbone_path,
            map_location=lambda storage, loc: storage.cuda(),
        )
        self.backbone_load(backbone_weight)

        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora

        if self.use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["qkv"],
                # target_modules=["query", "value"],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.backbone.enc = get_peft_model(self.backbone.enc, lora_config)

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if self.use_lora:
            for name, param in self.backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
        self.backbone.enc.print_trainable_parameters()

    def backbone_load(self, checkpoint):
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if not key.startswith("module."):
                key = "module." + key  # xxx.xxx -> module.xxx.xxx
            # Now all keys contain "module." no matter DDP or not.
            if self.keywords in key:
                key = key.replace(self.keywords, self.replacements)
            key = key[7:]  # module.xxx.xxx -> xxx.xxx
            if key.startswith("backbone."):
                key = key[9:]
            weight[key] = value
        load_state_info = self.backbone.load_state_dict(weight, strict=False)
        print(f"Missing keys: {load_state_info[0]}")
        print(f"Unexpected keys: {load_state_info[1]}")

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.freeze_backbone and not self.use_lora:
            with torch.no_grad():
                point = self.backbone(point)
        else:
            point = self.backbone(point)

        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            return_dict["point"] = point

        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DINOEnhancedSegmentor(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone) if backbone is not None else None
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.backbone is not None and self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.backbone is not None:
            if self.freeze_backbone:
                with torch.no_grad():
                    point = self.backbone(point)
            else:
                point = self.backbone(point)
            point_list = [point]
            while "unpooling_parent" in point_list[-1].keys():
                point_list.append(point_list[-1].pop("unpooling_parent"))
            for i in reversed(range(1, len(point_list))):
                point = point_list[i]
                parent = point_list[i - 1]
                assert "pooling_inverse" in point.keys()
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = point_list[0]
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = [point.feat]
        else:
            feat = []
        dino_coord = input_dict["dino_coord"]
        dino_feat = input_dict["dino_feat"]
        dino_offset = input_dict["dino_offset"]
        idx = torch_cluster.knn(
            x=dino_coord,
            y=point.origin_coord,
            batch_x=offset2batch(dino_offset),
            batch_y=offset2batch(point.origin_offset),
            k=1,
        )[1]

        feat.append(dino_feat[idx])
        feat = torch.concatenate(feat, dim=-1)
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
