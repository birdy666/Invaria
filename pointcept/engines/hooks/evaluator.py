"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import numpy as np
import wandb
import torch
import torch.distributed as dist
import torch.utils.data
import pointops
from uuid import uuid4

import pointcept.utils.comm as comm
from pointcept.utils.misc import intersection_and_union_gpu

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class ClsEvaluator(HookBase):
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["cls_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                label,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/loss": loss_avg,
                        "val/mIoU": m_iou,
                        "val/mAcc": m_acc,
                        "val/allAcc": all_acc,
                    },
                    step=wandb.run.step,
                )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_acc  # save for saver
        self.trainer.comm_info["current_metric_name"] = "allAcc"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("allAcc", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def __init__(self, write_cls_iou=False):
        self.write_cls_iou = write_cls_iou
        self.extra_val_loaders = {}

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")
        eval_grid_sizes = getattr(self.trainer.cfg, 'eval_grid_sizes', None)
        if eval_grid_sizes and self.trainer.cfg.evaluate:
            self._build_extra_val_loaders(eval_grid_sizes)
            if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
                for label in self.extra_val_loaders:
                    wandb.define_metric(f"val_{label}/*", step_metric="Epoch")

    def _build_extra_val_loaders(self, grid_sizes):
        import copy
        from pointcept.datasets import build_dataset, collate_fn

        default_gs = None
        for t in self.trainer.cfg.data.val.transform:
            if isinstance(t, dict) and t.get('type') == 'GridSample':
                default_gs = t.get('grid_size')
                break

        for gs in grid_sizes:
            if default_gs is not None and abs(gs - default_gs) < 1e-6:
                continue

            val_cfg = copy.deepcopy(self.trainer.cfg.data.val)
            for t in val_cfg.transform:
                if isinstance(t, dict) and t.get('type') == 'GridSample':
                    t['grid_size'] = gs
                    break

            val_data = build_dataset(val_cfg)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    val_data
                )
            else:
                val_sampler = None
            loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.trainer.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.trainer.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )

            gs_cm = gs * 100
            label = (
                f"{gs_cm:.0f}cm" if gs_cm == int(gs_cm) else f"{gs_cm:.1f}cm"
            )
            self.extra_val_loaders[label] = loader
            self.trainer.logger.info(
                f"Built extra val loader at {label} ({len(val_data)} samples)"
            )

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        m_iou = self._eval_single(self.trainer.val_loader, prefix="val")
        self.trainer.comm_info["current_metric_value"] = m_iou
        self.trainer.comm_info["current_metric_name"] = "mIoU"

        for label, loader in self.extra_val_loaders.items():
            self._eval_single(loader, prefix=f"val_{label}")

    def _eval_single(self, val_loader, prefix="val"):
        self.trainer.logger.info(
            f">>>>>>>>>>>>>>>> Start Evaluation ({prefix}) >>>>>>>>>>>>>>>>"
        )
        self.trainer.model.eval()
        for i, input_dict in enumerate(val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]
            if "inverse" in input_dict.keys():
                assert "origin_segment" in input_dict.keys()
                pred = pred[input_dict["inverse"]]
                segment = input_dict["origin_segment"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar(f"{prefix}_intersection", intersection)
            self.trainer.storage.put_scalar(f"{prefix}_union", union)
            self.trainer.storage.put_scalar(f"{prefix}_target", target)
            self.trainer.storage.put_scalar(f"{prefix}_loss", loss.item())
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(val_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(
                info
                + "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history(f"{prefix}_loss").avg
        intersection = self.trainer.storage.history(f"{prefix}_intersection").total
        union = self.trainer.storage.history(f"{prefix}_union").total
        target = self.trainer.storage.history(f"{prefix}_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "{} result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                prefix, m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar(f"{prefix}/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar(f"{prefix}/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar(f"{prefix}/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar(f"{prefix}/allAcc", all_acc, current_epoch)
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        f"{prefix}/loss": loss_avg,
                        f"{prefix}/mIoU": m_iou,
                        f"{prefix}/mAcc": m_acc,
                        f"{prefix}/allAcc": all_acc,
                    },
                    step=wandb.run.step,
                )
            if self.write_cls_iou:
                for i in range(self.trainer.cfg.data.num_classes):
                    self.trainer.writer.add_scalar(
                        f"{prefix}/cls_{i}-{self.trainer.cfg.data.names[i]} IoU",
                        iou_class[i],
                        current_epoch,
                    )
                if self.trainer.cfg.enable_wandb:
                    for i in range(self.trainer.cfg.data.num_classes):
                        wandb.log(
                            {
                                "Epoch": current_epoch,
                                f"{prefix}/cls_{i}-{self.trainer.cfg.data.names[i]} IoU": iou_class[
                                    i
                                ],
                            },
                            step=wandb.run.step,
                        )
        self.trainer.logger.info(
            f"<<<<<<<<<<<<<<<<< End Evaluation ({prefix}) <<<<<<<<<<<<<<<<<"
        )
        return m_iou

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class MultiDensitySemSegEvaluator(HookBase):
    """Evaluate semantic segmentation at multiple voxel densities; best-model uses best_metric_grid_size."""

    def __init__(self, val_datasets, grid_sizes, best_metric_grid_size=None):
        self.val_dataset_cfgs = val_datasets
        self.grid_sizes = grid_sizes
        self.best_metric_grid_size = best_metric_grid_size or grid_sizes[0]
        self.val_loaders = None

    def _build_val_loaders(self):
        from pointcept.datasets import build_dataset, collate_fn

        val_loaders = []
        for ds_cfg in self.val_dataset_cfgs:
            val_data = build_dataset(ds_cfg)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    val_data
                )
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.trainer.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.trainer.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
            val_loaders.append(val_loader)
        return val_loaders

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        if self.val_loaders is None:
            self.val_loaders = self._build_val_loaders()

        self.trainer.model.eval()
        best_miou = None

        for loader, gs in zip(self.val_loaders, self.grid_sizes):
            tag = f"{int(gs * 100)}cm"
            self.trainer.logger.info(
                f">>>>>>>>>>>>>>>> Start Evaluation ({tag}) >>>>>>>>>>>>>>>>"
            )
            key_prefix = f"val_{tag}_"

            for i, input_dict in enumerate(loader):
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    output_dict = self.trainer.model(input_dict)
                output = output_dict["seg_logits"]
                loss = output_dict["loss"]
                pred = output.max(1)[1]
                segment = input_dict["segment"]
                if "inverse" in input_dict.keys():
                    assert "origin_segment" in input_dict.keys()
                    pred = pred[input_dict["inverse"]]
                    segment = input_dict["origin_segment"]
                intersection, union, target = intersection_and_union_gpu(
                    pred,
                    segment,
                    self.trainer.cfg.data.num_classes,
                    self.trainer.cfg.data.ignore_index,
                )
                if comm.get_world_size() > 1:
                    dist.all_reduce(intersection), dist.all_reduce(
                        union
                    ), dist.all_reduce(target)
                intersection, union, target = (
                    intersection.cpu().numpy(),
                    union.cpu().numpy(),
                    target.cpu().numpy(),
                )
                self.trainer.storage.put_scalar(
                    f"{key_prefix}intersection", intersection
                )
                self.trainer.storage.put_scalar(f"{key_prefix}union", union)
                self.trainer.storage.put_scalar(f"{key_prefix}target", target)
                self.trainer.storage.put_scalar(f"{key_prefix}loss", loss.item())
                self.trainer.logger.info(
                    "Test ({tag}): [{iter}/{max_iter}] Loss {loss:.4f} ".format(
                        tag=tag,
                        iter=i + 1,
                        max_iter=len(loader),
                        loss=loss.item(),
                    )
                )

            loss_avg = self.trainer.storage.history(f"{key_prefix}loss").avg
            intersection = self.trainer.storage.history(
                f"{key_prefix}intersection"
            ).total
            union = self.trainer.storage.history(f"{key_prefix}union").total
            target = self.trainer.storage.history(f"{key_prefix}target").total
            iou_class = intersection / (union + 1e-10)
            acc_class = intersection / (target + 1e-10)
            m_iou = np.mean(iou_class)
            m_acc = np.mean(acc_class)
            all_acc = sum(intersection) / (sum(target) + 1e-10)
            self.trainer.logger.info(
                "Val result ({tag}): mIoU/mAcc/allAcc {miou:.4f}/{macc:.4f}/{allacc:.4f}.".format(
                    tag=tag, miou=m_iou, macc=m_acc, allacc=all_acc
                )
            )
            for i in range(self.trainer.cfg.data.num_classes):
                self.trainer.logger.info(
                    "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.trainer.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=acc_class[i],
                    )
                )
            current_epoch = self.trainer.epoch + 1
            if self.trainer.writer is not None:
                self.trainer.writer.add_scalar(
                    f"val/{tag}_loss", loss_avg, current_epoch
                )
                self.trainer.writer.add_scalar(
                    f"val/{tag}_mIoU", m_iou, current_epoch
                )
                self.trainer.writer.add_scalar(
                    f"val/{tag}_mAcc", m_acc, current_epoch
                )
                self.trainer.writer.add_scalar(
                    f"val/{tag}_allAcc", all_acc, current_epoch
                )
                if self.trainer.cfg.enable_wandb:
                    wandb.log(
                        {
                            "Epoch": current_epoch,
                            f"val/{tag}_loss": loss_avg,
                            f"val/{tag}_mIoU": m_iou,
                            f"val/{tag}_mAcc": m_acc,
                            f"val/{tag}_allAcc": all_acc,
                        },
                        step=wandb.run.step,
                    )
            self.trainer.logger.info(
                f"<<<<<<<<<<<<<<<<< End Evaluation ({tag}) <<<<<<<<<<<<<<<<<",
            )

            if gs == self.best_metric_grid_size:
                best_miou = m_iou

        self.trainer.comm_info["current_metric_value"] = best_miou
        self.trainer.comm_info["current_metric_name"] = "mIoU"

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class InsSegEvaluator(HookBase):
    def __init__(self, segment_ignore_index=(-1,), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

        self.valid_class_names = None  # update in before train
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def before_train(self):
        self.valid_class_names = [
            self.trainer.cfg.data.names[i]
            for i in range(self.trainer.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def associate_instances(self, pred, segment, instance):
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        void_mask = np.in1d(segment, self.segment_ignore_index)

        assert (
            pred["pred_classes"].shape[0]
            == pred["pred_scores"].shape[0]
            == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                gt_instances[self.trainer.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.trainer.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.trainer.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.trainer.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                    float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
        return ap_scores

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        scenes = {}
        for i, input_dict in enumerate(self.trainer.val_loader):
            assert (
                len(input_dict["offset"]) == 1
            )  # currently only support bs 1 for each GPU
            data_name = input_dict.pop("name")[0]
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]

            segment = input_dict["segment"]
            instance = input_dict["instance"]
            # map to origin
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                idx = idx.cpu().flatten().long()
                output_dict["pred_masks"] = output_dict["pred_masks"][:, idx]
                segment = input_dict["origin_segment"]
                instance = input_dict["origin_instance"]

            gt_instances, pred_instance = self.associate_instances(
                output_dict, segment, instance
            )
            scenes[data_name] = dict(gt=gt_instances, pred=pred_instance)

            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )

        loss_avg = self.trainer.storage.history("val_loss").avg
        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)

        if comm.is_main_process():
            scenes = {}
            for _ in range(len(scenes_sync)):
                r = scenes_sync.pop()
                scenes.update(r)
                del r
            scenes = list(scenes.values())
            ap_scores = self.evaluate_matches(scenes)
            all_ap = ap_scores["all_ap"]
            all_ap_50 = ap_scores["all_ap_50%"]
            all_ap_25 = ap_scores["all_ap_25%"]
            self.trainer.logger.info(
                "Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
                    all_ap, all_ap_50, all_ap_25
                )
            )
            for i, label_name in enumerate(self.valid_class_names):
                ap = ap_scores["classes"][label_name]["ap"]
                ap_50 = ap_scores["classes"][label_name]["ap50%"]
                ap_25 = ap_scores["classes"][label_name]["ap25%"]
                self.trainer.logger.info(
                    "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                        idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                    )
                )
            current_epoch = self.trainer.epoch + 1
            if self.trainer.writer is not None:
                self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
                self.trainer.writer.add_scalar("val/mAP", all_ap, current_epoch)
                self.trainer.writer.add_scalar("val/AP50", all_ap_50, current_epoch)
                self.trainer.writer.add_scalar("val/AP25", all_ap_25, current_epoch)
                if self.trainer.cfg.enable_wandb:
                    wandb.log(
                        {
                            "Epoch": current_epoch,
                            "val/loss": loss_avg,
                            "val/mAP": all_ap,
                            "val/AP50": all_ap_50,
                            "val/AP25": all_ap_25,
                        },
                        step=wandb.run.step,
                    )
            self.trainer.logger.info(
                "<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<"
            )
            self.trainer.comm_info["current_metric_value"] = all_ap_50  # save for saver
            self.trainer.comm_info["current_metric_name"] = "AP50"  # save for saver
