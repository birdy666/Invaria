_base_ = ["../_base_/default_runtime.py"]

batch_size = 36
num_worker = 48
mix_prob = 0.8
empty_cache = False
enable_amp = True
clip_grad = None

model = dict(
    type="DefaultSegmentorV2",
    num_classes=16,
    backbone_out_channels=64,
    ignore_index=-1,
    level_loss_enabled=False,
    bottleneck_consistency_weight=0.1,
    n0=4096,
    n0_scale=2,
    num_passes=3,
    loss_weights=[0.3, 0.6, 1.0],
    eval_pass="adaptive",
    backbone=dict(
        type="Invaria",
        in_channels=1,  # strength only; coord is consumed by the sparse backbone
        n0=32768,
        predictor_k=8,
        predictor_num_heads=4,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 2, 1),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        grid_size_samples=200,
        grid_size_scale=0.8,
        grid_size_reduce="p50",
        predictor_mode="nn",
        predictor_nn_blocks=2,
        predictor_nn_kernel_size=3,
        attn_norm="ln",
        other_norm="ln",
        verbose=True,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

epoch = 50
eval_epoch = 10
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

dataset_type = "NuScenesDataset"
data_root = "data/nuscenes"
ignore_index = -1
names = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]

grid_size = 0.05
eval_grid_sizes = [0.025, 0.1, 0.15]

_aug_pre = [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
]

_aug_post = [
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "segment"),
        feat_keys=("strength",),
    ),
]

_train_5cm = _aug_pre + [dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train", return_grid_coord=True)] + _aug_post

_val_collect = [
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
        feat_keys=("strength",),
    ),
]

_val_2_5cm = [dict(type="Copy", keys_dict={"segment": "origin_segment"}), dict(type="GridSample", grid_size=0.025, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True)] + _val_collect
_val_5cm = [dict(type="Copy", keys_dict={"segment": "origin_segment"}), dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True)] + _val_collect
_val_10cm = [dict(type="Copy", keys_dict={"segment": "origin_segment"}), dict(type="GridSample", grid_size=0.1, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True)] + _val_collect
_val_15cm = [dict(type="Copy", keys_dict={"segment": "origin_segment"}), dict(type="GridSample", grid_size=0.15, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True)] + _val_collect

data = dict(
    num_classes=16,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type="ConcatDataset",
        datasets=[
            dict(type=dataset_type, split="train", data_root=data_root, transform=_train_5cm, test_mode=False, ignore_index=ignore_index),
        ],
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=_val_5cm,
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.025,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="Update", keys_dict={"grid_size": 0.05}),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_size", "index"),
                    feat_keys=("strength",),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        ),
        ignore_index=ignore_index,
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="MultiDensitySemSegEvaluator",
        val_datasets=[
            dict(type=dataset_type, split="val", data_root=data_root, transform=_val_5cm, test_mode=False, ignore_index=ignore_index),
            dict(type=dataset_type, split="val", data_root=data_root, transform=_val_15cm, test_mode=False, ignore_index=ignore_index),
        ],
        grid_sizes=[0.05, 0.15],
        best_metric_grid_size=0.05,
    ),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
