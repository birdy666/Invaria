_base_ = ["../_base_/default_runtime.py"]
seed = 3701824
batch_size = 36
num_worker = 24
mix_prob = 0.8
empty_cache = False
enable_amp = True

model = dict(
    type="DefaultSegmentorV2",
    num_classes=20,
    backbone_out_channels=64,
    ignore_index=-1,
    level_loss_enabled=False,
    bottleneck_consistency_weight=0.1,
    n0=4096,
    n0_scale=4,
    num_passes=3,
    loss_weights=[0.3, 0.6, 1.0],
    eval_pass="adaptive",
    backbone=dict(
        type="Invaria-v3",
        in_channels=6,
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
        grid_size_samples=100,
        grid_size_scale=1,
        predictor_mode="nn",
        predictor_nn_blocks=2,
        predictor_nn_kernel_size=3,
        attn_norm="ln",
        other_norm="ln",
        verbose=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

epoch = 800
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

dataset_type = "ScanNetDataset"
data_root = "data/scannet"

_aug_pre = [
    dict(type="CenterShift", apply_z=True),
    dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
    dict(type="ChromaticJitter", p=0.95, std=0.05),
]

_aug_post = [
    dict(type="SphereCrop", point_max=102400, mode="random"),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "segment"),
        feat_keys=("color", "normal"),
    ),
]

_train_2cm = _aug_pre + [dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train", return_grid_coord=True)] + _aug_post

_val_collect = [
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "segment", "origin_segment", "inverse"),
        feat_keys=("color", "normal"),
    ),
]

_val_1cm = [dict(type="CenterShift", apply_z=True), dict(type="Copy", keys_dict={"segment": "origin_segment"}), dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True)] + _val_collect
_val_2cm = [dict(type="CenterShift", apply_z=True), dict(type="Copy", keys_dict={"segment": "origin_segment"}), dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True)] + _val_collect
_val_4cm = [dict(type="CenterShift", apply_z=True), dict(type="Copy", keys_dict={"segment": "origin_segment"}), dict(type="GridSample", grid_size=0.04, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True)] + _val_collect
_val_6cm = [dict(type="CenterShift", apply_z=True), dict(type="Copy", keys_dict={"segment": "origin_segment"}), dict(type="GridSample", grid_size=0.06, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True)] + _val_collect

data = dict(
    num_classes=20,
    ignore_index=-1,
    names=[
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refridgerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "otherfurniture",
    ],
    train=dict(
        type="ConcatDataset",
        datasets=[
            dict(type=dataset_type, split="train", data_root=data_root, transform=_train_2cm, test_mode=False),
        ],
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=_val_2cm,
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("color", "normal"),
                ),
            ],
            aug_transform=[[{
                'type': 'RandomRotateTargetAngle',
                'angle': [0],
                'axis': 'z',
                'center': [0, 0, 0],
                'p': 1
            }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [0.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }]])))

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="MultiDensitySemSegEvaluator",
        val_datasets=[
            dict(type=dataset_type, split="val", data_root=data_root, transform=_val_1cm, test_mode=False),
            dict(type=dataset_type, split="val", data_root=data_root, transform=_val_2cm, test_mode=False),
            dict(type=dataset_type, split="val", data_root=data_root, transform=_val_6cm, test_mode=False),
        ],
        grid_sizes=[0.01, 0.02, 0.06],
        best_metric_grid_size=0.02,
    ),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
