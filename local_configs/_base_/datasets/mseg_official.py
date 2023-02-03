# dataset settings
dataset_type = 'MsegUniversalDataset'
MSEG_DATASETS = [
    'ade20k-150-relabeled',
    'bdd-relabeled',
    'cityscapes-19-relabeled',
    'coco-panoptic-133-relabeled',
    'idd-39-relabeled',
    'mapillary-public65-relabeled',
    'sunrgbd-37-relabeled'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]


datasets_train = [
    dict(
        type=dataset_type,
        split='train',
        dataset_name=dataset
    ) for dataset in MSEG_DATASETS]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=datasets_train,
)
