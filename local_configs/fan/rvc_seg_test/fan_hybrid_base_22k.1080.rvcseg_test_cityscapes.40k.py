# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FAN/blob/main/LICENSE

_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_40k_8gpu_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='../work_dirs/official_weights/fan_hybrid_base_in22k_1k.pth.tar',
    backbone=dict(
        type='fan_base_16_p4_hybrid',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[128, 256, 448, 448],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=256,
        dropout_ratio=0.1,
        num_classes=256,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

dataset_type = 'CustomDataset'

RVCSeg_DATASETS = [
    'ade20k-151',
    'cityscapes-34',
    'coco-panoptic-201',
    'kitti-34',
    'mapillary-public66',
    'scannet-41',
    'viper-rvc-32',
    'wilddash2-rvc-39'
]

RVCSeg_img_suffixes = {
    'ade20k-151': '.jpg',
    'cityscapes-34': '.png',
    'coco-panoptic-201': '.jpg',
    'kitti-34': '.png',
    'mapillary-public66': '.jpg',
    'scannet-41': '.jpg',
    'viper-rvc-32': '.jpg',
    'wilddash2-rvc-39': '.jpg'
}


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
        pipeline=train_pipeline,
        data_root=None,
        img_dir=f'data/rvc_seg/rvc_uint8/images/train/{dataset}/',
        ann_dir=f'data/rvc_seg/rvc_uint8/annotations/train/{dataset}/',
        img_suffix=RVCSeg_img_suffixes[dataset],
        seg_map_suffix='.png',
    ) for dataset in RVCSeg_DATASETS]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


test_dataset = 'cityscapes-34'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    # train=datasets_train,
    test=dict(
        type='RVC_SEG_Dataset',
        pipeline=test_pipeline,
        data_root=None,
        img_dir=f'data/rvc_seg/rvc_uint8/images/val/{test_dataset}/',
        ann_dir=f'data/rvc_seg/rvc_uint8/annotations/val/{test_dataset}/',
        img_suffix=RVCSeg_img_suffixes[test_dataset],
        remap_rvc_dataset=test_dataset,
        rvc_seg_mapping_tsv='/saccadenet/jfxiao/codes/rvc_devkit/segmentation/ss_mapping_uint8_mseg.tsv',
        seg_map_suffix='.png',
    )
)



# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# uncomment to use fp16 training
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 = dict()
# fp16 = dict(loss_scale='dynamic')
