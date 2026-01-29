norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='DenseCLIP',
    pretrained='pretrained/ViT-B-16.pt',
    context_length=69,
    backbone=dict(
        type='CLIPVisionTransformer',
        layers=12,
        style='pytorch',
        patch_size=16,
        width=768,
        output_dim=512,
        get_embeddings=True,
        drop_path_rate=0.1,
        input_resolution=256),
    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=77,
        style='pytorch',
        embed_dim=512,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12),
    context_decoder=dict(
        type='ContextDecoder',
        context_length=16,
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=512,
        dropout=0.1,
        outdim=512,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[768, 768, 775, 768],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=256,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    identity_head=dict(
        type='IdentityHead',
        in_channels=1,
        channels=1,
        num_classes=1,
        dropout_ratio=0.1,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
    text_head=False,
    text_dim=512,
    score_concat_index=2)
dataset_type = 'SWJTUDataset'
data_root = 'data/SWJTU'
IMG_MEAN = [98.86502, 97.23819, 95.2317, 40.737988]
IMG_VAR = [27.610954, 24.760305, 23.057755, 27.252533]
img_norm_cfg = dict(
    mean=[98.86502, 97.23819, 95.2317, 40.737988],
    std=[27.610954, 24.760305, 23.057755, 27.252533],
    to_rgb=False)
crop_size = (256, 256)
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='unchanged',
        imdecode_backend='tifffile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=[98.86502, 97.23819, 95.2317, 40.737988],
        std=[27.610954, 24.760305, 23.057755, 27.252533],
        to_rgb=False),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='unchanged',
        imdecode_backend='tifffile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[98.86502, 97.23819, 95.2317, 40.737988],
                std=[27.610954, 24.760305, 23.057755, 27.252533],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=10,
    workers_per_gpu=4,
    train=dict(
        type='SWJTUDataset',
        data_root='data/SWJTU',
        img_dir='800挑选\img_optical_sar800',
        ann_dir='800挑选/ann800',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='unchanged',
                imdecode_backend='tifffile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
            dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Normalize',
                mean=[98.86502, 97.23819, 95.2317, 40.737988],
                std=[27.610954, 24.760305, 23.057755, 27.252533],
                to_rgb=False),
            dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='SWJTUDataset',
        data_root='data/SWJTU',
        img_dir='800挑选\img_optical_sar1431',
        ann_dir='800挑选/ann1431',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='unchanged',
                imdecode_backend='tifffile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[98.86502, 97.23819, 95.2317, 40.737988],
                        std=[27.610954, 24.760305, 23.057755, 27.252533],
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='SWJTUDataset',
        data_root='data/SWJTU',
        img_dir='800挑选\img_optical_sar1431',
        ann_dir='800挑选/ann1431',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='unchanged',
                imdecode_backend='tifffile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[98.86502, 97.23819, 95.2317, 40.737988],
                        std=[27.610954, 24.760305, 23.057755, 27.252533],
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
find_unused_parameters = True
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            text_encoder=dict(lr_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-06,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06)
runner = dict(type='EpochBasedRunner', max_epochs=500)
checkpoint_config = dict(by_epoch=True, interval=20)
evaluation = dict(
    interval=20, metric=['mIoU', 'mDice', 'mFscore'], save_best='mIoU')
work_dir = './work_dirs\denseclip_fpn_vit-b_256x256_100epoches'
gpu_ids = range(0, 1)
