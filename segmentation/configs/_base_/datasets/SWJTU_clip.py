# dataset settings
dataset_type = 'SWJTUDataset'
#dataset_type = 'ADE20KDataset'
data_root = 'data/SWJTU'
# IMG_MEAN = [ v*255 for v in [0.48145466, 0.4578275, 0.40821073]]  #ade数据集的mean
# IMG_VAR = [ v*255 for v in [0.26862954, 0.26130258, 0.27577711]]

IMG_MEAN = [104.02267, 112.33163 , 95.96643 , 69.79706]
IMG_VAR = [36.396572, 30.301855, 29.188494 ,39.089836]

img_norm_cfg = dict(mean=IMG_MEAN, std=IMG_VAR, to_rgb=False)
crop_size = (256, 256)
#crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged', imdecode_backend='tifffile'),
    dict(type='LoadAnnotations'),
    #dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
    #dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='PhotoMetricDistortion'),#4波段数据无法进行处理
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged', imdecode_backend='tifffile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        #img_scale=(2048, 512),
        #img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='suiji800\img_optical_sar800',
        #img_dir='img_dir/val',
        ann_dir='suiji800/ann800',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        #img_dir='800挑选\img_optical_sar1431',
        #ann_dir='800挑选/ann1431',
        img_dir='suiji800\img_optical_sar1912',
        ann_dir='suiji800/ann1912',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        #img_dir='800挑选\img_optical_sar1431',
        #ann_dir='800挑选/ann1431',
        img_dir='suiji800\img_optical_sar1912',
        ann_dir='suiji800/ann1912',
        pipeline=test_pipeline))
