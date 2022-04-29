_base_ = '../mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=1
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                {
                    'type': 'RandomCrop',
                    'crop_size': (1250, 600),
                    'allow_negative_crop': False
                },
                {
                    'type':'Resize',
                    'img_scale': [(1250, 480), (1250, 512), (1250, 544), (1250, 576),
                                  (1250, 608), (1250, 640), (1250, 672), (1250, 704),
                                  (1250, 736), (1250, 768), (1250, 800)],
                    'multiscale_mode': 'value',
                    'keep_ratio': True
                },
                {
                    'type': 'Rotate',
                    'level': 2,
                    'img_fill_val': 0
                }
            ],
            [
                {
                    'type':'Resize',
                    'img_scale': [(800, 800), (900, 900), (1000, 1000), (1100, 1100),
                                  (1150, 1100), (1250, 1200), (1250, 1250), (1200, 1200), (1300, 1300)],
                    'multiscale_mode': 'value',
                    'keep_ratio': True
                },
                {
                    'type': 'Rotate',
                    'level': 2,
                    'img_fill_val': 0
                }
            ],
            [
                {
                    'type': 'RandomCrop',
                    'crop_size': (600, 1250),
                    'allow_negative_crop': False
                },
                {
                    'type':'Resize',
                    'img_scale': [(480, 1250), (512, 1250), (544, 1250), (576, 1250),
                                  (608, 1250), (640, 1250), (672, 1250), (704, 1250),
                                  (736, 1250), (768, 1250), (800, 1250)],
                    'multiscale_mode': 'value',
                    'keep_ratio': True
                },
                {
                    'type': 'Rotate',
                    'level': 2,
                    'img_fill_val': 0
                }
            ],
        ]
    ),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1300, 1300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# data settings
dataset_type = 'CocoDataset'
classes = ('tronc',)
data_root = 'data/tronc/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'imagettes',
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'imagettes',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'imagettes',
        classes=classes,
        pipeline=test_pipeline
    )
)

# add tensorboard
log_config = dict(
#    interval=25,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

# training parameters
load_from='checkpoints/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth'
#optimizer = dict(lr=2e-4 * 1 / 32) # learning rate scaling done automatically with --auto-scale-lr argument
checkpoint_config = dict(interval=20) # save checkpoint every 10 epochs
evaluation = dict(interval=10)
runner = dict(max_epochs=100)
lr_config = dict(step=[80])
