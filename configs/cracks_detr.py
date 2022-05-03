_base_ = '../mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=6
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
                    'crop_size': (800, 1200),
                    'allow_negative_crop': False
                },
                {
                    'type':'Resize',
                    'img_scale': [(400, 700), (500, 800), (600, 900), (700, 1000),
                                  (800, 1200), (900, 800), (1000, 800), (800, 600),
                                  (700, 800), (1000, 1300), (400, 600)],
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
                    'img_scale': [(800, 900), (900, 1200), (1000, 1200), (1100, 1200), (600, 700), (800, 1200),
                                  (800, 1000), (600, 1000), (1000, 1300), (900, 1300), (700, 1200)],
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
        img_scale=(800, 1200),
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
classes = ('Arrachement_pelade', 'Faiencage', 'Nid_de_poule', 'Transversale', 'Longitudinale', 'Reparation')
data_root = 'data/cracks/11000/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'cracks_train.json',
        img_prefix='/home/finn/DATASET/Logiroad_6_classes_11107',
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'cracks_val.json',
        img_prefix='/home/finn/DATASET/Logiroad_6_classes_11107',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'cracks_test.json',
        img_prefix='/home/finn/DATASET/Logiroad_6_classes_11107',
        classes=classes,
        pipeline=test_pipeline
    )
)

# add tensorboard
log_config = dict(
    interval=1000,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

# training parameters
load_from='checkpoints/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth'
#optimizer = dict(lr=2e-4 * 1 / 32) # learning rate scaling done automatically with --auto-scale-lr argument
checkpoint_config = dict(interval=5) # save checkpoint every 10 epochs
evaluation = dict(interval=5)
runner = dict(max_epochs=100)
lr_config = dict(step=[80])
