_base_ = '../mmdetection/configs/tood/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=6
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(640, 480), (640, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# data settings
dataset_type = 'CocoDataset'
classes = ('Arrachement_pelade', 'Faiencage', 'Nid_de_poule', 'Transversale', 'Longitudinale', 'Reparation')
data_root = 'data/cracks/10746/'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'cracks_train.json',
        img_prefix='/home/finn/DATASET/CRACKS/Logiroad_10746_images/',
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'cracks_val.json',
        img_prefix='/home/finn/DATASET/CRACKS/Logiroad_10746_images/',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'cracks_test.json',
        img_prefix='/home/finn/DATASET/CRACKS/Logiroad_10746_images/',
        classes=classes,
        pipeline=test_pipeline
    )
)

# add tensorboard
log_config = dict(
    interval=200,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

# training parameters
load_from='checkpoints/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20211210_213728-4a824142.pth'
#optimizer = dict(lr=2e-4 * 1 / 32) # learning rate scaling done automatically with --auto-scale-lr argument
checkpoint_config = dict(interval=5) # save checkpoint every 10 epochs
evaluation = dict(interval=5)
#runner = dict(max_epochs=28)
#lr_config = dict(step=[32, 44])
