_base_ = '../mmdetection/configs/dyhead/atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=1
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=[(1000, 480), (1000, 600)],
        keep_ratio=True,
        backend='pillow'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


# data settings
dataset_type = 'CocoDataset'
classes = ('sign')
data_root = 'data/mapillary_traffic_signs/'

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='signs_train.json',
            data_prefix=dict(img='images/'),
            filter_cfg=dict(filter_empty_gt=False),
            metainfo=dict(classes=classes),
            pipeline=train_pipeline
        )
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='signs_val.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        metainfo=dict(classes=classes),
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader  # The configuration of the testing dataloader is the same as that of the validation dataloader, which is omitted here


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'signs_val.json',
    metric=['bbox'],
    format_only=False,
    classwise=True
)
test_evaluator = val_evaluator

# training parameters
train_cfg = dict(
    val_interval=1  # Interval for validation, check the performance every 2
)
load_from='pretrained_checkpoints/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco_20220509_100315-bc5b6516.pth'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto'
    ),
        logger=dict(
            type='LoggerHook',
            interval=5000
    )
)

log_processor = dict(
    type='LogProcessor',
    window_size=50
)

# add tensorboard
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
