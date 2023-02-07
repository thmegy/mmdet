_base_ = '../mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=20,
    ),
    test_cfg = dict(
        active_learning = dict(
            score_method = 'Entropy',
            aggregation_method = 'sum',
            selection_method = 'random',
            n_sel = 1000,
            selection_kwargs = dict(
                batch_size = 15,
            ),
            alpha = 0.5 # proba for sampler used if incremental learning
        )
    )
)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[{
            'type':
            'Resize',
            'img_scale': [(480, 1000), (512, 1000), (544, 1000), (576, 1000),
                          (608, 1000), (640, 1000), (672, 1000), (704, 1000),
                          (736, 1000), (768, 1000), (800, 1000)],
            'multiscale_mode':
            'value',
            'keep_ratio':
            True
        }],
#                  [{
#                      'type': 'Resize',
#                      'img_scale': [(400, 4200), (500, 4200), (600, 4200)],
#                      'multiscale_mode': 'value',
#                      'keep_ratio': True
#                  }, {
#                      'type': 'RandomCrop',
#                      'crop_type': 'absolute_range',
#                      'crop_size': (384, 600),
#                      'allow_negative_crop': True
#                  }, {
#                      'type':
#                      'Resize',
#                      'img_scale': [(480, 1000), (512, 1000), (544, 1000),
#                                    (576, 1000), (608, 1000), (640, 1000),
#                                    (672, 1000), (704, 1000), (736, 1000),
#                                    (768, 1000), (800, 1000)],
#                      'multiscale_mode':
#                      'value',
#                      'override':
#                      True,
#                      'keep_ratio':
#                      True
#                  }]
        ]),
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
        img_scale=(1000, 800),
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
data_root = 'data/VOC/'
classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/voc0712_trainval.json',
        img_prefix=data_root + 'train/',
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test/voc07_test.json',
        img_prefix=data_root + 'test/',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/voc07_test.json',
        img_prefix=data_root + 'test/',
        classes=classes,
        pipeline=test_pipeline
    )
)

# add tensorboard
log_config = dict(
    interval=500,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

# training parameters
load_from='pretrained_checkpoints/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth'
#optimizer = dict(lr=2e-4 * 1 / 32) # learning rate scaling done automatically with --auto-scale-lr argument
checkpoint_config = dict(interval=20) # save checkpoint every 10 epochs
evaluation = dict(interval=20)
runner = dict(max_epochs=20)
lr_config = dict(step=[16])
