_base_ = '../mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=2,
    ),
    test_cfg = dict(
        active_learning = dict(
            score_method = 'Entropy',
            aggregation_method = 'sum',
            selection_method = 'maximum',
            n_sel = 1000,
            selection_kwargs = dict(
                batch_size = 10,
            ),
            alpha = 0.5 # proba for sampler used if incremental learning
        )
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    dict(type='RandomFlip', flip_ratio=0.5, direction='diagonal'),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='AutoAugment',
        policies=[[{
            'type':
            'Resize',
            'img_scale': [(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                          (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                          (736, 1333), (768, 1333), (800, 1333)],
            'multiscale_mode':
            'value',
            'keep_ratio':
            True
        }],
                  [{
                      'type': 'Resize',
                      'img_scale': [(400, 4200), (500, 4200), (600, 4200)],
                      'multiscale_mode': 'value',
                      'keep_ratio': True
                  }, {
                      'type': 'RandomCrop',
                      'crop_type': 'absolute_range',
                      'crop_size': (384, 600),
                      'allow_negative_crop': True
                  }, {
                      'type':
                      'Resize',
                      'img_scale': [(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                      'multiscale_mode':
                      'value',
                      'override':
                      True,
                      'keep_ratio':
                      True
                  }]]),
    dict(type='CutOut', n_holes=[1, 2, 3], cutout_ratio=0.15),
    dict(type='RandomAffine'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]


# data settings
dataset_type = 'CocoDataset'
classes = ('Glasurbild', 'defaut')
data_root = 'data/optomachines/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix='data/optomachines/ImagesOPTOMACHINES-ASSIETTES/',
        classes=classes,
#        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val.json',
        img_prefix='data/optomachines/ImagesOPTOMACHINES-ASSIETTES/',
        classes=classes,
#        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val.json',
        img_prefix='data/optomachines/ImagesOPTOMACHINES-ASSIETTES/',
        classes=classes,
#        pipeline=test_pipeline
    )
)

# add tensorboard
log_config = dict(
    interval=20,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

# training parameters
load_from='checkpoints/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth'
#optimizer = dict(lr=2e-4 * 1 / 32) # learning rate scaling done automatically with --auto-scale-lr argument
checkpoint_config = dict(interval=5) # save checkpoint every 10 epochs
evaluation = dict(interval=5)
runner = dict(max_epochs=150)
lr_config = dict(step=[120])
