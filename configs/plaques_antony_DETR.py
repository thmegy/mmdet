_base_ = '../mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=1
    )
)

# data settings
dataset_type = 'CocoDataset'
classes = ('plaque',)
data_root = 'data/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'plaques_Antony_train.json',
        img_prefix='data/plaques_de_rue_Antony/',
        classes=classes,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'plaques_Antony_val.json',
        img_prefix='data/plaques_de_rue_Antony/',
        classes=classes,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'plaques_Antony_val.json',
        img_prefix='data/plaques_de_rue_Antony/',
        classes=classes,
    )
)

# add tensorboard
log_config = dict(
    interval=100,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

# training parameters
#load_from='checkpoints/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth'
optimizer = dict(
    type='AdamW',
    lr=2e-4 * 4 / 32,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }
    )
)
checkpoint_config = dict(interval=10) # save checkpoint every 10 epochs
evaluation = dict(interval=5)
runner = dict(max_epochs=70)
