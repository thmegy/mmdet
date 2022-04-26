_base_ = '../mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=1
    )
)

# data settings
dataset_type = 'CocoDataset'
classes = ('tronc',)
data_root = 'data/tronc/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'imagettes',
        classes=classes,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'imagettes',
        classes=classes,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'imagettes',
        classes=classes,
    )
)

# add tensorboard
log_config = dict(
#    interval=100,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

# training parameters
#load_from='checkpoints/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth'
optimizer = dict(lr=2e-4 * 1 / 32)
checkpoint_config = dict(interval=10) # save checkpoint every 10 epochs
evaluation = dict(interval=5)
#runner = dict(max_epochs=70)
