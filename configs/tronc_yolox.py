_base_ = '../mmdetection/configs/yolox/yolox_s_8x8_300e_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=1
    )
)

# data settings
dataset_type = 'CocoDataset'
classes = ('tronc',)
data_root = 'data/tronc/'

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'imagettes',
        classes=classes,
    ),
)

data = dict(
    samples_per_gpu=12,
    workers_per_gpu=4,
    train=train_dataset,
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
    interval=6,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

# training parameters
load_from='checkpoints/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
#optimizer = dict(lr=2e-4 * 1 / 32) # learning rate scaling done automatically with --auto-scale-lr argument
checkpoint_config = dict(interval=20) # save checkpoint every 10 epochs
evaluation = dict(interval=10, dynamic_intervals=[(285, 5)])
runner = dict(max_epochs=400)
#lr_config = dict(step=[80])
