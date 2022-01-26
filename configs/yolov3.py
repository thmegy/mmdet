_base_ = '../mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py'

# model settings
model = dict(
    bbox_head=dict(
        num_classes=6
    )
)

# data settings
dataset_type = 'CocoDataset'
classes = ('Arrachement_pelade', 'Faiencage', 'Nid_de_poule', 'Transversale', 'Longitudinale', 'Reparation')
data_root = 'data/cracks/'
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_train.json',
        img_prefix=data_root + 'yolo/',
        classes=classes,
#        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_val.json',
        img_prefix=data_root + 'yolo/',
        classes=classes,
#        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_val.json',
        img_prefix=data_root + 'yolo/',
        classes=classes,
#        pipeline=test_pipeline
    )
)

# add tensorboard
log_config = dict(
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

# training parameters
#load_from='checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
load_from='outputs/yolov3/latest.pth'
# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
checkpoint_config = dict(interval=10) # save checkpoint every 10 epochs
evaluation = dict(interval=5)
optimizer = dict(type='SGD', lr=0.001/8, momentum=0.9, weight_decay=0.0005)
runner = dict(type='EpochBasedRunner', max_epochs=25)
