_base_ = '../mmdetection/configs/yolo/yolov3_d53_fp16_mstrain-608_273e_coco.py'

# model settings
model = dict(
    bbox_head=dict(
        num_classes=2,
#        anchor_generator=dict(
#            base_sizes=[[(403, 99), (156, 220), (113, 109)],
#                        [(64, 185), (249, 37), (60, 80)],
#                        [(94, 26), (32, 77), (24, 28)]]
#        )
    )
)

# data settings
dataset_type = 'CocoDataset'
classes = ('plaque', '')
data_root = 'data/'
data = dict(
    samples_per_gpu=20,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'plaques_Antony_train.json',
        img_prefix='data/plaques_de_rue_Antony/',
        classes=classes,
#        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'plaques_Antony_val.json',
        img_prefix='data/plaques_de_rue_Antony/',
        classes=classes,
#        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'plaques_Antony_val.json',
        img_prefix='data/plaques_de_rue_Antony/',
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
load_from='checkpoints/yolov3_d53_fp16_mstrain-608_273e_coco_20210517_213542-4bc34944.pth'
#load_from='checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
checkpoint_config = dict(interval=10) # save checkpoint every 10 epochs
evaluation = dict(interval=10)
# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
optimizer = dict(type='SGD', lr=0.001*40/64, momentum=0.9, weight_decay=0.0005)
#runner = dict(type='EpochBasedRunner', max_epochs=75)