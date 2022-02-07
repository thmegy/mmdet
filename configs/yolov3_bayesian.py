_base_ = '../mmdetection/configs/yolo/yolov3_d53_fp16_mstrain-608_273e_coco.py'

# import modified neck, with dropout layers
custom_imports = dict(
    imports=['mmdet.models.necks.yolo_neck_bayesian'],
    allow_failed_imports=False
)

# model settings
model = dict(
    backbone=dict(
        init_cfg=None
    ),
    neck=dict(
        type='YOLOV3BayesianNeck'
    ),
    bbox_head=dict(
        num_classes=6,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True, # False to use softmax
            loss_weight=1.0,
            reduction='sum'
        )   
    )
)

# data settings
dataset_type = 'CocoDataset'
classes = ('Arrachement_pelade', 'Faiencage', 'Nid_de_poule', 'Transversale', 'Longitudinale', 'Reparation')
data_root = 'data/'
data = dict(
    samples_per_gpu=20,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'cracks_train.json',
        img_prefix='/home/finn/DATASET/Logiroad_6_classes_7231/',
        classes=classes,
#        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'cracks_val.json',
        img_prefix='/home/finn/DATASET/Logiroad_6_classes_7231/',
        classes=classes,
#        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'cracks_val.json',
        img_prefix='/home/finn/DATASET/Logiroad_6_classes_7231/',
        classes=classes,
#        pipeline=test_pipeline
    )
)

# add tensorboard
log_config = dict(
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

# training parameters
#load_from='checkpoints/yolov3_d53_fp16_mstrain-608_273e_coco_20210517_213542-4bc34944.pth'
#load_from='checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
load_from='outputs/yolov3/epoch_50.pth'
checkpoint_config = dict(interval=25) # save checkpoint every 10 epochs
evaluation = dict(interval=10)
# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
optimizer = dict(type='SGD', lr=0.001*44/64, momentum=0.9, weight_decay=0.0005)
runner = dict(type='EpochBasedRunner', max_epochs=223) # start from 50th epoch of baseline yolo --> 273 - 50 epochs
