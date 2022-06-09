_base_ = '../mmdetection/configs/yolo/yolov3_d53_fp16_mstrain-608_273e_coco.py'

# model settings
model = dict(
    type='YOLOV3LPM',
    bbox_head=dict(
        type='YOLOV3HeadLPM',
        num_classes=20,
        loss_cls=dict(reduction='none'), # per-event loss (instead of per-batch)
        loss_conf=dict(reduction='none'),
        loss_xy=dict(reduction='none'),
        loss_wh=dict(reduction='none'),
        test_cfg = dict(
            active_learning = dict(
                score_method = 'LossPrediction', # just for name
                aggregation_method = 'none', # just for name
                selection_method = 'maximum',
                n_sel = 1000,
                selection_kwargs = dict(
                    batch_size = 10,
                ),
                alpha = 0.5 # proba for sampler used if incremental learning
            )
        )
    )
)

# data settings
dataset_type = 'CocoDataset'
data_root = 'data/VOC/'
classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
data = dict(
    samples_per_gpu=20,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/voc0712_trainval.json',
        img_prefix=data_root + 'train/',
        classes=classes,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test/voc07_test.json',
        img_prefix=data_root + 'test/',
        classes=classes,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/voc07_test.json',
        img_prefix=data_root + 'test/',
        classes=classes,
    )
)

# add tensorboard
log_config = dict(
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")],
    interval=50
)

# training parameters
load_from='checkpoints/yolov3_d53_fp16_mstrain-608_273e_coco_20210517_213542-4bc34944.pth'
checkpoint_config = dict(interval=50) # save checkpoint every 25 epochs
evaluation = dict(interval=10, metric='bbox')
# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
optimizer = dict(type='SGD', lr=0.001*20/64, momentum=0.9, weight_decay=0.0005)
#runner = dict(type='EpochBasedRunner', max_epochs=5)
