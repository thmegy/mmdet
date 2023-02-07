_base_ = '../mmdetection/configs/tood/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py'

model = dict(
    bbox_head=dict(
        type='TOODHeadLPM',
        num_classes=20,
        initial_loss_cls=dict(reduction='none'), # per-event loss (instead of per-batch)
        loss_cls=dict(reduction='none'), # per-event loss (instead of per-batch) 
        loss_bbox=dict(reduction='none'),
    ),
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
#        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test/voc07_test.json',
        img_prefix=data_root + 'test/',
        classes=classes,
#        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/voc07_test.json',
        img_prefix=data_root + 'test/',
        classes=classes,
#        pipeline=test_pipeline
    )
)

# add tensorboard
log_config = dict(
    interval=500,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

# training parameters
load_from='checkpoints/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20211210_213728-4a824142.pth'
#optimizer = dict(lr=2e-4 * 1 / 32) # learning rate scaling done automatically with --auto-scale-lr argument
checkpoint_config = dict(interval=5) # save checkpoint every 10 epochs
evaluation = dict(interval=5)
#runner = dict(max_epochs=28)
#lr_config = dict(step=[32, 44])
