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
    samples_per_gpu=1,
    workers_per_gpu=2,
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
    interval=500,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

# training parameters
load_from='checkpoints/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth'
#optimizer = dict(lr=2e-4 * 1 / 32) # learning rate scaling done automatically with --auto-scale-lr argument
checkpoint_config = dict(interval=20) # save checkpoint every 10 epochs
evaluation = dict(interval=20)
#runner = dict(max_epochs=150)
#lr_config = dict(step=[120])
