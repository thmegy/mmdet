_base_ = '../mmdetection/configs/yolox/yolox_s_8x8_300e_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=6
    )
)

# data settings
dataset_type = 'CocoDataset'
classes = ('Arrachement_pelade', 'Faiencage', 'Nid_de_poule', 'Transversale', 'Longitudinale', 'Reparation')
data_root = 'data/cracks/11000/'

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'cracks_train.json',
        img_prefix='/home/finn/DATASET/Logiroad_6_classes_11107',
        classes=classes,
    ),
)

data = dict(
    samples_per_gpu=12,
    workers_per_gpu=6,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'cracks_val.json',
        img_prefix='/home/finn/DATASET/Logiroad_6_classes_11107',
        classes=classes,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'cracks_test.json',
        img_prefix='/home/finn/DATASET/Logiroad_6_classes_11107',
        classes=classes,
    )
)

# add tensorboard
log_config = dict(
    interval=500,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

# training parameters
load_from='checkpoints/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
#optimizer = dict(lr=2e-4 * 1 / 32) # learning rate scaling done automatically with --auto-scale-lr argument
checkpoint_config = dict(interval=20) # save checkpoint every 10 epochs
#evaluation = dict(interval=2, dynamic_intervals=[(285, 5)])
#runner = dict(max_epochs=30)
#lr_config = dict(step=[6])
