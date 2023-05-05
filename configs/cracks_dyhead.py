#_base_ = '../mmdetection/configs/dyhead/atss_r50_fpn_dyhead_1x_coco.py'
_base_ = '../mmdetection/configs/dyhead/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=12
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1000, 480), (1000, 600)],
        multiscale_mode='range',
        keep_ratio=True,
        backend='pillow'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=128),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, backend='pillow'),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=128),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


# data settings
dataset_type = 'CocoDataset'
#classes = ('Arrachement_pelade', 'Faiencage', 'Nid_de_poule', 'Transversale', 'Longitudinale', 'Reparation')
classes = ('Arrachement_pelade', 'Faiencage', 'Nid_de_poule', 'Transversale', 'Longitudinale', 'Pontage_de_fissures', 'Remblaiement_de_tranchees', 'Raccord_de_chaussee', 'Comblage_de_trou_ou_Projection_d_enrobe', 'Bouche_a_clef', 'Grille_avaloir', 'Regard_tampon')
data_root = 'data/cracks_12_classes/'
#data = dict(
#    samples_per_gpu=4,
#    workers_per_gpu=2,
#    train=dict(
#        type=dataset_type,
#        ann_file=data_root + 'cracks_train.json',
#        img_prefix='/home/finn/DATASET/ai4cracks-dataset/images/',
#        classes=classes,
##        pipeline=train_pipeline
#    ),
#    val=dict(
#        type=dataset_type,
#        ann_file=data_root + 'cracks_val_test.json',
#        img_prefix='/home/finn/DATASET/ai4cracks-dataset/images/',
#        classes=classes,
##        pipeline=test_pipeline
#    ),
#    test=dict(
#        type=dataset_type,
#        ann_file=data_root + 'cracks_val_test.json',
#        img_prefix='/home/finn/DATASET/ai4cracks-dataset/images/',
#        classes=classes,
##        pipeline=test_pipeline
#    )
#)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'cracks_train.json',
            img_prefix='/home/finn/DATASET/ai4cracks-dataset/images/train/',
            classes=classes,
            pipeline=train_pipeline,
            filter_empty_gt=False
        )
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'cracks_val_test.json',
        img_prefix='/home/finn/DATASET/ai4cracks-dataset/images/val_test/',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'cracks_val_test.json',
        img_prefix='/home/finn/DATASET/ai4cracks-dataset/images/val_test/',
        classes=classes,
        pipeline=test_pipeline
    )
)

# add tensorboard
log_config = dict(
    interval=3000,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

# training parameters
#load_from='pretrained_checkpoints/atss_r50_fpn_dyhead_4x4_1x_coco_20211219_023314-eaa620c6.pth'
load_from='pretrained_checkpoints/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco_20220509_100315-bc5b6516.pth'

#optimizer = dict(lr=2e-4 * 1 / 32) # learning rate scaling done automatically with --auto-scale-lr argument
checkpoint_config = dict(interval=2) # save checkpoint every 10 epochs
evaluation = dict(interval=2)
#runner = dict(max_epochs=16)
#lr_config = dict(step=[32, 44])
