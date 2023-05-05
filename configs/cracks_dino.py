_base_ = '../mmdetection/configs/dino/dino-5scale_swin-l_8xb2-12e_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=12
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [{
            'type':
            'RandomChoiceResize',
            'scales': [(480, 1000), (512, 1000), (544, 1000), (576, 1000),
                       (608, 1000), (640, 1000), (672, 1000), (700, 1000)],
            'keep_ratio':
            True
            }],
        ]
    ),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1000, 700), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


# data settings
dataset_type = 'CocoDataset'
classes = ('Arrachement_pelade', 'Faiencage', 'Nid_de_poule', 'Transversale', 'Longitudinale', 'Pontage_de_fissures', 'Remblaiement_de_tranchees', 'Raccord_de_chaussee', 'Comblage_de_trou_ou_Projection_d_enrobe', 'Bouche_a_clef', 'Grille_avaloir', 'Regard_tampon')
data_root = 'data/cracks_12_classes/'

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='cracks_train.json',
        data_prefix=dict(img='/home/finn/DATASET/ai4cracks-dataset/images/train/'),
        filter_cfg=dict(filter_empty_gt=False),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='cracks_val_test.json',
        data_prefix=dict(img='/home/finn/DATASET/ai4cracks-dataset/images/val_test/'),
        test_mode=True,
        metainfo=dict(classes=classes),
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader  # The configuration of the testing dataloader is the same as that of the validation dataloader, which is omitted here


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'cracks_val_test.json',
    metric=['bbox'],
    format_only=False,
    classwise=True
)
test_evaluator = val_evaluator

# training parameters
optim_wrapper = dict(
    optimizer=dict(lr=0.0001 * 1 / 16)
)
train_cfg = dict(
    val_interval=1  # Interval for validation, check the performance every 2
)

load_from='pretrained_checkpoints/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth'


default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto'
    ),
        logger=dict(
            type='LoggerHook',
            interval=1500
    )
)

log_processor = dict(
    type='LogProcessor',
    window_size=50
)

# add tensorboard
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
