_base_ = '../mmdetection/configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=13
    )
)


# data settings
dataset_type = 'CocoDataset'
classes = ('Déformation', 'Faïençage', 'Émergence dégradée', 'Glaçage - Ressuage' , 'Arrachement', 'Nid de poule', 'Fissure transversale', 'Fissure longitudinal', 'Fissure en dalles', 'Réparation en BB', 'Autre réparation', 'Tranchée longitudinale', 'Tranchée transversale')
data_root = 'data/crack_instance_segmentation/'


train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='cracks_annotations_full.json',
        backend_args=None,
        data_prefix=dict(img='images'),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        )
    )

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='cracks_annotations_full.json',
        backend_args=None,
        data_prefix=dict(img='images/'),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        )
    )


test_dataloader = val_dataloader  # The configuration of the testing dataloader is the same as that of the validation dataloader, which is omitted here


val_evaluator = dict(
    ann_file='data/crack_instance_segmentation/cracks_annotations_full.json',
)

test_evaluator = dict(
    ann_file='data/crack_instance_segmentation/cracks_annotations_full.json',
)


# training parameters
train_cfg = dict(
    val_interval=1  # Interval for validation, check the performance every epoch
)

load_from='pretrained_checkpoints/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto'
    ),
        logger=dict(
            type='LoggerHook',
            interval=250
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
