_base_ = '../mmdetection/configs/dyhead/atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=257
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=[(1000, 480), (1000, 600)],
        keep_ratio=True,
        backend='pillow'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


# data settings
dataset_type = 'CocoDataset'
classes = ('complementary_accident-area', 'complementary_both-directions', 'complementary_buses', 'complementary_chevron-left', 'complementary_chevron-right', 'complementary_chevron-right-unsure', 'complementary_distance', 'complementary_except-bicycles', 'complementary_extent-of-prohibition-area-both-direction', 'complementary_go-left', 'complementary_go-right', 'complementary_keep-left', 'complementary_keep-right', 'complementary_maximum-speed-limit-15', 'complementary_maximum-speed-limit-20', 'complementary_maximum-speed-limit-25', 'complementary_maximum-speed-limit-30', 'complementary_maximum-speed-limit-35', 'complementary_maximum-speed-limit-40', 'complementary_maximum-speed-limit-45', 'complementary_maximum-speed-limit-50', 'complementary_maximum-speed-limit-55', 'complementary_maximum-speed-limit-70', 'complementary_maximum-speed-limit-75', 'complementary_obstacle-delineator', 'complementary_one-direction-left', 'complementary_one-direction-right', 'complementary_pass-right', 'complementary_priority-route-at-intersection', 'complementary_tow-away-zone', 'complementary_trucks', 'complementary_trucks-turn-right', 'complementary_turn-left', 'complementary_turn-right', 'information_airport', 'information_bike-route', 'information_bus-stop', 'information_camp', 'information_central-lane', 'information_children', 'information_dead-end', 'information_dead-end-except-bicycles', 'information_disabled-persons', 'information_emergency-facility', 'information_end-of-built-up-area', 'information_end-of-limited-access-road', 'information_end-of-living-street', 'information_end-of-motorway', 'information_end-of-pedestrians-only', 'information_food', 'information_gas-station', 'information_highway-exit', 'information_highway-interstate-route', 'information_hospital', 'information_interstate-route', 'information_limited-access-road', 'information_living-street', 'information_lodging', 'information_minimum-speed-40', 'information_motorway', 'information_no-parking', 'information_parking', 'information_pedestrians-crossing', 'information_road-bump', 'information_safety-area', 'information_stairs', 'information_telephone', 'information_trailer-camping', 'information_tram-bus-stop', 'other-sign', 'regulatory_bicycles-only', 'regulatory_buses-only', 'regulatory_detour-left', 'regulatory_do-not-block-intersection', 'regulatory_do-not-stop-on-tracks', 'regulatory_dual-lanes-go-straight-on-left', 'regulatory_dual-lanes-go-straight-on-right', 'regulatory_dual-lanes-turn-left-no-u-turn', 'regulatory_dual-lanes-turn-left-or-straight', 'regulatory_dual-lanes-turn-right-or-straight', 'regulatory_dual-path-bicycles-and-pedestrians', 'regulatory_dual-path-pedestrians-and-bicycles', 'regulatory_end-of-bicycles-only', 'regulatory_end-of-buses-only', 'regulatory_end-of-maximum-speed-limit-30', 'regulatory_end-of-maximum-speed-limit-70', 'regulatory_end-of-no-parking', 'regulatory_end-of-priority-road', 'regulatory_end-of-prohibition', 'regulatory_end-of-speed-limit-zone', 'regulatory_give-way-to-oncoming-traffic', 'regulatory_go-straight', 'regulatory_go-straight-or-turn-left', 'regulatory_go-straight-or-turn-right', 'regulatory_height-limit', 'regulatory_keep-left', 'regulatory_keep-right', 'regulatory_lane-control', 'regulatory_left-turn-yield-on-green', 'regulatory_maximum-speed-limit-10', 'regulatory_maximum-speed-limit-100', 'regulatory_maximum-speed-limit-110', 'regulatory_maximum-speed-limit-120', 'regulatory_maximum-speed-limit-15', 'regulatory_maximum-speed-limit-20', 'regulatory_maximum-speed-limit-25', 'regulatory_maximum-speed-limit-30', 'regulatory_maximum-speed-limit-35', 'regulatory_maximum-speed-limit-40', 'regulatory_maximum-speed-limit-45', 'regulatory_maximum-speed-limit-5', 'regulatory_maximum-speed-limit-50', 'regulatory_maximum-speed-limit-55', 'regulatory_maximum-speed-limit-60', 'regulatory_maximum-speed-limit-65', 'regulatory_maximum-speed-limit-70', 'regulatory_maximum-speed-limit-80', 'regulatory_maximum-speed-limit-90', 'regulatory_maximum-speed-limit-led-100', 'regulatory_maximum-speed-limit-led-60', 'regulatory_maximum-speed-limit-led-80', 'regulatory_minimum-safe-distance', 'regulatory_mopeds-and-bicycles-only', 'regulatory_no-bicycles', 'regulatory_no-buses', 'regulatory_no-entry', 'regulatory_no-hawkers', 'regulatory_no-heavy-goods-vehicles', 'regulatory_no-heavy-goods-vehicles-or-buses', 'regulatory_no-left-turn', 'regulatory_no-mopeds-or-bicycles', 'regulatory_no-motor-vehicle-trailers', 'regulatory_no-motor-vehicles', 'regulatory_no-motor-vehicles-except-motorcycles', 'regulatory_no-motorcycles', 'regulatory_no-overtaking', 'regulatory_no-overtaking-by-heavy-goods-vehicles', 'regulatory_no-parking', 'regulatory_no-parking-or-no-stopping', 'regulatory_no-pedestrians', 'regulatory_no-pedestrians-or-bicycles', 'regulatory_no-right-turn', 'regulatory_no-stopping', 'regulatory_no-straight-through', 'regulatory_no-turn-on-red', 'regulatory_no-turns', 'regulatory_no-u-turn', 'regulatory_no-vehicles-carrying-dangerous-goods', 'regulatory_one-way-left', 'regulatory_one-way-right', 'regulatory_one-way-straight', 'regulatory_parking-restrictions', 'regulatory_pass-on-either-side', 'regulatory_passing-lane-ahead', 'regulatory_pedestrians-only', 'regulatory_priority-over-oncoming-vehicles', 'regulatory_priority-road', 'regulatory_radar-enforced', 'regulatory_reversible-lanes', 'regulatory_road-closed', 'regulatory_road-closed-to-vehicles', 'regulatory_roundabout', 'regulatory_shared-path-bicycles-and-pedestrians', 'regulatory_shared-path-pedestrians-and-bicycles', 'regulatory_stop', 'regulatory_stop-here-on-red-or-flashing-light', 'regulatory_stop-signals', 'regulatory_text-four-lines', 'regulatory_triple-lanes-turn-left-center-lane', 'regulatory_truck-speed-limit-60', 'regulatory_turn-left', 'regulatory_turn-left-ahead', 'regulatory_turn-right', 'regulatory_turn-right-ahead', 'regulatory_turning-vehicles-yield-to-pedestrians', 'regulatory_u-turn', 'regulatory_weight-limit', 'regulatory_weight-limit-with-trucks', 'regulatory_width-limit', 'regulatory_wrong-way', 'regulatory_yield', 'warning_accidental-area-unsure', 'warning_added-lane-right', 'warning_bicycles-crossing', 'warning_bus-stop-ahead', 'warning_children', 'warning_crossroads', 'warning_crossroads-with-priority-to-the-right', 'warning_curve-left', 'warning_curve-right', 'warning_dip', 'warning_divided-highway-ends', 'warning_domestic-animals', 'warning_double-curve-first-left', 'warning_double-curve-first-right', 'warning_double-reverse-curve-right', 'warning_double-turn-first-right', 'warning_dual-lanes-right-turn-or-go-straight', 'warning_emergency-vehicles', 'warning_equestrians-crossing', 'warning_falling-rocks-or-debris-right', 'warning_flaggers-in-road', 'warning_hairpin-curve-left', 'warning_hairpin-curve-right', 'warning_height-restriction', 'warning_horizontal-alignment-left', 'warning_horizontal-alignment-right', 'warning_junction-with-a-side-road-acute-left', 'warning_junction-with-a-side-road-acute-right', 'warning_junction-with-a-side-road-perpendicular-left', 'warning_junction-with-a-side-road-perpendicular-right', 'warning_kangaloo-crossing', 'warning_loop-270-degree', 'warning_narrow-bridge', 'warning_offset-roads', 'warning_other-danger', 'warning_pass-left-or-right', 'warning_pedestrian-stumble-train', 'warning_pedestrians-crossing', 'warning_playground', 'warning_railroad-crossing', 'warning_railroad-crossing-with-barriers', 'warning_railroad-crossing-without-barriers', 'warning_railroad-intersection', 'warning_restricted-zone', 'warning_road-bump', 'warning_road-narrows', 'warning_road-narrows-left', 'warning_road-narrows-right', 'warning_road-widens', 'warning_road-widens-right', 'warning_roadworks', 'warning_roundabout', 'warning_school-zone', 'warning_shared-lane-motorcycles-bicycles', 'warning_slippery-motorcycles', 'warning_slippery-road-surface', 'warning_steep-ascent', 'warning_stop-ahead', 'warning_t-roads', 'warning_texts', 'warning_traffic-merges-left', 'warning_traffic-merges-right', 'warning_traffic-signals', 'warning_trail-crossing', 'warning_trams-crossing', 'warning_trucks-crossing', 'warning_turn-left', 'warning_turn-right', 'warning_two-way-traffic', 'warning_uneven-road', 'warning_uneven-roads-ahead', 'warning_wild-animals', 'warning_winding-road-first-left', 'warning_winding-road-first-right', 'warning_wombat-crossing', 'warning_y-roads')

data_root = 'data/mapillary_traffic_signs/'

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='all_classes/signs_train.json',
            data_prefix=dict(img='images/'),
            filter_cfg=dict(filter_empty_gt=False),
            metainfo=dict(classes=classes),
            pipeline=train_pipeline
        )
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='all_classes/signs_val.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        metainfo=dict(classes=classes),
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader  # The configuration of the testing dataloader is the same as that of the validation dataloader, which is omitted here


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'all_classes/signs_val.json',
    metric=['bbox'],
    format_only=False,
    classwise=True
)
test_evaluator = val_evaluator

# training parameters
train_cfg = dict(
    val_interval=1  # Interval for validation, check the performance every 2
)
load_from='pretrained_checkpoints/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco_20220509_100315-bc5b6516.pth'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto'
    ),
        logger=dict(
            type='LoggerHook',
            interval=5000
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
