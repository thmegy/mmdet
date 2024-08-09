# mmdet

Wrapper for mmdetection


# Setup

Assuming you have already setup the mmdetection environment, as described in the [documentation](https://mmdetection.readthedocs.io/en/v2.20.0/get_started.html#installation)
```
git clone --recursive ssh://git@Gitlab.Logiroad.com:2316/theom/mmdet.git
cd mmdet/mmdetection/
pip install -v -e .
cd ..
```


# Run the code

## Training

You can launch a training on a single GPU with:
```
python mmdetection/tools/train.py configs/<config-file> --work-dir outputs/<dir> --auto-scale-lr
```

You can launch a training on a multiple GPUs with:
```
./mmdetection/tools/dist_train.sh configs/<config-file> <n_GPU> --work-dir outputs/<dir>
```
The '--resume-from' argument, followed by a checkpoint, can also be used to resume a training.


## Inference

To run an inference using a trained model:
```
python scripts/inference.py --config <path-to-config> --checkpoint <path-to-pth> --im-dir <path-to-images> --viz-dir <output-path> --score-threshold <minimum-score>
```


## Prepare data pipeline

In order to prepare the data pipeline, the resize and crop in particular, it might come handy to be aware of the size of the bounding boxes in the labelled dataset.
A distribution of the bbox sizes can be produced with:
```
python scripts/get_bbox_size.py --infile <path-to-coco-annotation-file.json>
```
The number of small (area < 32x32 pixels), medium (32x32 < area < 96x96) and large (area > 96x96) objects is also computed.


## Active Learning (AL)

The AL script `scripts/active_learning.py` has two modes: production and test.
The production mode is used to select images to be labelled, based on a given AL strategy. A typical command is:
```
python scripts/active_learning.py --config <path-to-config> prod --checkpoint <path-to-trained-model> --image-path <path-to-unlabelled-images> --output <path-to-output-file>
```

Active learning procedures can be tested with:
```
python scripts/active_learning.py --config <path-to-config> test --work-dir <output-directory> --n-round <AL-iterations>
```
This script manages the dataset and launches the trainings for each AL iteration. The dataset is expected to have the `CocoDataset` format.  
The full dataset is automatically split between a reduced training set and a pool set, if the sets do not already exist. The number of images in the initial pool set is chosen with `--n-init`. If the the pool and training sets already exist, a new split can be made using the argument `--do-split`.  
The initial training of the AL procedure, using the initial training set, is done automatically if the corresponding output cannot be found. If an output already exists, the training can be made again with the argument `--do-init-train`.  
By default, the detection model is trained from scratch at each AL iteration. However, an online learning strategy has been implemented to speed up things. It is activated with `--incremental-learning`, and fine-tunes the model for `--n-iter` training iterations, starting from the latest checkpoint of the previous AL iteration.

### Config and modification of models

The parameters defining a specific AL strategy are given in the config file. For YOLOv3, it looks like:
```
model = dict(
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=<n-classes>,
        test_cfg = dict(
            active_learning = dict(
		score_method = 'MarginSampling',
                aggregation_method = 'sum',
                selection_method = 'batch',
		n_sel = 1000,
		selection_kwargs = dict(
                    batch_size = 10,
                ),
                alpha = 0.5 # proba for sampler used if incremental learning
            )
        )
    )
)
```
The various score, aggregation and selection methods are implemented in [mmdetection/mmdet/utils/active_learning.py](https://github.com/thmegy/mmdetection/blob/master/mmdet/utils/active_learning.py).

Before using a given detection models in an AL procedure, it is necessary to modify the model's head such that, at inference time, it returns the images uncertainty instead of predicted bounding boxes if an AL procedure is ongoing.  
For YOLOv3, the modifications can be found in [mmdetection/mmdet/models/dense_heads/yolo_head.py](https://github.com/thmegy/mmdetection/blob/master/mmdet/models/dense_heads/yolo_head.py#L209)

### Plotting

Summary plots for AL tests can be produced with `scripts/plot_active_learning.py`.


## Interpretability with D-RISE

Use D-RISE method to convert detection annotations in yolo format into segmentation masks with `scripts/convert_det_to_seg.py`.
Explain results of a model with `scripts/explain_image_D_RISE.py`.

## Modify annotations

Turn detection annotations into classification annotations by cropping bboxes with `scripts/convert_yolo_to_cls.py`.  
Modifiers for diagonal defects in cracks dataset:
- add overlapping bboxes with `scripts/make_overlap_bboxes.py`;
- use trained model to infer smaller bboxes, within annotated bboxes, with `scripts/make_smaller_bboxes.py`.