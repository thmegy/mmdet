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
python mmdetection/tools/train.py configs/<config-file> --work-dir outputs/<dir>
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
The number of small (area < 32*32 pixels), medium (32*32 < area < 96*96) and large (area > 96*96) objects is also computed.