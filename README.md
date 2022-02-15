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