# SAC - Detection

Using object detection to do image ROI cropping For Task1 and Task3
Detection workspace using mmDetection and COCO format

### Step 1: prepare data

create dataset directory and put raw data in the directory.
run [data/make_task1_detection.py](../data/make_task1_detection.py) and [make_task3_detection](../data/make_task3_detection.py) to convert raw data into coco format

### Step 2: mmDetection install

install [mmDetection](https://github.com/open-mmlab/mmdetection) following the [guide documents](https://mmdetection.readthedocs.io/zh-cn/latest/get_started.html)

### Step 3: mmDetection config

edit [configs/*](configs/t1_yolox_s.py) ets. to config the model

### Step 4: mmDetection training

run [mmdetection/tools/train.py](mmdetection/tools/train.py) with the config file to train a model
```
python mmdetection/tools/train.py configs/t1_yolox_s.py
```

### Step 5: mmDetection inference

run [mmdetection/tools/test.py](mmdetection/tools/test.py) with the config file and trained model file to inference the results
```
python mmdetection/tools/test.py configs/t1_yolox_s.py work_dirs/t1_yolox_s/epoch_100.pth --show-dir yolox_s
```