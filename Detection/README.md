# SAC - Detection

For Task1 and Task3

## Structure

Detection workspace using mmDetection with COCO format

## Workflow

### Step 1: prepare data

Create dataset directory and put raw data in the directory.
Run [make_task1_detection.py](../data/make_task1_detection.py) and [make_task3_detection.py](../data/make_task3_detection.py) to convert raw data into coco format.

### Step 2: Install mmDetection

Install and config [mmDetection](https://github.com/open-mmlab/mmdetection).
```
git clone https://github.com/open-mmlab/mmdetection.git
```

### Step 3: mmDetection configs

Edit [configs files](configs) to setup the models.

### Step 4: mmDetection training

Run [mmdetection/tools/train.py](mmdetection/tools/train.py) with the config file to train a model.
```
python mmdetection/tools/train.py configs/t3_yolox_s.py
```

### Step 5: mmDetection evaluate

Run [mmdetection/tools/test.py](mmdetection/tools/test.py) with the config file and trained model file to evaluate the results.
```
python mmdetection/tools/test.py configs/t1_efficientdet_effb0.py work_dirs/t1_efficientdet_effb0/epoch_75.pth --show-dir t1_efficientdet_effb0 --out t1_effb0.pkl
```