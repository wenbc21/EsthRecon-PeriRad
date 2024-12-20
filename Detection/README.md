# SAC - Detection

For Task1

## Structure

Detection workspace using mmDetection and COCO format

## Workflow

### Step 1: prepare data

create dataset directory and put raw data in the directory.
run [labelme_to_coco.py](labelme_to_coco.py) to convert raw data into coco format

### Step 2: mmDetection config

edit configs/* to config the model

### Step 3: mmDetection training

run [mmdetection/tools/train.py](mmdetection/tools/train.py) with the config file to train a model
```
python mmdetection/tools/train.py configs/t3_yolox_s.py
```

### Step 4: mmDetection evaluate

run [mmdetection/tools/test.py](mmdetection/tools/test.py) with the config file and trained model file to evaluate the results
```
python mmdetection/tools/test.py configs/t3_yolox_s.py work_dirs/t3_yolox_s/epoch_100.pth --show-dir yolox_s_100