# SAC - Detection

For Task1

## Structure

Detection workspace using mmDetection and COCO format

## Workflow

### Step 1: prepare data

create dataset directory and put raw data in the directory.
run [make_task1_dataset.py](../data/make_task1_dataset.py) to convert raw data into coco format

### Step 2: install mmDetection

follow guide in https://github.com/open-mmlab/mmdetection

### Step 3: mmDetection training

run [mmdetection/tools/train.py](mmdetection/tools/train.py) with the config file to train a model
```
python mmdetection/tools/train.py configs/yolox_s_8xb8-200e_coco.py
```

### Step 4: mmDetection evaluate

run [mmdetection/tools/test.py](mmdetection/tools/test.py) with the config file and trained model file to evaluate the results
```
python mmdetection/tools/test.py configs/yolox_s_8xb8-200e_coco.py work_dirs/yolox_s_8xb8-200e_coco/epoch_200.pth --show-dir yolox_s_200
```
