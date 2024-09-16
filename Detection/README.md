# SAC - Detection

For Task1

## Structure

mmDetection workspace

## Workflow

### Step 1: prepare data

create dataset directory and put raw data in the directory.
run labelme2coco.py to convert raw data into coco format

### Step 2: mmDetection config

edit configs/* to config the model

### Step 3: mmDetection training

run mmdetection/tools/train.py with the config file to train a model

### Step 4: mmDetection evaluate

run mmdetection/tools/test.py with the config file and trained model file to evaluate the results
