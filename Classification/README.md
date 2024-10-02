# SAC - Classification

For Task1 and Task3

## Structure

Standard deep learning workspace

## Workflow

### Step 1: prepare data

create dataset directory and put raw data in the directory, run [make_task3_dataset.py](../data/make_task3_dataset.py) to automatically build the dataset. 

### Step 2: training

run train.py to train. Experiments show that pretrained models are a boost to the performance, so you may create a pretrain directory and download pretrained models following the links in model files (E.g model/ResNet.py, line 169)

### Step 3: inference

run predict.py to inference the testset

### Step 4: analyse

the results will be stored in results directory after training and inference
