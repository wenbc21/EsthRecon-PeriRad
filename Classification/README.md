# SAC - Classification

For Task1 and Task3

## Structure

Standard deep learning workspace

## Workflow

### Step 1: prepare data

Create dataset directory and put raw data in the directory.
Run [make_task1_classification.py](../data/make_task1_classification.py) and [make_task3_classification.py](../data/make_task3_classification.py) to make dataset.

### Step 2: training

Run [train.py](train.py) to train. 

### Step 3: inference

Run [predict.py](predict.py) to inference the testset.

### Step 4: analyse

The results will be stored in results directory after training and inference.
