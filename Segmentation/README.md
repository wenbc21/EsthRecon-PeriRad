# SAC - Segmentation

For Task2

## Structure

[main.py](main.py) - post process, visualization and get final result

## Workflow

### Step 1: prepare data

use labelme to label sementic segmentation for Task2, run [make_task2_dataset.py](../data/make_task2_dataset.py) to automatically build the dataset. 

### Step 2: install nnUNet

follow guide in https://github.com/MIC-DKFZ/nnUNet

### Step 3: nnUNet training

train a nnUNet model using data in dataset/Task2

### Step 4: nnUNet inference

inference images using trained nnUNet model

### Step 5: final classification

put nnUNet results in the directory, run [main.py](main.py) to post process and visualize the results
