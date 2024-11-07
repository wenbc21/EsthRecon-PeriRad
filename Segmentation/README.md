# SAC - Segmentation

For Task2

## Structure

[json_to_dataset](json_to_dataset.py) - transform labelme data to dataset

[main.py](main.py) - post process, visualization and get final result

## Workflow

### Step 1: prepare data

use labelme to label sementic segmentation for Task2, run [json_to_dataset](json_to_dataset.py) to automatically build the dataset. 

### Step 2: nnUNet training

train a nnUNet model using data in nnUNet_raw

### Step 3: nnUNet inference

inference images in testset using trained nnUNet model

### Step 4: final classification

put nnUNet results in the directory, run [main.py](main.py) to post process and visualize the results
