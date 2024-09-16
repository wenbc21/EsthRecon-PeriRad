# SAC - Segmentation

For Task2

## Structure

labelme2nnunet - data preprocess for nnUNet training

test2nnunet - data preprocess for nnUNet inference

postprocess - post process for nnUNet results

## Workflow

### Step 1: prepare data

use labelme to label sementic segmentation for Task2, use '''labelme_json_to_dataset''' to automatically build the dataset. 

put raw testset file in the directory.

### Step 2: transform data into nnUNet format

run labelme2nnunet.py and test2nnunet.py, the results will be stored in nnUNet_raw and nnUNet_test directory.

### Step 3: nnUNet training

train a nnUNet model using data in nnUNet_raw

### Step 4: nnUNet inference

inference images in nnUNet_test using trained nnUNet model

### Step 5: postprocess and visualize

put nnUNet results in the directory, run postprocess.py to post process and visualize the results

### Step 6: final classification

TODO