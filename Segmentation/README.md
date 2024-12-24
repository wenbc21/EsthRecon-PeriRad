# SAC - Segmentation

Using segmentation to generate missing region and locate the keypoints For Task2
Segmentation workspace using nnU-Net

### Step 1: prepare data

use labelme to label sementic segmentation for Task2, run [make_task2_dataset](../data/make_task2_dataset.py) to automatically build the dataset. 

### Step 2: nnU-Net install

install [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) following the github repo

### Step 3: nnU-Net training

train a nnU-Net model for 100 epochs

### Step 4: nnU-Net inference

inference images in testset using trained nnU-Net model

### Step 5: final classification

put nnU-Net results in the directory, run [main.py](main.py) to post process and visualize the results
