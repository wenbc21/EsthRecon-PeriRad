# SAC - Segmentation

For Task2

## Structure

[main.py](main.py) - post process, visualization and get final result

## Workflow

### Step 1: prepare data

Use labelme to label sementic segmentation for Task2, run [make_task2_dataset.py](make_task2_dataset.py) to automatically build the dataset. 

### Step 2: install nnUNet

Install and config [nnUNet](https://github.com/MIC-DKFZ/nnUNet).
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
```

### Step 3: nnUNet training

Train a nnUNet model using data in nnUNet_raw

### Step 4: nnUNet inference

Inference images in testset using trained nnUNet model

### Step 5: final classification

Put nnUNet results in the directory, run [main.py](main.py) to post process and visualize the results
