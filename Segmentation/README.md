# EsthRecon-PeriRad - Segmentation Module

Segmentation workspace for DISTAN task.
nnUNet is used for its auto configuration ability.

## Usage
1. Use [labelme](https://github.com/wkentaro/labelme) to label semantic masks, use [make_task2_dataset.py](make_task2_dataset.py) to build the dataset. 
2. Install and setup [nnUNet](https://github.com/MIC-DKFZ/nnUNet).
3. Train a segmentation model.
4. Inference on testset using trained model.
5. Use [main.py](main.py) to postprocess and visualize the results.
