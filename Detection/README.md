# EsthRecon-PeriRad - Detection Module

Detection workspace for INFLAM and RESTOR tasks.
mmDetection framework is used, with data in COCO format.

## Usage

1. Use [make_task1_detection.py](../data/make_task1_detection.py) and [make_task3_detection.py](../data/make_task3_detection.py) to convert raw data into coco format.
2. Install [mmDetection](https://github.com/open-mmlab/mmdetection) and edit [configs files](configs) to setup the models.
3. Train a model using the config files.
```
python mmdetection/tools/train.py configs/t1_efficientdet_effb0.py
```
4. Evaluate the results and visualize.
```
python mmdetection/tools/test.py configs/t1_efficientdet_effb0.py work_dirs/t1_efficientdet_effb0/epoch_75.pth --show-dir t1_efficientdet_effb0 --out t1_effb0.pkl
```