python mmdetection/tools/train.py configs/t1_yolox_s.py
python mmdetection/tools/train.py configs/t1_yolox_m.py
python mmdetection/tools/train.py configs/t1_efficientdet_effb0.py
python mmdetection/tools/train.py configs/t1_efficientdet_effb3.py
python mmdetection/tools/train.py configs/t1_centernet_r50.py
python mmdetection/tools/train.py configs/t1_centernet_r101.py

python mmdetection/tools/train.py configs/t3_yolox_s.py
python mmdetection/tools/train.py configs/t3_yolox_m.py
python mmdetection/tools/train.py configs/t3_efficientdet_effb0.py
python mmdetection/tools/train.py configs/t3_efficientdet_effb3.py
python mmdetection/tools/train.py configs/t3_centernet_r50.py
python mmdetection/tools/train.py configs/t3_centernet_r101.py


python mmdetection/tools/test.py configs/t1_efficientdet_effb0.py work_dirs/t1_efficientdet_effb0/epoch_75.pth --show-dir t1_efficientdet_effb0 --out t1_effb0.pkl
python mmdetection/tools/test.py configs/t3_efficientdet_effb3.py work_dirs/t3_efficientdet_effb3/epoch_75.pth --show-dir t3_efficientdet_effb3 --out t3_effb3.pkl

