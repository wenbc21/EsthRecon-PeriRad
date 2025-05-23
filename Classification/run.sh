
for x in $(seq 1 5)
do
    python train.py --fold $x --task Task1_balanced_5fold --model_config ResNet34 --data_path dataset/Task1_crop_balanced_5fold
    python train.py --fold $x --task Task1_balanced_5fold --model_config ResNet50 --data_path dataset/Task1_crop_balanced_5fold
    python train.py --fold $x --task Task1_balanced_5fold --model_config ResNet101 --data_path dataset/Task1_crop_balanced_5fold
    python train.py --fold $x --task Task1_balanced_5fold --model_config ResNeXt50_32x4d --data_path dataset/Task1_crop_balanced_5fold
    python train.py --fold $x --task Task1_balanced_5fold --model_config ResNeXt101_32x8d --data_path dataset/Task1_crop_balanced_5fold
    python train.py --fold $x --task Task1_balanced_5fold --model_config DenseNet121 --data_path dataset/Task1_crop_balanced_5fold
    python train.py --fold $x --task Task1_balanced_5fold --model_config DenseNet161 --data_path dataset/Task1_crop_balanced_5fold
    python train.py --fold $x --task Task1_balanced_5fold --model_config DenseNet169 --data_path dataset/Task1_crop_balanced_5fold
    python train.py --fold $x --task Task1_balanced_5fold --model_config DenseNet201 --data_path dataset/Task1_crop_balanced_5fold
    python train.py --fold $x --task Task1_balanced_5fold --model_config EfficientNetV2_s --data_path dataset/Task1_crop_balanced_5fold
    python train.py --fold $x --task Task1_balanced_5fold --model_config EfficientNetV2_m --data_path dataset/Task1_crop_balanced_5fold
    python train.py --fold $x --task Task1_balanced_5fold --model_config EfficientNetV2_l --data_path dataset/Task1_crop_balanced_5fold
    python train.py --fold $x --task Task1_balanced_5fold --model_config ConvNeXt_tiny --data_path dataset/Task1_crop_balanced_5fold
    python train.py --fold $x --task Task1_balanced_5fold --model_config ConvNeXt_small --data_path dataset/Task1_crop_balanced_5fold
    python train.py --fold $x --task Task1_balanced_5fold --model_config ConvNeXt_base --data_path dataset/Task1_crop_balanced_5fold
    python train.py --fold $x --task Task1_balanced_5fold --model_config ConvNeXt_large --data_path dataset/Task1_crop_balanced_5fold
    python train.py --fold $x --task Task1_balanced_5fold --model_config ConvNeXt_xlarge --data_path dataset/Task1_crop_balanced_5fold
done

# final
python train.py --fold 0 --task Task1_balanced --model_config DenseNet169 --data_path dataset/Task1_crop_balanced_5fold --epochs 80 --lr 2e-4
python train.py --fold 0 --task Task3_crop --model_config DenseNet161 --data_path dataset/Task3_crop_5fold --epochs 30 --lr 1e-4

python inference.py --task Task1_balanced --model_config DenseNet169 --data_path dataset/Task1_crop_balanced_5fold
python inference.py --task Task3_crop --model_config DenseNet161 --data_path dataset/Task3_crop_5fold