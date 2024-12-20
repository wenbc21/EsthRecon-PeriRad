for x in $(seq 1 5)
do
    python train.py --fold $x --task Task3_crop_5fold --model_config ConvNeXt_large --data_path dataset/Task3_crop_5fold
done

for x in $(seq 1 5)
do
    python train.py --fold $x --task Task3_crop_5fold --model_config ConvNeXt_xlarge --data_path dataset/Task3_crop_5fold
done

# final
python train.py --fold 0 --task Task1_crop_balanced --model_config DenseNet169 --data_path dataset/Task1_crop_balanced --epochs 50 --lr 2e-4
python train.py --fold 0 --task Task3_crop_balanced --model_config DenseNet169 --data_path dataset/Task3_crop_balanced --epochs 30 --lr 1e-4
