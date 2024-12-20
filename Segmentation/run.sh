export nnUNet_raw="/home/amax/Project/nnUNet/dataset/nnUNet_raw"
export nnUNet_preprocessed="/home/amax/Project/nnUNet/dataset/nnUNet_preprocessed"
export nnUNet_results="/home/amax/Project/nnUNet/dataset/nnUNet_results"

nnUNetv2_plan_and_preprocess -d 835 --verify_dataset_integrity
nnUNetv2_train 835 2d 0   # remenber to change the epochs number to 100
nnUNetv2_predict -i dataset/nnUNet_raw/Dataset835_SACT2/imagesTs -o dataset/nnUNet_raw/Dataset835_SACT2/predictTs -d 835 -c 2d -f 0