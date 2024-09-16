import os
import shutil

for cls in ["H", "L"] :
    os.makedirs(f"nnUNet_test/{cls}", exist_ok=True)
    image_dirs = [item.path for item in os.scandir(f"test/{cls}") if item.is_file()]
    for imd in image_dirs :
        sig = os.path.split(imd)[-1].split(".")[0]
        shutil.copyfile(imd, f"nnUNet_test/{cls}/{sig}_0000.png")
    