import os
import numpy as np
import shutil
from PIL import Image

os.makedirs("nnUNet_raw/mask", exist_ok=True)
os.makedirs("nnUNet_raw/source", exist_ok=True)
image_dirs = [item.path for item in os.scandir("dataset") if item.is_dir()]

for imd in image_dirs :
    sig = os.path.split(imd)[-1].split("_")[0]
    mask = os.path.join(imd, f"{sig}_mask.png")
    source = os.path.join(imd, f"{sig}_source.png")
    shutil.copyfile(source, os.path.join("nnUNet_raw", "source", f"{sig}_YS_T2_0000.png"))
    
    img = Image.open(mask)
    img = np.array(img)
    img[img > 0] = 1
    img = Image.fromarray(img)
    img.save(os.path.join("nnUNet_raw", "mask", f"{sig}_YS_T2.png"))