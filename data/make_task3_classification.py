import shutil
import os
import cv2
import json
import random
import pandas as pd
import numpy as np
from PIL import Image

def crop_image_from_labelme(raw_file, out_file):
    os.makedirs(out_file, exist_ok=True)
    files = [item.path for item in os.scandir(raw_file) if item.is_file()]
    img_files = [f for f in files if f.endswith("jpg")]
    label_files = [f for f in files if f.endswith("json")]
    img_files.sort()
    label_files.sort()
    
    ids = [str(i) for i in range(1, 276)]
    random.seed(75734)
    # train = 0.64 val = 0.16 test = 0.2
    random.shuffle(ids)
    train = ids[:round(0.64*len(ids))]
    val = ids[round(0.64*len(ids)):round(0.8*len(ids))]
    test = ids[round(0.8*len(ids)):]
    # fold1 = ids[:round(0.16*len(ids))]
    # fold2 = ids[round(0.16*len(ids)):round(0.32*len(ids))]
    # fold3 = ids[round(0.32*len(ids)):round(0.48*len(ids))]
    # fold4 = ids[round(0.48*len(ids)):round(0.64*len(ids))]
    # fold5 = ids[round(0.64*len(ids)):round(0.8*len(ids))]
    # test = ids[round(0.8*len(ids)):]
    test.sort()
    print(test)
    
    split_compose = {"train":train, "val":val, "test":test}
    for split in ["train", "val", "test"] :
    # split_compose = {"fold1":fold1, "fold2":fold2, "fold3":fold3, "fold4":fold4, "fold5":fold5, "test":test}
    # for split in ["fold1", "fold2", "fold3", "fold4", "fold5", "test"] :
        os.makedirs(os.path.join(out_file, split, "Y"), exist_ok=True)
        os.makedirs(os.path.join(out_file, split, "N"), exist_ok=True)

        for i in range(len(img_files)) :
            filename = os.path.split(img_files[i])[-1]
            if filename.split("R")[0][1:] not in split_compose[split] :
                continue
            
            image = Image.open(img_files[i])
            
            with open(label_files[i], "r") as f:
                labels = json.load(f)
                
            for j in range(len(labels["shapes"])) :
                label = labels["shapes"][j]
                
                x1 = label["points"][0][0]
                x2 = label["points"][1][0]
                y1 = label["points"][0][1]
                y2 = label["points"][1][1]
                
                x_min = int(min(x1, x2))
                x_max = int(max(x1, x2))
                y_min = int(min(y1, y2))
                y_max = int(max(y1, y2))
                
                x_delta = int((x_max - x_min) * 0.1)
                y_delta = int((y_max - y_min) * 0.1)
                
                x_displacement = list(range(-x_delta, x_delta))
                y_displacement = list(range(-y_delta, y_delta))
                random.shuffle(x_displacement)
                random.shuffle(y_displacement)
                
                category = label["label"]
                
                img_crop = image.crop((x_min, y_min, x_max, y_max))
                
                img_crop.save(os.path.join(out_file, split, category, f"{filename.split('.')[0]}_{j}.jpg"))
                
                if split == "test" :
                    continue
                
                if category == "Y" :
                    aug_size = 5
                else :
                    aug_size = 2

                for k in range(aug_size) :
                    xx_min = x_min + x_displacement[k]
                    yy_min = y_min + y_displacement[k]
                    xx_max = x_max + x_displacement[-k]
                    yy_max = y_max + y_displacement[-k]
                    if xx_min < 0 :
                        xx_min = 0
                    if yy_min < 0 :
                        yy_min = 0
                    if xx_min >= image.height :
                        xx_min = image.height-1
                    if yy_min >= image.width :
                        yy_min = image.width-1
                
                    img_crop = image.crop((xx_min, yy_min, xx_max, yy_max))
                    
                    img_crop.save(os.path.join(out_file, split, category, f"{filename.split('.')[0]}_{j}_aug{k+1}.jpg"))


if __name__ == '__main__':
    
    crop_image_from_labelme(raw_file='data/raw_data/Task3_labelme',
                            out_file='data/dataset/Task3_crop_balanced')