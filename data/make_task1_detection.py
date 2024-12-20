import shutil
import os
import cv2
import json
import random
import pandas as pd
import numpy as np

def convert_labelme_to_coco(raw_file, external_file, out_file):
    os.makedirs(out_file, exist_ok=True)
    files = [item.path for item in os.scandir(raw_file) if item.is_file()]
    img_files = [f for f in files if f.endswith("jpg")]
    label_files = [f for f in files if f.endswith("json")]
    img_files.sort()
    label_files.sort()

    externals = [item.path for item in os.scandir(external_file) if item.is_file()]
    img_external = [f for f in externals if f.endswith("jpg")]
    label_external = [f for f in externals if f.endswith("json")]
    img_external.sort()
    label_external.sort()
    external_id = list(range(len(img_external)))
    print(external_id)
    
    df = pd.read_csv("data/groundtruth.csv", encoding="utf-8")
    pat_id = list({pid.split("R")[0][1:]: 1 for pid in df["id"].to_list()})
    print(len(pat_id))

    random.seed(75734) # 1234 for 4, 12345 for 5
    # train = 0.64 val = 0.16 test = 0.2
    random.shuffle(pat_id)
    train = pat_id[:round(0.64*len(pat_id))]
    val = pat_id[round(0.64*len(pat_id)):round(0.8*len(pat_id))]
    test = pat_id[round(0.8*len(pat_id)):]
    test.sort()
    print(test)
    random.shuffle(external_id)
    train_external = external_id[:round(0.8*len(external_id))]
    val_external = external_id[round(0.8*len(external_id)):]
    
    split_compose = {"train":train, "val":val, "test":test}
    split_external = {"train":train_external, "val":val_external, "test":[]}
    for split in ["train", "val", "test"] :
        os.makedirs(os.path.join(out_file, split), exist_ok=True)

        annotations = []
        images = []
        obj_count = 0
        
        for i in range(len(img_files)) :
            filename = os.path.split(img_files[i])[-1]
            if filename.split("R")[0][1:] not in split_compose[split] :
                continue
            shutil.copy(img_files[i], os.path.join(out_file, split, os.path.split(img_files[i])[-1]))
            height, width = cv2.imread(img_files[i]).shape[:2]

            images.append(
                dict(id=i, file_name=filename, height=height, width=width))

            with open(label_files[i], "r") as f:
                labels = json.load(f)
                
            for label in labels["shapes"] :
            
                x1 = label["points"][0][0]
                x2 = label["points"][1][0]
                y1 = label["points"][0][1]
                y2 = label["points"][1][1]
                
                x_min = int(min(x1, x2))
                x_max = int(max(x1, x2))
                y_min = int(min(y1, y2))
                y_max = int(max(y1, y2))
                
                poly = [x1, y1, x2, y1, x2, y2, x1, y2]
                
                # cat_id = 1 if label["label"] == "Y" else 0
                cat_id = 0
            
                data_anno = dict(
                    image_id=i,
                    id=obj_count,
                    category_id=cat_id,
                    bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                    area=(x_max - x_min) * (y_max - y_min),
                    segmentation=[poly],
                    iscrowd=0)
                annotations.append(data_anno)
                obj_count += 1
        
        for i in range(len(img_external)) :
            filename = os.path.split(img_external[i])[-1]
            if i not in split_external[split] :
                continue
            shutil.copy(img_external[i], os.path.join(out_file, split, os.path.split(img_external[i])[-1]))
            height, width = cv2.imread(img_external[i]).shape[:2]

            images.append(
                dict(id=i+len(img_files), file_name=filename, height=height, width=width))

            with open(label_external[i], "r") as f:
                labels = json.load(f)
                
            for label in labels["shapes"] :
            
                x1 = label["points"][0][0]
                x2 = label["points"][1][0]
                y1 = label["points"][0][1]
                y2 = label["points"][1][1]
                
                x_min = int(min(x1, x2))
                x_max = int(max(x1, x2))
                y_min = int(min(y1, y2))
                y_max = int(max(y1, y2))
                
                poly = [x1, y1, x2, y1, x2, y2, x1, y2]
                
                # cat_id = 1 if label["label"] == "Y" else 0
                cat_id = 0
            
                data_anno = dict(
                    image_id=i+len(img_files),
                    id=obj_count,
                    category_id=cat_id,
                    bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                    area=(x_max - x_min) * (y_max - y_min),
                    segmentation=[poly],
                    iscrowd=0)
                annotations.append(data_anno)
                obj_count += 1

        coco_format_json = dict(
            images=images,
            annotations=annotations,
            categories=[{
                'id': 0,
                'name': 'ROI'
            }])
        
        with open(os.path.join(out_file, f"annotation_coco_{split}.json"), "w", encoding='utf-8') as f:
            json.dump(coco_format_json, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    
    convert_labelme_to_coco(raw_file='data/raw_data/Task1_labelme',
                            external_file='data/raw_data/origin_T1/origin',
                            out_file='data/dataset/Task1_detection')