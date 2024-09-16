import shutil
import os
import cv2
import json


def convert_labelme_to_coco(ann_file, out_file, image_prefix):
    files = [item.path for item in os.scandir(image_prefix) if item.is_file()]
    img_files = [f for f in files if f.endswith("jpg")]
    label_files = [f for f in files if f.endswith("json")]
    img_files.sort()
    label_files.sort()

    annotations = []
    images = []
    obj_count = 0
    
    for i in range(len(img_files)) :
        filename = os.path.split(img_files[i])[-1].split(".")[0]
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
            
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            y_min = min(y1, y2)
            y_max = max(y1, y2)
            
            poly = [x1, y1, x2, y1, x2, y2, x1, y2]
        
            data_anno = dict(
                image_id=i,
                id=obj_count,
                category_id=0,
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
            'name': 'target'
        }])
    
    with open(out_file, "w", encoding='utf-8') as f:
        json.dump(coco_format_json, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    
    # files = [item.path for item in os.scandir("dataset/Task1") if item.is_file()]
    # for file in files :
    #     sig = os.path.split(file)[-1].split("_")[0]
    #     typ = os.path.split(file)[-1].split(".")[-1]
    #     shutil.move(file, os.path.join("dataset", "Task1", f"{sig}_YS_T1.{typ}"))
    
    convert_labelme_to_coco(ann_file='dataset/Task1',
                            out_file='dataset/Task1_coco/annotation_coco.json',
                            image_prefix='dataset/Task1')