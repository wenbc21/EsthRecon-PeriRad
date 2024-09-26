import shutil
import os
import cv2
import json


def convert_labelme_to_coco(ann_file, out_file):
    os.makedirs(out_file, exist_ok=True)
    files = [item.path for item in os.scandir(ann_file) if item.is_file()]
    img_files = [f for f in files if f.endswith("jpg")]
    label_files = [f for f in files if f.endswith("json")]
    img_files.sort()
    label_files.sort()
    
    pat = {}
    for i in img_files :
        imgid = os.path.split(i)[-1].split('R')[0]
        pat[imgid] = 1
    pat_num = len(pat)
    img_num = len(img_files)
    print(pat_num, img_num)

    import random
    random.seed(42)
    # train = 0.64 val = 0.16 test = 0.2
    total = [i+1 for i in range(pat_num)]
    random.shuffle(total)
    train = total[:round(0.64*pat_num)]
    val = total[round(0.64*pat_num):round(0.8*pat_num)]
    test = total[round(0.8*pat_num):]
    
    split_compose = {"train":train, "val":val, "test":test}
    for split in ["train", "val", "test"] :
        os.makedirs(os.path.join(out_file, split), exist_ok=True)

        annotations = []
        images = []
        obj_count = 0
        
        for i in range(len(img_files)) :
            filename = os.path.split(img_files[i])[-1]
            if int(filename.split("R")[0][1:]) not in split_compose[split] :
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
                
                x_min = min(x1, x2)
                x_max = max(x1, x2)
                y_min = min(y1, y2)
                y_max = max(y1, y2)
                
                poly = [x1, y1, x2, y1, x2, y2, x1, y2]
                
                cat_id = 1 if label["label"] == "Y" else 0
            
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

        coco_format_json = dict(
            images=images,
            annotations=annotations,
            categories=[{
                'id': 0,
                'name': 'N'
            }, {
                'id': 1,
                'name': 'Y'
            }])
        
        with open(os.path.join(out_file, f"annotation_coco_{split}.json"), "w", encoding='utf-8') as f:
            json.dump(coco_format_json, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    
    convert_labelme_to_coco(ann_file='dataset/Task1',
                            out_file='dataset/Task1_coco')