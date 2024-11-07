import json
import os
import os.path as osp
import PIL.Image
import shutil
import pandas as pd
import random
from labelme import utils
import cv2

def convert_labelme_to_nnunet(raw_file, out_file):
    os.makedirs(f"{out_file}/imagesTr", exist_ok=True)
    os.makedirs(f"{out_file}/imagesTs", exist_ok=True)
    os.makedirs(f"{out_file}/labelsTr", exist_ok=True)
    os.makedirs(f"{out_file}/labelsTs", exist_ok=True)
    
    df = pd.read_csv("data/groundtruth.csv", encoding="utf-8")
    pat_id = list({pid.split("R")[0][1:]: 1 for pid in df["id"].to_list()})
    print(len(pat_id))

    random.seed(75734)
    # train = 0.64 val = 0.16 test = 0.2
    random.shuffle(pat_id)
    train = pat_id[:round(0.64*len(pat_id))]
    val = pat_id[round(0.64*len(pat_id)):round(0.8*len(pat_id))]
    test = pat_id[round(0.8*len(pat_id)):]
    test.sort()
    print(test)

    files = [item.path for item in os.scandir(raw_file) if item.is_file()]
    img_files = [f for f in files if f.endswith("jpg")]
    label_files = [f for f in files if f.endswith("json")]
    img_files.sort()
    label_files.sort()

    # make final split
    train_split = []
    val_split = []
    for i in range(len(img_files)):
        filename = os.path.split(img_files[i])[-1].split(".")[0]
        if filename.split("R")[0][1:] in train :
            train_split.append(filename)
        if filename.split("R")[0][1:] in val :
            val_split.append(filename)
    split_json = 5 * [{"train":train_split, "val":val_split}]
    with open(f"{out_file}/splits_final.json","w") as f:
        json.dump(split_json, f, indent=4)

    for i in range(len(img_files)):
        filename = os.path.split(label_files[i])[-1].split(".")[0]    # 提取出.json前的字符作为文件名，以便后续保存Label图片的时候使用
        split = "Ts" if filename.split("R")[0][1:] in test else "Tr"
        data = json.load(open(label_files[i]))
        # lbl为label图片（标注的地方用类别名对应的数字来标，其他为0）lbl_names为label名和数字的对应关系字典
        lbl, lbl_names = utils.shape.labelme_shapes_to_label(cv2.imread(img_files[i]).shape[:2], data['shapes'])   # data['shapes']是json文件中记录着标注的位置及label等信息的字段

        #captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
        #lbl_viz = utils.draw.draw_label(lbl, img, captions)
        # out_dir = osp.basename(list[i])[:-5]+'_json'
        # out_dir = osp.join(osp.dirname(list[i]), out_dir)
        # if not osp.exists(out_dir):
        #     os.mkdir(out_dir)
        shutil.copy(img_files[i], os.path.join(out_file, f"images{split}", f"{filename}_0000.png"))
        PIL.Image.fromarray(lbl).save(osp.join(out_file, f"labels{split}", '{}.png'.format(filename)))
        #PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, '{}_viz.jpg'.format(filename)))

        # with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
        #     for lbl_name in lbl_names:
        #         f.write(lbl_name + '\n')

        # warnings.warn('info.yaml is being replaced by label_names.txt')
        # info = dict(label_names=lbl_names)
        # with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
        #     yaml.safe_dump(info, f, default_flow_style=False)

        # print('Saved to: %s' % out_dir)


if __name__ == '__main__':
    
    convert_labelme_to_nnunet(raw_file='data/raw_data/Task2_labelme',
                            out_file='data/dataset/Task2')
