import os
from PIL import Image
import pandas as pd
import random

raw_file = 'data/raw_data/YS'
out_file = 'data/dataset/Task3'

image_dirs = [item.path for item in os.scandir(raw_file) if item.is_file()]
groundtruth = pd.read_csv("data/groundtruth.csv").values.tolist()

pat = {}
for i in groundtruth :
    imgid = i[0]
    if imgid[1:].split('R')[0] not in pat :
        pat[imgid[1:].split('R')[0]] = 1
    else :
        pat[imgid[1:].split('R')[0]] += 1
pat_num = len(pat)
img_num = len(image_dirs)
print(pat_num, img_num)

df = pd.read_csv("data/groundtruth.csv", encoding="utf-8")
pat_id = list({pid.split("R")[0][1:]: 1 for pid in df["id"].to_list()})

random.seed(75734)
# train = 0.64 val = 0.16 test = 0.2
random.shuffle(pat_id)
train = pat_id[:round(0.64*len(pat_id))]
val = pat_id[round(0.64*len(pat_id)):round(0.8*len(pat_id))]
test = pat_id[round(0.8*len(pat_id)):]

split_compose = {"train":train, "val":val, "test":test}
for split in ["train", "val", "test"] :
    os.makedirs(os.path.join(out_file, split, "Y"), exist_ok=True)
    os.makedirs(os.path.join(out_file, split, "N"), exist_ok=True)

    for i in range(len(image_dirs)) :
        img = Image.open(image_dirs[i])
        sig = os.path.split(image_dirs[i])[-1].split("_")[0]
        idx = 0
        patid2 = sig[1:].split('R')[0]
        for g in range(len(groundtruth)) :
            if sig == groundtruth[g][0] :
                idx = g
                break
        print(groundtruth[idx], image_dirs[i])
        t1 = groundtruth[idx][1]
        t2 = groundtruth[idx][2]
        t3 = groundtruth[idx][3]
        
        if patid2 in split_compose[split] :
            if t3 == 1 :
                img.save(os.path.join(out_file, split, "Y", f"{sig}.png"))
            elif t3 == 0 :
                img.save(os.path.join(out_file, split, "N", f"{sig}.png"))