import os
import numpy as np
import pandas as pd

image_dirs = [item.path for item in os.scandir("raw_data/YS") if item.is_file()]
groundtruth = pd.read_csv("groundtruth.csv").values.tolist()

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

import random
random.seed(42)
# train = 0.64 val = 0.16 test = 0.2
total = [i+1 for i in range(pat_num)]
random.shuffle(total)
# fold1 = total[:round(0.16*pat_num)]
# fold2 = total[round(0.16*pat_num):round(0.32*pat_num)]
# fold3 = total[round(0.32*pat_num):round(0.48*pat_num)]
# fold4 = total[round(0.48*pat_num):round(0.64*pat_num)]
# fold5 = total[round(0.64*pat_num):round(0.8*pat_num)]
train = total[:round(0.64*pat_num)]
val = total[round(0.64*pat_num):round(0.8*pat_num)]
test = total[round(0.8*pat_num):]

print(total)
print(train)
print(val)
# print(fold1)
# print(fold2)
# print(fold3)
# print(fold4)
# print(fold5)
test.sort()
print(test)
exit()

for it in ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'test'] :
    os.makedirs(f"dataset/Task1/{it}/Y", exist_ok=True)
    os.makedirs(f"dataset/Task1/{it}/N", exist_ok=True)
    os.makedirs(f"dataset/Task2/{it}/H", exist_ok=True)
    os.makedirs(f"dataset/Task2/{it}/L", exist_ok=True)
    os.makedirs(f"dataset/Task3/{it}/Y", exist_ok=True)
    os.makedirs(f"dataset/Task3/{it}/N", exist_ok=True)

from PIL import Image
for i in range(img_num) :
    img = Image.open(image_dirs[i])
    sig = os.path.split(image_dirs[i])[-1].split("_")[0]
    idx = 0
    patid2 = int(sig[1:].split('R')[0])
    for g in range(len(groundtruth)) :
        if sig == groundtruth[g][0] :
            idx = g
            break
    print(groundtruth[idx], image_dirs[i])
    t1 = groundtruth[idx][1]
    t2 = groundtruth[idx][2]
    t3 = groundtruth[idx][3]
    
    if patid2 in fold1 :
        if t1 == 1 :
            img.save(f"dataset/Task1/fold1/Y/{sig}_YS_T1.jpg")
        elif t1 == 0 :
            img.save(f"dataset/Task1/fold1/N/{sig}_YS_T1.jpg")
        if t2 == 'H' :
            img.save(f"dataset/Task2/fold1/H/{sig}_YS_T2.jpg")
        elif t2 == 'L' :
            img.save(f"dataset/Task2/fold1/L/{sig}_YS_T2.jpg")
        if t3 == 1 :
            img.save(f"dataset/Task3/fold1/Y/{sig}_YS_T3.jpg")
        elif t3 == 0 :
            img.save(f"dataset/Task3/fold1/N/{sig}_YS_T3.jpg")
            
    if patid2 in fold2 :
        if t1 == 1 :
            img.save(f"dataset/Task1/fold2/Y/{sig}_YS_T1.jpg")
        elif t1 == 0 :
            img.save(f"dataset/Task1/fold2/N/{sig}_YS_T1.jpg")
        if t2 == 'H' :
            img.save(f"dataset/Task2/fold2/H/{sig}_YS_T2.jpg")
        elif t2 == 'L' :
            img.save(f"dataset/Task2/fold2/L/{sig}_YS_T2.jpg")
        if t3 == 1 :
            img.save(f"dataset/Task3/fold2/Y/{sig}_YS_T3.jpg")
        elif t3 == 0 :
            img.save(f"dataset/Task3/fold2/N/{sig}_YS_T3.jpg")
            
    if patid2 in fold3 :
        if t1 == 1 :
            img.save(f"dataset/Task1/fold3/Y/{sig}_YS_T1.jpg")
        elif t1 == 0 :
            img.save(f"dataset/Task1/fold3/N/{sig}_YS_T1.jpg")
        if t2 == 'H' :
            img.save(f"dataset/Task2/fold3/H/{sig}_YS_T2.jpg")
        elif t2 == 'L' :
            img.save(f"dataset/Task2/fold3/L/{sig}_YS_T2.jpg")
        if t3 == 1 :
            img.save(f"dataset/Task3/fold3/Y/{sig}_YS_T3.jpg")
        elif t3 == 0 :
            img.save(f"dataset/Task3/fold3/N/{sig}_YS_T3.jpg")
            
    if patid2 in fold4 :
        if t1 == 1 :
            img.save(f"dataset/Task1/fold4/Y/{sig}_YS_T1.jpg")
        elif t1 == 0 :
            img.save(f"dataset/Task1/fold4/N/{sig}_YS_T1.jpg")
        if t2 == 'H' :
            img.save(f"dataset/Task2/fold4/H/{sig}_YS_T2.jpg")
        elif t2 == 'L' :
            img.save(f"dataset/Task2/fold4/L/{sig}_YS_T2.jpg")
        if t3 == 1 :
            img.save(f"dataset/Task3/fold4/Y/{sig}_YS_T3.jpg")
        elif t3 == 0 :
            img.save(f"dataset/Task3/fold4/N/{sig}_YS_T3.jpg")
            
    if patid2 in fold5 :
        if t1 == 1 :
            img.save(f"dataset/Task1/fold5/Y/{sig}_YS_T1.jpg")
        elif t1 == 0 :
            img.save(f"dataset/Task1/fold5/N/{sig}_YS_T1.jpg")
        if t2 == 'H' :
            img.save(f"dataset/Task2/fold5/H/{sig}_YS_T2.jpg")
        elif t2 == 'L' :
            img.save(f"dataset/Task2/fold5/L/{sig}_YS_T2.jpg")
        if t3 == 1 :
            img.save(f"dataset/Task3/fold5/Y/{sig}_YS_T3.jpg")
        elif t3 == 0 :
            img.save(f"dataset/Task3/fold5/N/{sig}_YS_T3.jpg")
            
    if patid2 in test :
        if t1 == 1 :
            img.save(f"dataset/Task1/test/Y/{sig}_YS_T1.jpg")
        elif t1 == 0 :
            img.save(f"dataset/Task1/test/N/{sig}_YS_T1.jpg")
        if t2 == 'H' :
            img.save(f"dataset/Task2/test/H/{sig}_YS_T2.jpg")
        elif t2 == 'L' :
            img.save(f"dataset/Task2/test/L/{sig}_YS_T2.jpg")
        if t3 == 1 :
            img.save(f"dataset/Task3/test/Y/{sig}_YS_T3.jpg")
        elif t3 == 0 :
            img.save(f"dataset/Task3/test/N/{sig}_YS_T3.jpg")