import os
from PIL import Image
import numpy as np
import cv2

distance = []

img_file = "imagesTs"
pdt_file = "predictTs"
    
os.makedirs(f"results/post", exist_ok=True)
os.makedirs(f"results/vis_seg", exist_ok=True)
os.makedirs(f"results/vis_ori/", exist_ok=True)
os.makedirs(f"results/bbox", exist_ok=True)
images = [item.path for item in os.scandir(img_file) if item.is_file()]
predicts = [item.path for item in os.scandir(pdt_file) if item.is_file()]
images.sort()
predicts.sort()

for i in range(len(images)) :
    sig = os.path.split(images[i])[-1].split("_")[0]
    img = cv2.imread(images[i])
    res = cv2.imread(predicts[i])
    # print(images[i], predicts[i])
    
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))
    for k in range(len(contours)):
        if k != max_idx :
            res = cv2.drawContours(res, contours, k, 0, cv2.FILLED)
    post = res
    post[post > 0] = 255
    cv2.imwrite(f"results/post/{sig}_post.png", res)
    
    bbox = np.zeros(post.shape, dtype=np.uint8)
    cnt_len = cv2.arcLength(contours[max_idx], True)
    cnt = cv2.approxPolyDP(contours[max_idx], 0.05*cnt_len, True)
    if len(cnt) == 4:
        cv2.drawContours(bbox, [cnt], -1, 255, 3)
        cv2.imwrite(f"results/bbox/{sig}_bbox.png", bbox)
    else :
        print(f"{sig}!!!")
    
    res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    img = np.array(img).astype(np.uint8)
    res = np.array(res).astype(np.uint8)
    vis = img + res * (75, 0, 0)
    vis = np.uint8(vis)
    for pnt in cnt :
        cv2.circle(vis, pnt[0], 3, (255, 255, 255), 3)
    vis = Image.fromarray(np.uint8(vis))
    vis.save(f"results/vis_seg/{sig}_vis_seg.png")
    
    vis_ori = img
    for pnt in cnt :
        cv2.circle(vis_ori, pnt[0], 3, (255, 255, 255), 3)
    vis_ori = Image.fromarray(np.uint8(vis_ori))
    vis_ori.save(f"results/vis_ori/{sig}_vis_ori.png")
    
    cnt = cnt.reshape(4, 2)
    sorted_indices = np.argsort(cnt[:, 1]+cnt[:, 0])  # 获取按第二列排序的索引
    sorted_cnt = cnt[sorted_indices]
    upper_left = sorted_cnt[0]
    lower_right = sorted_cnt[3]
    
    sorted_indices = np.argsort(cnt[:, 1]-cnt[:, 0])  # 获取按第二列排序的索引
    sorted_cnt = cnt[sorted_indices]
    upper_right = sorted_cnt[0]
    lower_left = sorted_cnt[3]

    if upper_left[1] > lower_left[1] or upper_left[1] > lower_right[1]:
        print("!!!", sig)
    if upper_right[1] > lower_left[1] or upper_right[1] > lower_right[1]:
        print("!!!", sig)
    if upper_right[0] < lower_left[0] or upper_right[0] < upper_left[0]:
        print("!!!", sig)
    if lower_right[0] < lower_left[0] or lower_right[0] < upper_left[0]:
        print("!!!", sig)
        
    print(sig, upper_left, upper_right, lower_left, lower_right)
    left_distance = np.linalg.norm(lower_left - upper_left)
    right_distance = np.linalg.norm(lower_right - upper_right)
    # print(sig, img.shape, left_distance, right_distance)
    distance.append([sig, left_distance, right_distance])

import pandas as pd
data2 = pd.DataFrame(data = distance,index = None,columns = ["img_name", "left_distance", "right_distance"])
data2.to_csv("distance.csv")