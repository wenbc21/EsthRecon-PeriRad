import os
from PIL import Image
import numpy as np
import cv2
import pandas as pd

metadata = []

img_file = "test_png"
pdt_file = "predict"

os.makedirs(f"results/vis_seg", exist_ok=True)
os.makedirs(f"results/vis_ori", exist_ok=True)
images = [item.path for item in os.scandir(img_file) if item.is_file()]
predicts = [item.path for item in os.scandir(pdt_file) if item.is_file()]
images.sort()
predicts.sort()

for i in range(len(images)) :
    image_name = os.path.split(images[i])[-1].split("_")[0]
    img = cv2.imread(images[i])
    res = cv2.imread(predicts[i])
    
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]
    max_idx = np.argmax(areas)
    for k in range(len(contours)):
        if k != max_idx :
            res = cv2.drawContours(res, contours, k, 0, cv2.FILLED)
    cnt_len = cv2.arcLength(contours[max_idx], True)
    cnt = cv2.approxPolyDP(contours[max_idx], 0.05*cnt_len, True)

    cnt = cnt.reshape(-1, 2)
    sorted_indices = np.argsort(cnt[:, 1]+cnt[:, 0]) 
    sorted_cnt = cnt[sorted_indices]
    upper_left = sorted_cnt[0]
    lower_right = sorted_cnt[-1]
    sorted_indices = np.argsort(cnt[:, 1]-cnt[:, 0]) 
    sorted_cnt = cnt[sorted_indices]
    upper_right = sorted_cnt[0]
    lower_left = sorted_cnt[-1]
    cnt = np.array([upper_left, lower_right, upper_right, lower_left])
    
    res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    vis = img + res * (0, 0, 75)
    vis = np.uint8(vis)
    for pnt in cnt :
        cv2.circle(vis, pnt, 3, (255, 255, 255), 3)
        cv2.circle(img, pnt, 3, (255, 255, 255), 3)
    cv2.imwrite(f"results/vis_seg/{image_name}_vis_seg.png", vis)
    cv2.imwrite(f"results/vis_ori/{image_name}_vis_ori.png", img)

    if upper_left[1] > lower_left[1] or upper_left[1] > lower_right[1]:
        print("check", image_name, "!!!")
    if upper_right[1] > lower_left[1] or upper_right[1] > lower_right[1]:
        print("check", image_name, "!!!")
    if upper_right[0] < lower_left[0] or upper_right[0] < upper_left[0]:
        print("check", image_name, "!!!")
    if lower_right[0] < lower_left[0] or lower_right[0] < upper_left[0]:
        print("check", image_name, "!!!")
        
    print(image_name, upper_left, upper_right, lower_left, lower_right)
    left_distance = np.linalg.norm(lower_left - upper_left)
    right_distance = np.linalg.norm(lower_right - upper_right)
    metadata.append([image_name, np.mean(cnt[:, 0]), np.mean(cnt[:, 1]), left_distance, right_distance])

data2 = pd.DataFrame(data = metadata,index = None,columns = ["img_name", "center_x", "center_y", "left_distance", "right_distance"])
data2.to_csv("results/distance.csv")