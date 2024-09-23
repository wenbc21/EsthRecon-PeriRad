import os
from PIL import Image
import numpy as np
import cv2

distance = []

for cls in ["H", "L"] :
    
    os.makedirs(f"result_img/post/{cls}", exist_ok=True)
    os.makedirs(f"result_img/vis_seg/{cls}", exist_ok=True)
    os.makedirs(f"result_img/vis_ori/{cls}", exist_ok=True)
    os.makedirs(f"result_img/bbox/{cls}", exist_ok=True)
    images = [item.path for item in os.scandir(f"test/{cls}") if item.is_file()]
    results = [item.path for item in os.scandir(f"result_img/{cls}") if item.is_file()]

    for i in range(len(images)) :
        sig = os.path.split(images[i])[-1].split("_")[0]
        img = cv2.imread(images[i])
        res = cv2.imread(results[i])
        
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
        cv2.imwrite(f"result_img/post/{cls}/{sig}_YS_T2_post.png", res)
        
        bbox = np.zeros(post.shape, dtype=np.uint8)
        cnt_len = cv2.arcLength(contours[max_idx], True)
        cnt = cv2.approxPolyDP(contours[max_idx], 0.05*cnt_len, True)
        if len(cnt) == 4:
            cv2.drawContours(bbox, [cnt], -1, 255, 3)
            cv2.imwrite(f"result_img/bbox/{cls}/{sig}_YS_T2_bbox.png", bbox)
        else :
            print(f"{cls}/{sig}_YS_T2")
        
        # res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        # img = np.array(img).astype(np.uint8)
        # res = np.array(res).astype(np.uint8)
        # vis = img + res * (75, 0, 0)
        # for pnt in cnt :
        #     cv2.circle(vis, pnt[0], 3, (255, 255, 255), 3)
        # vis = Image.fromarray(np.uint8(vis))
        # vis.save(f"result_img/vis_seg/{cls}/{sig}_YS_T2_vis_seg.png")
        
        # vis_ori = img
        # for pnt in cnt :
        #     cv2.circle(vis_ori, pnt[0], 3, (255, 255, 255), 3)
        # vis_ori = Image.fromarray(np.uint8(vis_ori))
        # vis_ori.save(f"result_img/vis_ori/{cls}/{sig}_YS_T2_vis_ori.png")
        
        cnt = cnt.reshape(4, 2)
        sorted_indices = np.argsort(cnt[:, 1])  # 获取按第二列排序的索引
        sorted_cnt = cnt[sorted_indices]
        
        if sorted_cnt[0][0] > sorted_cnt[1][0] :
            upper_left = sorted_cnt[1]
            upper_right = sorted_cnt[0]
        else :
            upper_left = sorted_cnt[0]
            upper_right = sorted_cnt[1]
        
        if sorted_cnt[2][0] > sorted_cnt[3][0] :
            lower_left = sorted_cnt[3]
            lower_right = sorted_cnt[2]
        else :
            lower_left = sorted_cnt[2]
            lower_right = sorted_cnt[3]
            
        # print(f"{cls}/{sig}", upper_left, upper_right, lower_left, lower_right)
        left_distance = np.linalg.norm(lower_left - upper_left)
        right_distance = np.linalg.norm(lower_right - upper_right)
        print(f"{cls}/{sig}", img.shape, left_distance, right_distance)
        distance.append([f"{cls}/{sig}", left_distance, right_distance])

import pandas as pd
data2 = pd.DataFrame(data = distance,index = None,columns = ["img_name", "left_distance", "right_distance"])
data2.to_csv("distance.csv")