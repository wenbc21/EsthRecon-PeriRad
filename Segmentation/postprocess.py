import os
from PIL import Image
import numpy as np
import cv2

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
        
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        img = np.array(img).astype(np.uint8)
        res = np.array(res).astype(np.uint8)
        vis = img + res * (75, 0, 0)
        for pnt in cnt :
            cv2.circle(vis, pnt[0], 3, (255, 255, 255), 3)
        vis = Image.fromarray(np.uint8(vis))
        vis.save(f"result_img/vis_seg/{cls}/{sig}_YS_T2_vis_seg.png")
        
        vis_ori = img
        for pnt in cnt :
            cv2.circle(vis_ori, pnt[0], 3, (255, 255, 255), 3)
        vis_ori = Image.fromarray(np.uint8(vis_ori))
        vis_ori.save(f"result_img/vis_ori/{cls}/{sig}_YS_T2_vis_ori.png")