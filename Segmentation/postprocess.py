import os
from PIL import Image
import numpy as np
import cv2

for cls in ["H", "L"] :
    
    os.makedirs(f"result_img/post/{cls}", exist_ok=True)
    os.makedirs(f"result_img/vis/{cls}", exist_ok=True)
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
        cv2.imwrite(f"result_img/post/{cls}/{sig}_YS_T2_post.png", res)
        
        
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        img = np.array(img)
        res = np.array(res)
        vis = img + res * (75, 0, 0)
        vis = Image.fromarray(np.uint8(vis))
        vis.save(f"result_img/vis/{cls}/{sig}_YS_T2_vis.png")
            