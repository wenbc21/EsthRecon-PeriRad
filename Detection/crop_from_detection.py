import pickle
import numpy as np
import os
import pandas as pd
import cv2

with open('t1_effb0.pkl', 'rb') as f:
	t1_inference_results = pickle.load(f)
with open('t3_effb3.pkl', 'rb') as f:
	t3_inference_results = pickle.load(f)

# t2 metadata
t2_metadata = pd.read_csv('/home/amax/Project/wbc/SAC/Segmentation/results/distance.csv')
image_roi = {}
image_name = t2_metadata["img_name"].values.tolist()
center_x = t2_metadata["center_x"].values.tolist()
for i in range(len(image_name)) :
	image_roi[image_name[i]] = float(center_x[i])

# make result file
os.makedirs("crop_from_detection/t1_bboxes", exist_ok=True)
os.makedirs("crop_from_detection/t3_bboxes", exist_ok=True)
os.makedirs("crop_from_detection/t1_cropped", exist_ok=True)
os.makedirs("crop_from_detection/t3_cropped", exist_ok=True)

# t1
for t1_result in t1_inference_results :
	img_path = t1_result["img_path"]
	img_name = os.path.split(img_path)[-1].split('.')[0]
	bbox_pred = np.array(t1_result["pred_instances"]["scores"])
	bbox_instance = np.array(t1_result["pred_instances"]["bboxes"])

	bbox_pred = np.argwhere(bbox_pred > 0.3)
	bbox_instance = bbox_instance[bbox_pred].reshape(-1, 4)

	ori_bbox_num = bbox_instance.shape[0]
	if ori_bbox_num > 2 :
		if img_name in image_roi :
			roi_center_x = image_roi[img_name]
			left_bbox = []
			left_bbox_dis = []
			right_bbox = []
			right_bbox_dis = []
			for bbox in bbox_instance :
				bbox_x = (bbox[0] + bbox[2]) / 2
				if bbox_x < roi_center_x :
					left_bbox.append(bbox)
					left_bbox_dis.append(abs(bbox_x - roi_center_x))
				else :
					right_bbox.append(bbox)
					right_bbox_dis.append(abs(bbox_x - roi_center_x))
			bbox_instance = np.array([left_bbox[np.argmin(left_bbox_dis)], right_bbox[np.argmin(right_bbox_dis)]])
			print("t1", img_name, ori_bbox_num, "->", bbox_instance.shape[0])
	elif ori_bbox_num == 2 :
		if bbox_instance[0][0] + bbox_instance[0][3] > bbox_instance[1][0] + bbox_instance[1][3] :
			bbox_instance = np.array([bbox_instance[1], bbox_instance[0]])
		print("t1", img_name)
	else :
		bbox_instance = np.array(bbox_instance)
	
	img = cv2.imread(img_path)
	for i in range(bbox_instance.shape[0]) :
		bbox = bbox_instance[i]
		cv2.imwrite(f"crop_from_detection/t1_cropped/{img_name}_{i+1}.png", img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :])
	for i in range(bbox_instance.shape[0]) :
		bbox = bbox_instance[i]
		cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 3)
	cv2.imwrite(f"crop_from_detection/t1_bboxes/{img_name}.png", img)



# t3
for t3_result in t3_inference_results :
	img_path = t3_result["img_path"]
	img_name = os.path.split(img_path)[-1].split('.')[0]
	bbox_pred = np.array(t3_result["pred_instances"]["scores"])
	bbox_instance = np.array(t3_result["pred_instances"]["bboxes"])

	bbox_pred = np.argwhere(bbox_pred > 0.3)
	bbox_instance = bbox_instance[bbox_pred].reshape(-1, 4)

	ori_bbox_num = bbox_instance.shape[0]
	if ori_bbox_num > 2 :
		if img_name in image_roi :
			roi_center_x = image_roi[img_name]
			left_bbox = []
			left_bbox_dis = []
			right_bbox = []
			right_bbox_dis = []
			for bbox in bbox_instance :
				bbox_x = (bbox[0] + bbox[2]) / 2
				if bbox_x < roi_center_x :
					left_bbox.append(bbox)
					left_bbox_dis.append(abs(bbox_x - roi_center_x))
				else :
					right_bbox.append(bbox)
					right_bbox_dis.append(abs(bbox_x - roi_center_x))
			bbox_instance = np.array([left_bbox[np.argmin(left_bbox_dis)], right_bbox[np.argmin(right_bbox_dis)]])
			print("t3", img_name, ori_bbox_num, "->", bbox_instance.shape[0])
	elif ori_bbox_num == 2 :
		if bbox_instance[0][0] + bbox_instance[0][3] > bbox_instance[1][0] + bbox_instance[1][3] :
			bbox_instance = np.array([bbox_instance[1], bbox_instance[0]])
		print("t3", img_name)
	else :
		bbox_instance = np.array(bbox_instance)
	
	img = cv2.imread(img_path)
	for i in range(bbox_instance.shape[0]) :
		bbox = bbox_instance[i]
		cv2.imwrite(f"crop_from_detection/t3_cropped/{img_name}_{i+1}.png", img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :])
	for i in range(bbox_instance.shape[0]) :
		bbox = bbox_instance[i]
		cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 3)
	cv2.imwrite(f"crop_from_detection/t3_bboxes/{img_name}.png", img)
