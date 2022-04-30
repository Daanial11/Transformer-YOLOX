from itertools import count
import json
import matplotlib.pyplot as plt
import torch
import torchvision.ops.boxes as box

import pycocotools.coco as coco_
from pycocotools.cocoeval import COCOeval

result_file_path = 'result_Swin-l_640.json'
ann_path = 'instances_val2017.json'


with open(result_file_path, 'r') as f:
    result_data = json.load(f)

with open(ann_path, 'r') as f:
    ann_data = json.load(f)


results_dict_imgId = {}

for result in result_data:
    
    img_id = result['image_id']
    

    if img_id not in results_dict_imgId:
        results_dict_imgId[img_id] = []
    
    results_dict_imgId[img_id].append(result)


for k, v in results_dict_imgId.items():
    break

annotation_dict_imgId = {}
for ann in ann_data['annotations']:
    img_id = ann['image_id']

    if img_id not in annotation_dict_imgId:
        annotation_dict_imgId[img_id] = []
    
    annotation_dict_imgId[img_id].append(ann)

preds_per_image_highest_ious = {}
for k, v in annotation_dict_imgId.items():
    img_id = k

    highest_ious = []
    for ann in v:
        gt_box = ann['bbox']

        highest_iou = 0

        predictions_for_img = results_dict_imgId[img_id]

        for pred in predictions_for_img:
            bbox = pred['bbox']

            box1 = torch.tensor([[gt_box[0], gt_box[1], gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]]], dtype=torch.float)
            box2 = torch.tensor([[bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]], dtype=torch.float)

            iou = box.box_iou(box1, box2)

            if iou.item() > highest_iou:
                highest_iou = iou.item()
        
        highest_ious.append(highest_iou)
    
    preds_per_image_highest_ious[img_id] = highest_ious


ious = []
for k,v in preds_per_image_highest_ious.items():
    print(f"imageID {k}, preds ious: {v}")

    for iou in v:    
            if iou > 0:
                ious.append(iou)


plt.hist(ious, density=True, bins=30)

plt.ylabel('Probability')
plt.xlabel('Data')

plt.show()







        


        

