import torch
import json
import matplotlib.pyplot as plt
import torch
import torchvision.ops.boxes as box
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd

result_file_path = 'result_Yolo-l_640.json'
ann_path = 'instances_val2017.json'

class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
display_labels = ['BG', 'Bottle', 'Bottle cap', 'Can', 'Cigarette', 'Cup', 'Lid', 'Other', 'PB/W', 'Pop tab', 'Straw']


iou_threshold = 0.35

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

preds_per_image_highest = {}
for k, v in annotation_dict_imgId.items():
    img_id = k

    highest_ious = []
    for ann in v:
        gt_box = ann['bbox']

        highest_iou = 0
        highest_conf = 0

        predictions_for_img = results_dict_imgId[img_id]

        for pred in predictions_for_img:
            bbox = pred['bbox']

            box1 = torch.tensor([[gt_box[0], gt_box[1], gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]]], dtype=torch.float)
            box2 = torch.tensor([[bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]], dtype=torch.float)

            iou = box.box_iou(box1, box2)

            if iou.item() > highest_iou:
                highest_iou = iou.item()
                highest_conf = pred['score']
        
        highest_ious.append([highest_iou, highest_conf, pred['category_id'], ann['category_id']])
    
    preds_per_image_highest[img_id] = highest_ious

