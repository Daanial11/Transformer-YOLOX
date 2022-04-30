# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 21:50
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import os
import cv2
import torch
import tqdm
import json
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import pycocotools.coco as coco_
from pycocotools.cocoeval import COCOeval

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from confusion_matrix import calculate_iou
import torchvision.ops.boxes as bops
from config import opt
from utils.util import NpEncoder
from models.yolox import Detector


def evaluate():

    if '~' in opt.test_ann:
        opt.test_ann = os.path.expanduser(opt.test_ann)
        opt.load_model = os.path.expanduser(opt.load_model)

    detector = Detector(opt)
    gt_ann = opt.val_ann if "test_ann" not in opt.keys() else opt.test_ann
    img_dir = os.getenv("YOLOX_DATADIR", None) + "TACO/images"
    batch_size = opt.batch_size

    
    assert os.path.isfile(gt_ann), 'cannot find gt {}'.format(gt_ann)
    coco = coco_.COCO(gt_ann)
    images = coco.getImgIds()
    class_ids = sorted(coco.getCatIds())
    num_samples = len(images)

    class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    display_labels = ['BG', 'Bottle', 'Bottle cap', 'Can', 'Cigarette', 'Cup', 'Lid', 'Other', 'PB/W', 'Pop tab', 'Straw']

    

    print("==>> evaluating batch_size={}".format(batch_size))
    print('find {} samples in {}'.format(num_samples, gt_ann))

    result_file = "result_{}_{}.json".format(opt.backbone, opt.test_size[0])
    coco_res = []
    
    samples_idx = list(range(num_samples))
    iterations = int(np.ceil(num_samples / float(batch_size)))

    preds_matrix = []
    labels_matrix = []
    ious = []

    for its in tqdm.tqdm(range(iterations)):
        batch_index = samples_idx[its * batch_size: (its + 1) * batch_size]
        batch_images = []
        batch_img_ids = []
        for index in batch_index:
            img_id = images[index]
            file_name = coco.loadImgs(ids=[img_id])[0]['file_name']
            image_path = img_dir + "/" + file_name
            assert os.path.isfile(image_path), "cannot find img {}".format(image_path)
            img = cv2.imread(image_path)

            batch_images.append(img)
            batch_img_ids.append(img_id)

        batch_results = detector.run(batch_images, vis_thresh=0.001)

        
        for index in range(len(batch_images)):
            results = batch_results[index]
            img_id = batch_img_ids[index]

            ann_id = coco.getAnnIds(imgIds=[img_id])
            image_annotation = coco.loadAnns([ann_id[0]])
            gt_box = image_annotation[0]['bbox']
            gt_label = image_annotation[0]['category_id']
            highest_iou_for_image = -1.0
            current_best_prediction = None
            for res in results:
                cls, conf, bbox = res[0], res[1], res[2]
                if len(res) > 3:
                    reid_feat = res[4]
                cls_index = opt.label_name.index(cls)
                coco_res.append(
                    {'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                     'category_id': class_ids[cls_index],
                     'image_id': int(img_id),
                     'score': conf})
                
                box1 = torch.tensor([[gt_box[0], gt_box[1], gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]]], dtype=torch.float)
                box2 = torch.tensor([[bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]], dtype=torch.float)

                iou = bops.box_iou(box1, box2)
                #average iou yolo = 0.012
                #average iou swin = 0.011
                if iou > highest_iou_for_image:
                    highest_iou_for_image = iou
                    current_best_prediction = class_ids[cls_index]
                    
            
            if current_best_prediction is not None:
                preds_matrix.append(current_best_prediction)
                labels_matrix.append(gt_label)
                ious.append(highest_iou_for_image)    
        
    categdf=pd.DataFrame({
                 'y_true': labels_matrix,
                 'y_pred': preds_matrix
                })

    cm = confusion_matrix(categdf['y_true'], categdf['y_pred'], labels=class_labels)

    #rounding
    """for i in range(0, len(class_labels)):
        for j in range(0, len(class_labels)):
            value = cm[i][j]
            cm[i][j] = round(value, 2)"""

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'{opt.backbone}_matrix.png', dpi = 300)



    #use grab annotaion function from coco, pass in all ids from test set
    #test with main below

    with open(result_file, 'w') as f_dump:
        json.dump(coco_res, f_dump, cls=NpEncoder)

    if "test" in os.path.basename(gt_ann):
        print("save result to {}, you can zip the result and upload it to COCO website"
              "(https://competitions.codalab.org/competitions/20794#participate)".format(result_file))
        try:
            zip_file = result_file.replace(".json", ".zip")
            os.system("zip {} {}".format(zip_file, result_file))
            print("--> create upload file done: {}".format(zip_file))
        except:
            print("please zip it before uploading")
        

    coco_det = coco.loadRes(result_file)
    coco_eval = COCOeval(coco, coco_det, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    
    

    
    # Each class is evaluated separately
    # classes = [c["name"] for c in coco.loadCats(coco.getCatIds())]
    # for i, cat_id in enumerate(class_ids):
    #     print('-------- evaluate class: {} --------'.format(classes[i]))
    #     coco_eval.params.catIds = cat_id
    #     coco_eval.evaluate()
    #     coco_eval.accumulate()
    #     coco_eval.summarize()
   # os.remove(result_file)




if __name__ == "__main__":
    opt.load_model = opt.load_model if opt.load_model != "" else os.path.join(opt.save_dir, "model_best.pth")

    evaluate()

    


    




   
    

    
