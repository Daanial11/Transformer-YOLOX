import json
from tkinter import font
from numpy import average
import pycocotools.coco as coco_
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt

import numpy as np
from scipy.interpolate import make_interp_spline

m_o = "m"
model = "yolo"




averaged_results = {'0.05:0.95':[0]*101, '0.5':[0]*101, '0.75':[0]*101}

recall_x = []
for n in range(0, 101):
    recall_x.append(0.01*n)

for i in range(1, 6):
    gt_ann = f"data/annotations/{m_o}/fold{i}.json"
    results_file = f"data/results_{m_o}/result_{m_o}_{model}_f{i}.json"
    coco = coco_.COCO(gt_ann)
    coco_det = coco.loadRes(results_file)
    coco_eval = COCOeval(coco, coco_det, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    all_precision = coco_eval.eval['precision']
    
    #0.5
    for n in range(0, 101):
        p = all_precision[0, n, 0, 0, 2]
        averaged_results['0.5'][n] += p
        

    #0.75
    for n in range(0, 101):
        p = all_precision[5, n, 0, 0, 2]
        averaged_results['0.75'][n] += p
        
    
    #0.05:0.95
    for n in range(0, 101):

        precision_y = 0
        for iou_thres in range(0, 10):
            precision_y += all_precision[iou_thres, n, 0, 0, 2]
        
        precision_y = precision_y/10.0

        averaged_results['0.05:0.95'][n] += precision_y


for k,v in averaged_results.items():

    for i, precision_value in enumerate(v):
        averaged_results[k][i] = precision_value/5.0



        


spline1 = make_interp_spline(recall_x, averaged_results['0.5'])
spline2 = make_interp_spline(recall_x, averaged_results['0.75'])
spline3 = make_interp_spline(recall_x, averaged_results['0.05:0.95'])

 
# Returns evenly spaced numbers
# over a specified interval.
new_x = np.linspace(np.array(recall_x).min(), np.array(recall_x).max(), 500)


new_y1 = spline1(new_x)
new_y2 = spline2(new_x)
new_y3 = spline3(new_x)

plt.figure(figsize=(11,8))
plt.grid(alpha=0.4)
plt.plot(new_x, new_y1, label='IoU@0.5', linewidth=4.0)
plt.plot(new_x, new_y2, label='IoU@0.75', linewidth=4.0)
plt.plot(new_x, new_y3, label='IoU@0.05:0.95', linewidth=4.0)
plt.legend(loc="upper right", fontsize=18)
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)


plt.show()




