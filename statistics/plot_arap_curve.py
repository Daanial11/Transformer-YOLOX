import json
from tkinter import font
from numpy import average
import pycocotools.coco as coco_
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt

import numpy as np
from scipy.interpolate import make_interp_spline

m_o = "o"




def get_average_pr_curve_data(model, class_type):
    averaged_results = {'0.05:0.95':[0]*101}
    
    class_count = 1.0 if class_type == "o" else 10.0
    for i in range(1, 6):
        gt_ann = f"data/annotations/{class_type}/fold{i}.json"
        results_file = f"data/results_{class_type}/result_{class_type}_{model}_f{i}.json"
        coco = coco_.COCO(gt_ann)
        coco_det = coco.loadRes(results_file)
        coco_eval = COCOeval(coco, coco_det, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        all_precision = coco_eval.eval['precision']
        
        
        #0.05:0.95
        for n in range(0, 101):
            average_p_class = 0
            
            for c in range(0, int(class_count)):
                average_p_classs_iou = 0
                for iou_thres in range(0, 10):
                    average_p_classs_iou+=all_precision[iou_thres, n, c, 0, 2]
                
                average_p_class+=average_p_classs_iou/10.0
            
            averaged_results['0.05:0.95'][n] += average_p_class/class_count



    for k,v in averaged_results.items():

        for i, precision_value in enumerate(v):
            averaged_results[k][i] = precision_value/5.0

    return averaged_results['0.05:0.95']




recall_x = []
for n in range(0, 101):
    recall_x.append(0.01*n)
        
yolo_precision = get_average_pr_curve_data("yolo", m_o)
swin_precision = get_average_pr_curve_data("swin", m_o)
cov_precision = get_average_pr_curve_data("cov", m_o)

spline1 = make_interp_spline(recall_x, yolo_precision)
spline2 = make_interp_spline(recall_x, swin_precision)
spline3 = make_interp_spline(recall_x, cov_precision)

 
# Returns evenly spaced numbers
# over a specified interval.
new_x = np.linspace(np.array(recall_x).min(), np.array(recall_x).max(), 10000)


new_y1 = spline1(new_x)
new_y2 = spline2(new_x)
new_y3 = spline3(new_x)

plt.figure(figsize=(11,8))
plt.grid(alpha=0.4)
plt.plot(new_x, new_y1, label='CSP-Darknet', linewidth=2.0)
plt.plot(new_x, new_y2, label='Swin-Transformer', linewidth=2.0)
plt.plot(new_x, new_y3, label='ConvNeXt', linewidth=2.0)
plt.legend(loc="upper right", fontsize=18)
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)


plt.show()




