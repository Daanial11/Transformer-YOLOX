import enum
import json
import matplotlib.pyplot as plt
import numpy as np



swin_path = "ious_Swin-l.json"
yolo_path = "ious_Yolo-l.json"



buckets = [round(x,1) for x in np.arange(0.0, 1.1, 0.1)]


data = {"swin":[], "yolo":[]}

with open(swin_path, 'r') as f:
    swin_data = json.load(f)

with open(yolo_path, 'r') as f:
    yolo_data = json.load(f)



for i, v in enumerate(buckets):
    if i == len(buckets)-1:
        break
    
    b1 = v
    b2 = buckets[i+1]
    for k,v in swin_data.items():
        for iou in v:
            if iou>0:
                if iou >=b1 and iou <=b2:
                    data["swin"].append(round(b1+0.05,2))

for i, v in enumerate(buckets):
    if i == len(buckets)-1:
        break
    
    b1 = v
    b2 = buckets[i+1]
    for k,v in yolo_data.items():
        for iou in v:
            if iou>0:
                if iou >=b1 and iou <=b2:
                    data["yolo"].append(round(b1+0.05,2))


"""for k,v in swin_data.items():
    for iou in v:
        if iou>0:
            data["swin"].append(iou)

for k,v in yolo_data.items():
    for iou in v:
        if iou>0:
            data["yolo"].append(iou)"""

plt.hist([data["swin"],data["yolo"]], bins=10)
plt.xlabel('IoU')
plt.legend(['Swin', 'Yolo'])
plt.ylabel('Count')
plt.show()




