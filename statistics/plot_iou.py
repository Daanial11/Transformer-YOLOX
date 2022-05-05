import enum
import json
import matplotlib.pyplot as plt
import numpy as np



m_o="o"



buckets = [round(x,1) for x in np.arange(0.0, 1.1, 0.1)]


data = {"swin":[], "yolo":[], "cov": []}

swin_total_iou = 0
yolo_total_iou = 0
cov_total_iou = 0
num_swin = 0
num_cov = 0
num_yolo = 0

for fold in range(1, 6):
    swin_path = f"data/iou_data_{m_o}/iou_swin_f{fold}.json"
    yolo_path = f"data/iou_data_{m_o}/iou_yolo_f{fold}.json"
    cov_path = f"data/iou_data_{m_o}/iou_cov_f{fold}.json"

    with open(swin_path, 'r') as f:
        swin_data = json.load(f)

    with open(yolo_path, 'r') as f:
        yolo_data = json.load(f)
    
    with open(cov_path, 'r') as f:
        cov_data = json.load(f)

    for k,v in swin_data.items():
        for result_data in v:
            iou=result_data[0]
            swin_total_iou+=iou
            num_swin+=1
    
    for k,v in yolo_data.items():
        for result_data in v:
            iou=result_data[0]
            yolo_total_iou+=iou
            num_yolo+=1

    for k,v in cov_data.items():
        for result_data in v:
            iou=result_data[0]
            cov_total_iou+=iou
            num_cov+=1
    

    for i, bucket_value in enumerate(buckets):
        if i == len(buckets)-1:
            break
        
        b1 = bucket_value
        b2 = buckets[i+1]
        for k,v in swin_data.items():
            for result_data in v:
                iou = result_data[0]
                if iou >=b1 and iou <b2:
                    data["swin"].append(round(b1+0.05,2))

    for i, bucket_value in enumerate(buckets):
        if i == len(buckets)-1:
            break
        
        b1 = bucket_value
        b2 = buckets[i+1]
        for k,v in yolo_data.items():
            for result_data in v:
                iou=result_data[0]
                if iou >=b1 and iou <b2:
                    data["yolo"].append(round(b1+0.05,2))

    for i, bucket_value in enumerate(buckets):
        if i == len(buckets)-1:
            break
        
        b1 = bucket_value
        b2 = buckets[i+1]
        for k,v in cov_data.items():
            for result_data in v:
                iou=result_data[0]
                if iou >=b1 and iou <b2:
                    data["cov"].append(round(b1+0.05,2))




avg_swin = swin_total_iou/num_swin
avg_yolo = yolo_total_iou/num_yolo
avg_cov = cov_total_iou/num_cov

print(f"swin: {avg_swin}, count {num_swin}")
print(f"yolo: {avg_yolo}, count {num_yolo}")
print(f"cov: {avg_cov}, count {num_cov}")


plt.figure(figsize=(11,8))
plt.grid(alpha=0.4, axis='y', zorder=5)
plt.hist([data["swin"], data["yolo"], data["cov"]], bins=10, range=[0.0, 1.0], zorder=2)
plt.xlabel('IoU', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.legend(['Swin-Transformer', 'CSP-Darknet', 'ConvNeXt'], fontsize=18)

plt.show()




