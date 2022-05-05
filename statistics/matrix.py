import torch
import json
import matplotlib.pyplot as plt
import torch
import torchvision.ops.boxes as box
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd

from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from PIL import Image, ImageDraw, ImageOps


class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
display_labels = ['BG', 'Bottle', 'Bottle cap', 'Can', 'Cigarette', 'Cup', 'Lid', 'Other', 'PB/W', 'Pop tab', 'Straw']

"""class_labels = [0, 1]
display_labels = ['BG', 'Litter']"""


iou_threshold = 0.45
conf_threshold = 0.25


labels_array = []
preds_array = []


m_o = "m"
model= "cov"

for fold in range(1, 6):
    iou_data_path = f"data/iou_data_{m_o}/iou_{model}_f{fold}.json"

    with open(iou_data_path, 'r') as f:
        iou_result_data = json.load(f)

    for img, results_for_img in iou_result_data.items():
        for result in results_for_img:
            if result[1]>conf_threshold:
                if result[0] > iou_threshold:
                    labels_array.append(result[3])
                    preds_array.append(result[2])
                else:
                    labels_array.append(0)
                    preds_array.append(result[2])



categdf=pd.DataFrame({
                 'y_true': labels_array,
                 'y_pred': preds_array
                })


cm = confusion_matrix(categdf['y_true'], categdf['y_pred'])
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=display_labels, yticklabels=display_labels)
plt.ylabel('True label', fontsize=20)
plt.xlabel('Predicted label', fontsize=20)
plt.xticks(fontsize=15, rotation=45)
plt.yticks(fontsize=15, rotation=45)

plt.savefig(f"{model}_matrix.png", bbox_inches='tight', dpi=300)




