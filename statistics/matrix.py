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

labels_array = []
preds_array = []

for img in preds_per_image_highest:
    
    preds_for_img = preds_per_image_highest[img]

    for prediction in preds_for_img:
        
        if prediction[1]>0:
            print(prediction)
            if prediction[0] < iou_threshold:
                labels_array.append(0)
                preds_array.append(prediction[2])

            else:
                labels_array.append(prediction[3])
                preds_array.append(prediction[2])

            




categdf=pd.DataFrame({
                 'y_true': labels_array,
                 'y_pred': preds_array
                })

cm = confusion_matrix(categdf['y_true'], categdf['y_pred'], labels=class_labels)


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot()
plt.xticks(rotation=90)
plt.tight_layout()

results_name = result_file_path.split(".")[0]
plt.savefig(f'{results_name}_matrix.png', dpi = 300)





