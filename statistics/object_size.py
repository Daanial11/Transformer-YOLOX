from cmath import isinf, isnan
import json
import numpy as np
import matplotlib.pyplot as plt
import math

with open("data/annotations/new_merged.json") as f:
    dataset = json.load(f)


def get_image(id):
    for img in dataset["images"]:
        if id == img["id"]:
            return img
bbox_widths = []
bbox_heights = []
obj_areas_sqrt = []
obj_areas_sqrt_fraction = []
nr_small_objects = 0
for ann in dataset['annotations']:
    bbox_widths.append(ann['bbox'][2])
    bbox_heights.append(ann['bbox'][3])
    obj_area = ann['bbox'][2]*ann['bbox'][3] # ann['area']

    obj_arear = np.sqrt(obj_area)

    if not isnan(obj_arear) and not isinf(obj_arear):
        obj_areas_sqrt.append(obj_arear)
    

    img = get_image(ann['image_id'])
    img_area = (img['width'])*(img['height'])
    area_frac = np.sqrt(obj_area/img_area)
    
    if area_frac <= 1.0 and area_frac>=0:

        obj_areas_sqrt_fraction.append(area_frac)
    
    
print('According to MS COCO Evaluation. This dataset has:')
print(np.sum(np.array(obj_areas_sqrt_fraction)<0.15), 'small objects (area<32*32 px)')
print(np.sum(np.array(obj_areas_sqrt_fraction)>0.15), 'medium objects (area<96*96 px)')

plt.figure(figsize=(17,13))
plt.grid(axis='y', alpha=0.4, zorder=5)
plt.hist(obj_areas_sqrt_fraction, color = "skyblue", edgecolor='black', bins=40, zorder=2)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel("Number of objects", fontsize=20)
plt.xlabel(r'Object relative size as $\sqrt{ Bbox\_area \ /  \ Image\_area}$', fontsize=20)
plt.show()