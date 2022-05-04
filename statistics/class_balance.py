import json
from turtle import color
import matplotlib.pyplot as plt
import numpy as np



class_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
labels = ['Bottle', 'Bottle cap', 'Can', 'Cigarette', 'Cup', 'Lid', 'Other', 'Plasic Bag', 'Pop tab', 'Straw']

count_dict = {}

with open("data/annotations/new_merged.json") as f:
    data = json.load(f)



for c in class_labels:
    count_dict[c] = 0


for annotation in data["annotations"]:
    count_dict[annotation["category_id"]] +=1

colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
cm = plt.get_cmap("Accent", len(class_labels))

plt.figure(figsize=(17,13))
plt.grid(axis='y', alpha=0.4, zorder=5)
for i in class_labels:
    plt.bar(labels[i-1], count_dict[i], color=colours[i-1], zorder=2)


plt.xticks(fontsize=18, rotation=70)
plt.yticks(fontsize=18)
plt.ylabel("Number of annotations", fontsize=20)
plt.xlabel("Category", fontsize=20)
plt.show()






