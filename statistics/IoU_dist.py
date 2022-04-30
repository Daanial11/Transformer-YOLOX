import json
import matplotlib as plt



result_file_path = 'result_Swin-l_640.json'
ann_path = 'instances_val2017.json'


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


for k, v in annotation_dict_imgId.items():
    print(len(v))
    print(k)
    print(v)
    break
