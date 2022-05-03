import json
from numpy import average
import pycocotools.coco as coco_
from pycocotools.cocoeval import COCOeval



m_o = "o"




avg_results = {}

for i in range(1, 6):
    gt_ann = f"data/annotations/{m_o}/fold{i}.json"
    results_file = f"data/results_{m_o}/result_{m_o}_yolo_f{i}.json"
    coco = coco_.COCO(gt_ann)
    coco_det = coco.loadRes(results_file)
    coco_eval = COCOeval(coco, coco_det, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    avg_results[i] = coco_eval.stats

print(avg_results)

avg_result_array = [0] * 12


for result in avg_results.values():
    
    for i, value in enumerate(result):
        if i==0:
            print(value)
        avg_result_array[i]+=value


for i, value in enumerate(avg_result_array):
    avg_result_array[i]=round(value/5.0, 3)

print(avg_result_array)



