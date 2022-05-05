#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition gpu_short
#SBATCH --job-name=evalYOlo
#SBATCH --mem=85000M


export YOLOX_DATADIR=~/scratch/

module load lang/python/anaconda/3.8.8-2021.05-torch
module load lang/gcc/9.3.0


source ~/work-env/bin/activate

cd ~/Transformer-YOLOX


python evaluate.py gpus='0,1,2,3' backbone="Yolo-l" eval_result_name="m_yolo_f1" load_model="~/scratch/weights/multi/m_yolo_f1.pth" test_ann="~/scratch/TACO/fold1/instances_val2017.json" test_size=[640,640]
python evaluate.py gpus='0,1,2,3' backbone="Yolo-l" eval_result_name="m_yolo_f2" load_model="~/scratch/weights/multi/m_yolo_f2.pth" test_ann="~/scratch/TACO/fold2/instances_val2017.json" test_size=[640,640]
python evaluate.py gpus='0,1,2,3' backbone="Yolo-l" eval_result_name="m_yolo_f3" load_model="~/scratch/weights/multi/m_yolo_f3.pth" test_ann="~/scratch/TACO/fold3/instances_val2017.json" test_size=[640,640]
python evaluate.py gpus='0,1,2,3' backbone="Yolo-l" eval_result_name="m_yolo_f4" load_model="~/scratch/weights/multi/m_yolo_f4.pth" test_ann="~/scratch/TACO/fold4/instances_val2017.json" test_size=[640,640]
python evaluate.py gpus='0,1,2,3' backbone="Yolo-l" eval_result_name="m_yolo_f5" load_model="~/scratch/weights/multi/m_yolo_f5.pth" test_ann="~/scratch/TACO/fold5/instances_val2017.json" test_size=[640,640]


sleep 60