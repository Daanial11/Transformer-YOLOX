#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition gpu_short
#SBATCH --job-name=evalSwin
#SBATCH --mem=85000M


export YOLOX_DATADIR=~/scratch/

module load lang/python/anaconda/3.8.8-2021.05-torch
module load lang/gcc/9.3.0


source ~/work-env/bin/activate

cd ~/Transformer-YOLOX


python evaluate.py gpus='0,1,2,3' backbone="Swin-l" eval_result_name="o_swin_f1" taco_one=True load_model="~/scratch/weights/one/o_swin_f1.pth" test_ann="~/scratch/TACO/onefold1/instances_val2017.json" test_size=[640,640]
python evaluate.py gpus='0,1,2,3' backbone="Swin-l" eval_result_name="o_swin_f2" taco_one=True load_model="~/scratch/weights/one/o_swin_f2.pth" test_ann="~/scratch/TACO/onefold2/instances_val2017.json" test_size=[640,640]
python evaluate.py gpus='0,1,2,3' backbone="Swin-l" eval_result_name="o_swin_f3" taco_one=True load_model="~/scratch/weights/one/o_swin_f3.pth" test_ann="~/scratch/TACO/onefold3/instances_val2017.json" test_size=[640,640]
python evaluate.py gpus='0,1,2,3' backbone="Swin-l" eval_result_name="o_swin_f4" taco_one=True load_model="~/scratch/weights/one/o_swin_f4.pth" test_ann="~/scratch/TACO/onefold4/instances_val2017.json" test_size=[640,640]
python evaluate.py gpus='0,1,2,3' backbone="Swin-l" eval_result_name="o_swin_f5" taco_one=True load_model="~/scratch/weights/one/o_swin_f5.pth" test_ann="~/scratch/TACO/onefold5/instances_val2017.json" test_size=[640,640]


sleep 60
