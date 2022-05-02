#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition gpu_short
#SBATCH --job-name=evalCov
#SBATCH --mem=85000M


export YOLOX_DATADIR=~/scratch/

module load lang/python/anaconda/3.8.8-2021.05-torch
module load lang/gcc/9.3.0


source ~/work-env/bin/activate

cd ~/Transformer-YOLOX


python evaluate.py gpus='0,1,2,3' backbone="Cov-l" eval_result_name="o_cov_f1" load_model="~/scratch/weights/one/o_cov_f1.pth" test_ann="~/scratch/TACO/onefold1/instances_val2017.json" test_size=[640,640]
python evaluate.py gpus='0,1,2,3' backbone="Cov-l" eval_result_name="o_cov_f2" load_model="~/scratch/weights/one/o_cov_f2.pth" test_ann="~/scratch/TACO/onefold2/instances_val2017.json" test_size=[640,640]
python evaluate.py gpus='0,1,2,3' backbone="Cov-l" eval_result_name="o_cov_f3" load_model="~/scratch/weights/one/o_cov_f3.pth" test_ann="~/scratch/TACO/onefold3/instances_val2017.json" test_size=[640,640]
python evaluate.py gpus='0,1,2,3' backbone="Cov-l" eval_result_name="o_cov_f4" load_model="~/scratch/weights/one/o_cov_f4.pth" test_ann="~/scratch/TACO/onefold4/instances_val2017.json" test_size=[640,640]
python evaluate.py gpus='0,1,2,3' backbone="Cov-l" eval_result_name="o_cov_f5" load_model="~/scratch/weights/one/o_cov_f5.pth" test_ann="~/scratch/TACO/onefold5/instances_val2017.json" test_size=[640,640]


sleep 60
