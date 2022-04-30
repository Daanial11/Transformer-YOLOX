#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition gpu_short
#SBATCH --job-name=kfold5Swin
#SBATCH --mem=85000M


export YOLOX_DATADIR=~/scratch/

module load lang/python/anaconda/3.8.8-2021.05-torch
module load lang/gcc/9.3.0


source ~/work-env/bin/activate

cd ~/Transformer-YOLOX


python evaluate.py gpus='0,1,2,3' backbone="Swin-l" load_model="~/scratch/weights/m_swin_f5.pth" test_ann="~/scratch/TACO/fold5/instances_val2017.json" test_size=[640,640]

sleep 60
