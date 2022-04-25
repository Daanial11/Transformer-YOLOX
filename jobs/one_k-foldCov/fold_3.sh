#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition gpu
#SBATCH --job-name=kf3Cov
#SBATCH --mem=85000M


export YOLOX_DATADIR=~/scratch/

module load lang/python/anaconda/3.8.8-2021.05-torch
module load lang/gcc/9.3.0


source ~/work-env/bin/activate

cd ~/Transformer-YOLOX

python train.py gpus='0,1,2,3' backbone="Cov-l" num_epochs=44 exp_id="fold_Cov_3" taco_one=True fold=3 no_aug_epochs=5 save_epoch=2 use_amp=True val_intervals=2 data_num_workers=14 basic_lr_per_img=0.001 batch_size=16 freeze_backbone=True use_amp=True random_size=[14,26] input_size=[640,640] test_size=[640,640] load_model="~/scratch/exp/Covnext-YOLO/model_50.pth"

sleep 60
