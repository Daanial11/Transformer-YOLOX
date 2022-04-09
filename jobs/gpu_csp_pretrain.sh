#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition gpu
#SBATCH --job-name=gpujob
#SBATCH --mem=60000M


export YOLOX_DATADIR=~/scratch/

module load lang/python/anaconda/3.8.8-2021.05-torch
module load lang/gcc/9.3.0

cd "${SLURM_SUBMIT_DIR}"
cd ../..
source work-env/bin/activate

cd Transformer-YOLOX

python train.py gpus='0,1' backbone="CSP-l" num_epochs=100 exp_id="CSP_l_pretrained_yolo_l" freeze_backbone=True csp_pretrained=True csp_weights_path="weights/yolox_l.pth" use_amp=True val_intervals=2 data_num_workers=14 batch_size=64 random_size=[14,26] input_size=[640,640] test_size=[640,640]

sleep 60
