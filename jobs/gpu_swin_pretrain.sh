#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition gpu
#SBATCH --job-name=gpujob
#SBATCH --mem=85000M


export YOLOX_DATADIR=~/scratch/

module load lang/python/anaconda/3.8.8-2021.05-torch
module load lang/gcc/9.3.0

cd "${SLURM_SUBMIT_DIR}"
cd ../..
source work-env/bin/activate

cd yolox-pytorch

python train.py gpus='0,1,2,3' backbone="Swin-l" num_epochs=50 exp_id="Swin_l_pretrained_yolo_l" use_amp=True val_intervals=2 data_num_workers=14 batch_size=20 random_size=[14,24] input_size=[608,608] test_size=[608,608] swin_pretrained=True swin_weights_path='weights/swin_t.pth'

sleep 60
