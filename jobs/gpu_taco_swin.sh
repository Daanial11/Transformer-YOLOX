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

python train.py gpus='0,1,2,3' backbone="YOLO-l" num_epochs=100 exp_id="Taco_yolo_100_epoch" use_amp=True val_intervals=2 data_num_workers=14 batch_size=32 random_size=[14,26] input_size=[608,608] test_size=[608,608] load_model="exp/Final_yolo_l_50_epoch/model_best.pth"

sleep 60
