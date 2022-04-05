#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --job-name=gpu_short
#SBATCH --mem=16000M


export YOLOX_DATADIR=~/scratch/

module load lang/python/anaconda/3.8.8-2021.05-torch
module load lang/gcc/9.3.0

cd "${SLURM_SUBMIT_DIR}"
cd ../..
source work-env/bin/activate

cd yolox-pytorch

python train.py gpus='0,1' backbone="SwinTiny-s" num_epochs=100 exp_id="GPUNodetest3" use_amp=True val_intervals=2 data_num_workers=4 batch_size=32 cache=True

sleep 60
