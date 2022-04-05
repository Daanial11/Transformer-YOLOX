#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --partition gpu
#SBATCH --job-name=gpujob
#SBATCH --mem=170000M


export YOLOX_DATADIR=~/scratch/

module load lang/python/anaconda/3.8.8-2021.05-torch
module load lang/gcc/9.3.0

cd "${SLURM_SUBMIT_DIR}"
cd ../..
source work-env/bin/activate

cd yolox-pytorch

python train.py gpus='0,1,2,3,4,5,6,7' backbone="SwinTiny-s" num_epochs=100 exp_id="GPUNodetest3" use_amp=True val_intervals=2 data_num_workers=14 batch_size=12 cache=True

sleep 60
