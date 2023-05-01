#!/bin/bash
#SBATCH --gres=gpu:1 # Request 1 GPU core. This takes up the complete computation node
#SBATCH --cpus-per-task=4 # Request 4 CPU cores. This takes up the complete computation node
#SBATCH --mem=22000M # Memory proportional to GPUs: 22000M per GPU core
#SBATCH --time=1:00:00 # DD-HH:MM:SS
#SBATCH --mail-user=yizhao.wang@umontreal.ca # Emails me when job starts, ends or fails
#SBATCH --mail-type=ALL
#SBATCH --account=def-sponsor00 # Resource Allocation Project Identifier module load python/3.9 cuda cudnn scipy-stack
module load python/3.9 cuda cudnn scipy-stack
nvidia-smi
SLURM_TMPDIR=~/scratch
SOURCEDIR=~/projects/def-sponsor00/yizhaowang
# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env

source $SLURM_TMPDIR/env/bin/activate
# Install packages on the virtualenv

pip install -r $SOURCEDIR/requirements.txt
# Prepare data
# mkdir $SLURM_TMPDIR/data
# cp -r $SOURCEDIR/torchxrayvision/tests  $SLURM_TMPDIR/data
# date


# Start training

python  $SOURCEDIR/torchxrayvision/new/model_calibrate.py nih densenet121-res224-all \
  -mdtable\
  --num_epochs 100\
  -batch_size 8\
  --model densenet \
  -name nonlocal \
  --output_dir $SOURCEDIR/torchxrayvision/output \
  --dataset_dir /home/yizhaowang/projects/def-sponsor00/fourguys 
  
date