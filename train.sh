#!/bin/bash
#SBATCH --gres=gpu:1 # Request 1 GPU core. This takes up the complete computation node
#SBATCH --cpus-per-task=4 # Request 4 CPU cores. This takes up the complete computation node
#SBATCH --mem=22000M # Memory proportional to GPUs: 22000M per GPU core
#SBATCH --time=1-03:00 # DD-HH:MM:SS
#SBATCH --mail-user=haooyuee@gmail.com # Emails me when job starts, ends or fails
#SBATCH --mail-type=ALL
#SBATCH --account=def-sponsor00 # Resource Allocation Project Identifier

module load python/3.9 cuda cudnn scipy-stack

SOURCEDIR=~/projects/def-sponsor00/haooyuee/torchxrayvision_

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install packages on the virtualenv
pip install -r $SOURCEDIR/requirements.txt

# Start training
python $SOURCEDIR/scripts/train_model.py \
  --dataset_dir /home/haooyuee/projects/def-sponsor00/fourguys \
  --model resnet50 \
  --num_epochs 20