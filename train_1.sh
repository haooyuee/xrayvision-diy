#!/bin/bash
#SBATCH --gres=gpu:1 # Request 1 GPU core. This takes up the complete computation node
#SBATCH --cpus-per-task=4 # Request 4 CPU cores. This takes up the complete computation node
#SBATCH --mem=22000M # Memory proportional to GPUs: 22000M per GPU core
#SBATCH --time=1:00:00 # DD-HH:MM:SS
#SBATCH --mail-user=haoyue.sheng@umontreal.ca # Emails me when job starts, ends or fails
#SBATCH --mail-type=ALL
#SBATCH --account=def-sponsor00 # Resource Allocation Project Identifier module load python/3.9 cuda cudnn scipy-stack


module load python/3.9 cuda cudnn scipy-stack
nvidia-smi
SLURM_TMPDIR=~/scratch
SOURCEDIR=~/projects/def-sponsor00/haooyuee/torchxrayvision_
# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env

source $SLURM_TMPDIR/env/bin/activate

# Install packages on the virtualenv
pip install -r $SOURCEDIR/requirements.txt


# Start training
python $SOURCEDIR/train_model_2.py \
  --dataset_dir /home/haooyuee/projects/def-sponsor00/fourguys \
  --batch_size 32 \
  --num_epochs 10