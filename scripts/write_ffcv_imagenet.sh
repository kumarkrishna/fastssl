#!/bin/bash
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --mail-type=END,FAL
#SBATCH --mail-user=lindongy@mila.quebec
#SBATCH --output=write_imagenet.out
#SBATCH --error=write_imagenet.err
#SBATCH --job-name=write_imagenet

module load anaconda/3 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
conda activate ffcv

python write_ffcv_datasets.py
python write_ffcv_double_datasets.py


