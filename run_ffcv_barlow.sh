#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:1
#SBATCH --reservation=DGXA100
#SBATCH --mem=10GB
#SBATCH --time=2:00:00
#SBATCH --output=sbatch_out/exp_run_ffcv_barlow_default.out
#SBATCH --error=sbatch_err/exp_run_ffcv_barlow_default.err
#SBATCH --job-name=exp_run_ffcv_barlow_default

. /etc/profile
module load anaconda/3
conda activate ffcv

python scripts/train_model.py --training.log_interval 50 --training.epochs 100
python scripts/train_model.py --training.algorithm linear --training.model linear --training.epochs 200