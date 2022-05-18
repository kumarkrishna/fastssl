#!/usr/bin/env bash
#SBATCH --array=0-9%10
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:1
#SBATCH --reservation=DGXA100
#SBATCH --mem=10GB
#SBATCH --time=2:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/exp_run_fastssl_lamda_sweep.%A.%a.out
#SBATCH --error=sbatch_err/exp_run_fastssl_lamda_sweep.%A.%a.err
#SBATCH --job-name=exp_run_fastssl_lamda_sweep

. /etc/profile
module load anaconda/3
conda activate ffcv

lamda_arr=(0.01 0.01544452 0.02385332 0.03684031 0.0568981 0.08787639 0.13572088 0.2096144 0.3237394 0.5)
dataset='cifar10'

# python scripts/train_model.py --config-file configs/cc_barlow_twins.yaml --training.lambd=${lamda_arr[$SLURM_ARRAY_TASK_ID]} --training.dataset=$dataset
python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=${lamda_arr[$SLURM_ARRAY_TASK_ID]} --training.dataset=$dataset

dataset='cifar100'
python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=${lamda_arr[$SLURM_ARRAY_TASK_ID]} --training.dataset=$dataset

# python train_ffcv.py --config-file default_config.yaml
# python train_classifier.py --dataset cifar10 --epochs 200
# python train_classifier.py --dataset cifar10 --epochs 200 --use_projector True
# cp $SLURM_TMPDIR/*.csv saved_models/
# cp $SLURM_TMPDIR/*.pth saved_models/