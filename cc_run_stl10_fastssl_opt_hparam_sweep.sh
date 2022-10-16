#!/usr/bin/env bash

#SBATCH --array=0
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:1
#SBATCH --reservation=DGXA100
#SBATCH --mem=16GB
#SBATCH --time=6:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/exp_stl10_run_fastssl_opt_hparam_sweep.%A.%a.out
#SBATCH --error=sbatch_err/exp_stl10_run_fastssl_opt_hparam_sweep.%A.%a.err
#SBATCH --job-name=exp_stl10_run_fastssl_opt_hparam_sweep

. /etc/profile
module load anaconda/3
conda activate ffcv

lamda=0.001
proj=1024
lr_arr=(0.0001 0.0002 0.0004 0.001 0.002 0.004 0.01)
wd_arr=(1e-8 1e-6 1e-4)
# lr_arr=(0.0001 0.00017783 0.001)
# wd_arr=(1e-06 1e-06 0.0001)
checkpt_dir='checkpoints_opt_hparams_stl10'

lenL=${#lr_arr[@]}
lidx=$((SLURM_ARRAY_TASK_ID%lenL))
widx=$((SLURM_ARRAY_TASK_ID/lenL))
# lidx=$SLURM_ARRAY_TASK_ID
# widx=$SLURM_ARRAY_TASK_ID

dataset='stl10'
batch_size=256
python scripts/train_model.py --config-file configs/cc_barlow_twins.yaml --training.lambd=$lamda --training.projector_dim=$proj --training.lr=${lr_arr[$lidx]} --training.weight_decay=${wd_arr[$widx]} --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.batch_size=$batch_size

dataset='stl10'
python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=$lamda --training.projector_dim=$proj --training.lr=${lr_arr[$lidx]} --training.weight_decay=${wd_arr[$widx]} --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=1
python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=$lamda --training.projector_dim=$proj --training.lr=${lr_arr[$lidx]} --training.weight_decay=${wd_arr[$widx]} --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=2
# python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=$lamda --training.projector_dim=$proj --training.lr=${lr_arr[$lidx]} --training.weight_decay=${wd_arr[$widx]} --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=3

dataset='cifar10'
python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=$lamda --training.projector_dim=$proj --training.lr=${lr_arr[$lidx]} --training.weight_decay=${wd_arr[$widx]} --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=1
python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=$lamda --training.projector_dim=$proj --training.lr=${lr_arr[$lidx]} --training.weight_decay=${wd_arr[$widx]} --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=2
