#!/usr/bin/env bash
#SBATCH --array=0-44%25
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
####SBATCH --reservation=DGXA100
#SBATCH --mem=16GB
#SBATCH --time=6:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/exp_run_fastssl_opt_hparam_sweep.%A.%a.out
#SBATCH --error=sbatch_err/exp_run_fastssl_opt_hparam_sweep.%A.%a.err
#SBATCH --job-name=exp_run_fastssl_opt_hparam_sweep

. /etc/profile
module load anaconda/3
conda activate ffcv

lamda=0.016
proj=1024
lr_arr=(0.0001 0.00017783 0.00031623 0.00056234 0.001 0.00177828 0.00316228 0.00562341 0.01)
wd_arr=(1e-8 1e-7 1e-6 1e-5 1e-4)
dataset='cifar10'
checkpt_dir='checkpoints_opt_hparams'

lenL=${#lr_arr[@]}
lidx=$((SLURM_ARRAY_TASK_ID%lenL))
widx=$((SLURM_ARRAY_TASK_ID/lenL))

python scripts/train_model.py --config-file configs/cc_barlow_twins.yaml --training.lambd=$lamda --training.projector_dim=$proj --training.lr=${lr_arr[$lidx]} --training.weight_decay=${wd_arr[$widx]} --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir
dataset='stl10'
python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=$lamda --training.projector_dim=$proj --training.lr=${lr_arr[$lidx]} --training.weight_decay=${wd_arr[$widx]} --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=1
python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=$lamda --training.projector_dim=$proj --training.lr=${lr_arr[$lidx]} --training.weight_decay=${wd_arr[$widx]} --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=2
python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=$lamda --training.projector_dim=$proj --training.lr=${lr_arr[$lidx]} --training.weight_decay=${wd_arr[$widx]} --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=3
