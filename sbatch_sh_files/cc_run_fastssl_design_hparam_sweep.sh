#!/usr/bin/env bash
#SBATCH --array=0-79%8
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:1
#SBATCH --reservation=DGXA100
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/exp_run_fastssl_design_hparam_sweep.%A.%a.out
#SBATCH --error=sbatch_err/exp_run_fastssl_design_hparam_sweep.%A.%a.err
#SBATCH --job-name=exp_run_fastssl_design_hparam_sweep

. /etc/profile
module load anaconda/3
conda activate ffcv

lamda_arr=(0.001 0.00199474 0.00397897 0.00793701 0.01583223 0.03158114 0.06299605 0.12566053 0.25065966 0.5)
proj_arr=(128 256 512 768 1024 2048 3072 4096)
dataset='cifar10'
checkpt_dir='checkpoints_design_hparams'

lenL=${#lamda_arr[@]}
lidx=$((SLURM_ARRAY_TASK_ID%lenL))
pidx=$((SLURM_ARRAY_TASK_ID/lenL))

python scripts/train_model.py --config-file configs/cc_barlow_twins.yaml --training.lambd=${lamda_arr[$lidx]} --training.projector_dim=${proj_arr[$pidx]} --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir
dataset='stl10'
python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=${lamda_arr[$lidx]} --training.projector_dim=${proj_arr[$pidx]} --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=1
python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=${lamda_arr[$lidx]} --training.projector_dim=${proj_arr[$pidx]} --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=2
python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=${lamda_arr[$lidx]} --training.projector_dim=${proj_arr[$pidx]} --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=3
