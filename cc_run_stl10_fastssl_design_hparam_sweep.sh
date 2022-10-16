#!/usr/bin/env bash
#SBATCH --array=0-7%8
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#####SBATCH --reservation=DGXA100
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/exp_stl10_run_fastssl_design_hparam_sweep.%A.%a.out
#SBATCH --error=sbatch_err/exp_stl10_run_fastssl_design_hparam_sweep.%A.%a.err
#SBATCH --job-name=exp_stl10_run_fastssl_design_hparam_sweep

. /etc/profile
module load anaconda/3
conda activate ffcv

# lamda_arr=(0.001 0.00199474 0.00397897 0.00793701 0.01583223 0.03158114 0.06299605 0.12566053 0.25065966 0.5)
# proj_arr=(128 256 512 768 1024 2048 3072 4096)
# lamda_arr=(0.001 0.004 0.016 0.064 0.256 0.5)

lamda_arr=(1e-5 4e-5 1e-4 4e-4)
# proj_arr=(1024 2048 3072 4096)
proj_arr=(1024 2048)
# proj_arr=(128 256 512 768)
checkpt_dir='checkpoints_design_hparams_stl10'

lenL=${#lamda_arr[@]}
lidx=$((SLURM_ARRAY_TASK_ID%lenL))
pidx=$((SLURM_ARRAY_TASK_ID/lenL))

dataset='stl10'
batch_size=256
python scripts/train_model.py --config-file configs/cc_barlow_twins.yaml --training.lambd=${lamda_arr[$lidx]} --training.projector_dim=${proj_arr[$pidx]} --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.batch_size=$batch_size

# echo ../bt_ckpt_cc/ckpt_${lamda_arr[$lidx]}_${proj_arr[$pidx]}
# cp ../bt_ckpt_cc/ckpt_${lamda_arr[$lidx]}_${proj_arr[$pidx]}/*.pth $SLURM_TMPDIR/
dataset='stl10'
python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=${lamda_arr[$lidx]} --training.projector_dim=${proj_arr[$pidx]} --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=1
python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=${lamda_arr[$lidx]} --training.projector_dim=${proj_arr[$pidx]} --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=2
# python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=${lamda_arr[$lidx]} --training.projector_dim=${proj_arr[$pidx]} --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=3

dataset='cifar10'
python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=${lamda_arr[$lidx]} --training.projector_dim=${proj_arr[$pidx]} --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=1
python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=${lamda_arr[$lidx]} --training.projector_dim=${proj_arr[$pidx]} --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=2
