#!/usr/bin/env bash
#####SBATCH --array=0-7%20
#####SBATCH --array=0-13%10
#SBATCH --array=0-9%10
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB
######SBATCH --time=18:00:00
#SBATCH --time=8:00:00
#SBATCH --cpus-per-gpu=6
#SBATCH --output=sbatch_out/resnet18_barlowtwins_imagenet_sweep.%A.%a.out
#SBATCH --error=sbatch_err/resnet18_barlowtwins_imagenet_sweep.%A.%a.err
#SBATCH --job-name=resnet18_barlowtwins_imagenet_sweep

. /etc/profile
module load anaconda/3
conda activate ffcv_new
# alias python=$HOME/.conda/envs/ffcv_new/bin/python
WANDB__SERVICE_WAIT=300
which python

# # lambd_arr=(0.0002 0.0004 0.0008 0.001 0.002 0.004 0.008 0.01 0.02 0.04)
lambd_arr=(0.001 0.005)
# # pdim_arr=(32 64 128 256 512 1024 2048 4096)
# # rewriting the order to make sure the jobs are well distributed
# pdim_arr=(32 64 4096 128 2048 256 1024 512)
pdim_arr=(256 4096 512 2048 1024)
lenL=${#lambd_arr[@]}
lenP=${#pdim_arr[@]}
pidx=$((SLURM_ARRAY_TASK_ID/lenL))
lidx=$((SLURM_ARRAY_TASK_ID%lenL))

# lambd_arr=(0.5 0.05 0.3 0.04 0.1 0.03 0.08 0.02 0.04 0.014 0.02 0.01 0.01 0.007)
# pdim_arr=(32 32 64 64 128 128 256 256 512 512 1024 1024 2048 2048)
# rewriting the order to make sure the jobs are well distributed
# lambd_arr=(0.5 0.05 0.01 0.007 0.3 0.04 0.02 0.01 0.1 0.03 0.08 0.02 0.04 0.014)
# pdim_arr=(32 32 2048 2048 64 64 1024 1024 128 128 256 256 512 512)
# lambd_arr=(0.02 0.03 0.05 0.07 0.01 0.015 0.007 0.05)
# pdim_arr=(512 512 512 512 256 256 256 256)
# lenL=${#lambd_arr[@]}
# lenP=${#pdim_arr[@]}
# pidx=$SLURM_ARRAY_TASK_ID
# lidx=$SLURM_ARRAY_TASK_ID

lambd=${lambd_arr[$lidx]}
pdim=${pdim_arr[$pidx]}
# dataset='imagenet'
dataset='imagenet100'
batch_size=512

wandb_group='blake-richards'
# wandb_projname='Barlow-resnet18-hparam-imagenet'
wandb_projname='Barlow-resnet18-hparam-imagenet100'

# checkpt_dir=$SCRATCH/fastssl/checkpoints_imagenet96_barlow
checkpt_dir=$SCRATCH/fastssl/checkpoints_imagenet100_barlow
train_dpath=$SCRATCH/ffcv/ffcv_datasets/$dataset/train_no10k_500_0.50_90.ffcv
val_dpath=$SCRATCH/ffcv/ffcv_datasets/$dataset/val_500_0.50_90.ffcv
num_workers=6

model=resnet18proj
# epochs=30
epochs=40
log_epochs=5
# Let's train a SSL (BarlowTwins) model with the above hyperparams
$HOME/.conda/envs/ffcv_new/bin/python scripts/train_model_imagenet.py \
                --config-file configs/cc_barlow_twins.yaml \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.batch_size=$batch_size --training.seed=42 \
                --training.ckpt_dir=$checkpt_dir \
                --training.epochs=$epochs --training.log_interval=$log_epochs \
                --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                --logging.wandb_project=$wandb_projname \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath \
                --training.num_workers=$num_workers

model=resnet18feat

# Let's precache features, should take ~27 seconds (rtx8000)
$HOME/.conda/envs/ffcv_new/bin/python scripts/train_model.py \
                --config-file configs/cc_precache.yaml \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.batch_size=$batch_size --training.seed=42 \
                --training.ckpt_dir=$checkpt_dir \
                --eval.epoch=$epochs \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

# run linear eval on precached features from model: using default seed 42
$HOME/.conda/envs/ffcv_new/bin/python scripts/train_model.py \
                --config-file configs/cc_classifier.yaml \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.batch_size=$batch_size --training.seed=42 \
                --training.ckpt_dir=$checkpt_dir \
                --eval.epoch=$epochs \
                --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                --logging.wandb_project=$wandb_projname \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath
