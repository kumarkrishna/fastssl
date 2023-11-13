#!/usr/bin/env bash
#SBATCH --array=0-79%20
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=20GB
#SBATCH --time=10:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/resnet50_barlowtwins_stl10_sweep.%A.%a.out
#SBATCH --error=sbatch_err/resnet50_barlowtwins_stl10_sweep.%A.%a.err
#SBATCH --job-name=resnet50_barlowtwins_stl10_sweep

. /etc/profile
module load anaconda/3
conda activate ffcv_new
WANDB__SERVICE_WAIT=300

lambd_arr=(0.0002 0.0004 0.0008 0.001 0.002 0.004 0.008 0.01 0.02 0.04)
# pdim_arr=(64 128 256 512 1024 2048 4096 8192)
# rewriting the order to make sure the jobs are well distributed
pdim_arr=(64 8192 128 4096 256 2048 512 1024)
dataset='stl10'
if [ $dataset = 'stl10' ]
then
    batch_size=400
else
    batch_size=512
fi

lenL=${#lambd_arr[@]}
lenP=${#pdim_arr[@]}
pidx=$((SLURM_ARRAY_TASK_ID/lenL))
lidx=$((SLURM_ARRAY_TASK_ID%lenL))
lambd=${lambd_arr[$lidx]}
pdim=${pdim_arr[$pidx]}

wandb_group='blake-richards'
wandb_projname='Barlow-resnet50-hparam-stl10-v2'

checkpt_dir=$SCRATCH/fastssl/checkpoints_stl10_barlow
train_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/unlabeled.beton
val_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/test.beton

model=resnet50proj
epochs=100
# Let's train a SSL (BarlowTwins) model with the above hyperparams
python scripts/train_model.py --config-file configs/cc_barlow_twins.yaml \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.batch_size=$batch_size --training.seed=42 \
                --training.ckpt_dir=$checkpt_dir \
                --training.epochs=$epochs --training.log_interval=$epochs \
                --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                --logging.wandb_project=$wandb_projname \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

model=resnet50feat
train_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/train.beton
val_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/test.beton
# Let's precache features, should take ~27 seconds (rtx8000)
python scripts/train_model.py --config-file configs/cc_precache.yaml \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.batch_size=$batch_size --training.seed=42 \
                --training.ckpt_dir=$checkpt_dir \
                --eval.epoch=$epochs \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

# run linear eval on precached features from model: using default seed 42
python scripts/train_model.py --config-file configs/cc_classifier.yaml \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.batch_size=$batch_size --training.seed=42 \
                --training.ckpt_dir=$checkpt_dir \
                --eval.epoch=$epochs \
                --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                --logging.wandb_project=$wandb_projname \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath
