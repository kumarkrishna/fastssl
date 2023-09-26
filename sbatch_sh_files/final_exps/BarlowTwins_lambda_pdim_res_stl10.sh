#!/usr/bin/env bash
#SBATCH --array=0-23%25
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=20GB
#SBATCH --time=5:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/resnet50_barlowtwins_stl10_result.%A.%a.out
#SBATCH --error=sbatch_err/resnet50_barlowtwins_stl10_result.%A.%a.err
#SBATCH --job-name=resnet50_barlowtwins_stl10_result

. /etc/profile
module load anaconda/3
conda activate ffcv_new
WANDB__SERVICE_WAIT=300

# lambd_arr=(0.04 0.0004 0.04 0.0004 0.008 0.0008 0.002 0.001 0.0004 0.0008)
lambd_arr=(0.0004 0.0004 0.0004 0.0004 0.0004 0.0004 0.0004 0.0004)
# pdim_arr=(64 128 256 512 1024 2048 4096 8192)
# rewriting the order to make sure the jobs are well distributed
# pdim_arr=(64 8192 128 4096 256 2048 2048 512 512 1024)
pdim_arr=(64 8192 128 4096 256 2048 512 1024)
# lambd_arr=(0.0008 0.002)
# pdim_arr=(512 512)
dataset='stl10'
if [ $dataset = 'stl10' ]
then
    batch_size=400
else
    batch_size=512
fi

lenL=${#lambd_arr[@]}
sidx=$((SLURM_ARRAY_TASK_ID/lenL))
cfgidx=$((SLURM_ARRAY_TASK_ID%lenL))
lambd=${lambd_arr[$cfgidx]}
pdim=${pdim_arr[$cfgidx]}

wandb_group='eigengroup'
wandb_projname='BarlowTwins-pdim-lambda-result-stl10'

checkpt_dir=$SCRATCH/fastssl/checkpoints_stl10_barlow
train_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/unlabeled.beton
val_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/test.beton

model=resnet50proj
# Let's train a SSL (BarlowTwins) model with the above hyperparams
python scripts/train_model.py --config-file configs/cc_barlow_twins.yaml \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.batch_size=$batch_size --training.seed=$sidx \
                --training.ckpt_dir=$checkpt_dir \
                --training.epochs=50 --training.log_interval=50 \
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
                --training.batch_size=$batch_size --training.seed=$sidx \
                --training.ckpt_dir=$checkpt_dir \
                --eval.epoch=50 \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

# run linear eval on precached features from model: using default seed 42
python scripts/train_model.py --config-file configs/cc_classifier.yaml \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.batch_size=$batch_size --training.seed=$sidx \
                --training.ckpt_dir=$checkpt_dir \
                --eval.epoch=50 \
                --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                --logging.wandb_project=$wandb_projname \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath
