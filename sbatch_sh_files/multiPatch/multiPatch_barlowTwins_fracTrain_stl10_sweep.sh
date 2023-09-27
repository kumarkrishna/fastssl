#!/usr/bin/env bash
#SBATCH --array=0-17%20
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=20GB
#SBATCH --time=11:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/multiPatch_barlow_fracTrain_stl10_sweep.%A.%a.out
#SBATCH --error=sbatch_err/multiPatch_barlow_fracTrain_stl10_sweep.%A.%a.err
#SBATCH --job-name=multiPatch_barlow_fracTrain_stl10_sweep

. /etc/profile
module load anaconda/3
conda activate ffcv_new
WANDB__SERVICE_WAIT=300

lambd_arr=(0.0004 0.0008 0.001 0.002 0.004 0.005 0.006 0.008 0.01)
# pdim_arr=(64 128 256 512 1024 2048 4096 8192)
pdim_arr=(256)
augs_arr=(4 8)
dataset='stl10'
if [ $dataset = 'stl10' ]
then
    batch_size=100
else
    batch_size=128
fi

lenL=${#lambd_arr[@]}
lenP=${#pdim_arr[@]}
len12=$((lenL*lenP))
aidx=$((SLURM_ARRAY_TASK_ID/len12))
lpidx=$((SLURM_ARRAY_TASK_ID%len12))
pidx=$((lpidx/lenL))
lidx=$((lpidx%lenL))
lambd=${lambd_arr[$lidx]}
pdim=${pdim_arr[$pidx]}
augs=${augs_arr[$aidx]}

wandb_group='blake-richards'
wandb_projname='multiPatch-Barlow-sweep-ep50-stl10'

model=resnet50proj

if [ $augs = 4 ]
then
    train_frac='_0.50'
elif [ $augs = 8 ]
then
    train_frac='_0.25'
else
    train_frac=''
fi
train_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/unlabeled${train_frac}.beton
val_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/test.beton

checkpt_dir=$SCRATCH/fastssl/checkpoints${train_frac}_mp_stl10_barlow

model=resnet50proj
# Let's train a SSL (BarlowTwins) model with the above hyperparams
python scripts/train_model_multiPatch.py --config-file configs/cc_barlow_twins.yaml \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.num_augmentations=$augs \
                --training.batch_size=$batch_size --training.seed=42 \
                --training.ckpt_dir=$checkpt_dir \
                --training.epochs=50 --training.log_interval=50 \
                --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                --logging.wandb_project=$wandb_projname \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

model=resnet50feat
train_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/train.beton
val_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/test.beton
# Let's precache features
python scripts/train_model_multiPatch.py --config-file configs/cc_precache.yaml \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --eval.num_augmentations_pretrain=$augs --eval.epoch=50 \
                --training.num_augmentations=16 \
                --training.batch_size=$batch_size --training.seed=42 \
                --training.ckpt_dir=$checkpt_dir \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

# run linear eval on precached features from model: using default seed 42
python scripts/train_model_multiPatch.py --config-file configs/cc_classifier.yaml \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --eval.num_augmentations_pretrain=$augs --eval.epoch=50 \
                --training.num_augmentations=16 \
                --training.batch_size=$batch_size --training.seed=42 \
                --training.ckpt_dir=$checkpt_dir \
                --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                --logging.wandb_project=$wandb_projname \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath
