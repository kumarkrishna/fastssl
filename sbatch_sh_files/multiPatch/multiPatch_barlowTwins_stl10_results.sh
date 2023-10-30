#!/usr/bin/env bash
#SBATCH --array=0-8%10
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=20GB
#SBATCH --time=27:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/multiPatch_barlow_stl10_results.%A.%a.out
#SBATCH --error=sbatch_err/multiPatch_barlow_stl10_results.%A.%a.err
#SBATCH --job-name=multiPatch_barlow_stl10_results

. /etc/profile
module load anaconda/3
conda activate ffcv_new
WANDB__SERVICE_WAIT=300

lambd_arr=(0.008 0.0004 0.001)
# pdim_arr=(64 128 256 512 1024 2048 4096 8192)
pdim_arr=(256 8192 256)
augs_arr=(2 2 4)
dataset='stl10'
if [ $dataset = 'stl10' ]
then
    batch_size=100
else
    batch_size=128
fi

len1=${#lambd_arr[@]}
sidx=$((SLURM_ARRAY_TASK_ID/len1))
cfg_idx=$((SLURM_ARRAY_TASK_ID%len1))
lambd=${lambd_arr[$cfg_idx]}
pdim=${pdim_arr[$cfg_idx]}
augs=${augs_arr[$cfg_idx]}

wandb_group='eigengroup'
wandb_projname='multiPatch-Barlow-best-hparam-ep100-stl10'

checkpt_dir=$SCRATCH/fastssl/checkpoints_mp_stl10_barlow
train_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/unlabeled.beton
val_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/test.beton
epochs=100

model=resnet50proj
# Let's train a SSL (BarlowTwins) model with the above hyperparams
python scripts/train_model_multiPatch.py --config-file configs/cc_barlow_twins.yaml \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.num_augmentations=$augs \
                --training.batch_size=$batch_size --training.seed=$sidx \
                --training.ckpt_dir=$checkpt_dir \
                --training.epochs=$epochs --training.log_interval=5 \
                --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                --logging.wandb_project=$wandb_projname \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

model=resnet50feat
train_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/train.beton
val_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/test.beton

dataset='stl10'
for eval_i in $( eval echo {5..$epochs..5} )
do
    # Let's precache features
    python scripts/train_model_multiPatch.py --config-file configs/cc_precache.yaml \
                    --training.model=$model --training.dataset=$dataset \
                    --training.lambd=$lambd --training.projector_dim=$pdim \
                    --eval.num_augmentations_pretrain=$augs --eval.epoch=$eval_i \
                    --training.num_augmentations=16 \
                    --training.batch_size=$batch_size --training.seed=$sidx \
                    --training.ckpt_dir=$checkpt_dir \
                    --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

    # run linear eval on precached features from model: using default seed 42
    python scripts/train_model_multiPatch.py --config-file configs/cc_classifier.yaml \
                    --training.model=$model --training.dataset=$dataset \
                    --training.lambd=$lambd --training.projector_dim=$pdim \
                    --eval.num_augmentations_pretrain=$augs --eval.epoch=$eval_i \
                    --training.num_augmentations=16 \
                    --training.batch_size=$batch_size --training.seed=$sidx \
                    --training.ckpt_dir=$checkpt_dir \
                    --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                    --logging.wandb_project=$wandb_projname \
                    --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath
done

dataset='cifar10'
for eval_i in $( eval echo {5..$epochs..5} )
do
    # Let's precache features
    python scripts/train_model_multiPatch.py --config-file configs/cc_precache.yaml \
                    --training.model=$model --training.dataset=$dataset \
                    --training.lambd=$lambd --training.projector_dim=$pdim \
                    --eval.num_augmentations_pretrain=$augs --eval.epoch=$eval_i \
                    --training.num_augmentations=16 \
                    --training.batch_size=$batch_size --training.seed=$sidx \
                    --training.ckpt_dir=$checkpt_dir \
                    --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

    # run linear eval on precached features from model: using default seed 42
    python scripts/train_model_multiPatch.py --config-file configs/cc_classifier.yaml \
                    --training.model=$model --training.dataset=$dataset \
                    --training.lambd=$lambd --training.projector_dim=$pdim \
                    --eval.num_augmentations_pretrain=$augs --eval.epoch=$eval_i \
                    --training.num_augmentations=16 \
                    --training.batch_size=$batch_size --training.seed=$sidx \
                    --training.ckpt_dir=$checkpt_dir \
                    --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                    --logging.wandb_project=$wandb_projname \
                    --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath
done