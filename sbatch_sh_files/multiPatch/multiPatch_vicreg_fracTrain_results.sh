#!/usr/bin/env bash
#SBATCH --array=0-5%10
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=24GB
#SBATCH --time=12:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/multiPatch_vicreg_fracTrain_results.%A.%a.out
#SBATCH --error=sbatch_err/multiPatch_vicreg_fracTrain_results.%A.%a.err
#SBATCH --job-name=multiPatch_vicreg_fracTrain_results

. /etc/profile
module load anaconda/3
conda activate ffcv_new

lambd_arr=(5.0 75.0)
pdim_arr=(256 256)
augs_arr=(4 8)
dataset='cifar10'
if [ $dataset = 'stl10' ]
then
    batch_size=64
else
    batch_size=128
fi

len1=${#lambd_arr[@]}
sidx=$((SLURM_ARRAY_TASK_ID/len1))
cfg_idx=$((SLURM_ARRAY_TASK_ID%len1))
lambd=${lambd_arr[$cfg_idx]}
mu=${lambd_arr[$cfg_idx]}
pdim=${pdim_arr[$cfg_idx]}
augs=${augs_arr[$cfg_idx]}

wandb_group='eigengroup'
wandb_projname='multiPatch-VICReg-best-hparam-ep100'

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
train_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/train${train_frac}.beton
val_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/test.beton

checkpt_dir=$SCRATCH/fastssl/checkpoints${train_frac}_vicreg_mp

# Let's train a SSL (BarlowTwins) model with the above hyperparams
python scripts/train_model_multiPatch.py --config-file configs/cc_VICReg.yaml \
                --training.lambd=$lambd --training.mu=$mu \
                --training.projector_dim=$pdim \
                --training.num_augmentations=$augs --training.seed=$sidx \
                --training.epochs=100 --training.log_interval=5 \
                --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                --training.batch_size=$batch_size --training.model=$model \
                --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                --logging.wandb_project=$wandb_projname \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

model=resnet50feat
train_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/train.beton
val_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/test.beton

for eval_i in {5..100..5}
do
    # Let's precache features
    python scripts/train_model_multiPatch.py --config-file configs/cc_precache.yaml \
                    --eval.train_algorithm='VICReg' \
                    --training.lambd=$lambd --training.mu=$mu \
                    --training.projector_dim=$pdim \
                    --eval.num_augmentations_pretrain=$augs --eval.epoch=$eval_i \
                    --training.num_augmentations=16 --training.seed=$sidx \
                    --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                    --training.batch_size=$batch_size --training.model=$model \
                    --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

    # run linear eval on precached features from model
    python scripts/train_model_multiPatch.py --config-file configs/cc_classifier.yaml \
                    --eval.train_algorithm='VICReg' \
                    --training.lambd=$lambd --training.mu=$mu \
                    --training.projector_dim=$pdim \
                    --eval.num_augmentations_pretrain=$augs --eval.epoch=$eval_i \
                    --training.num_augmentations=16 \
                    --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                    --training.batch_size=$batch_size --training.model=$model \
                    --training.seed=$sidx \
                    --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                    --logging.wandb_project=$wandb_projname \
                    --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath
done

dataset='stl10'
for eval_i in {5..100..5}
do
    # Let's precache features
    python scripts/train_model_multiPatch.py --config-file configs/cc_precache.yaml \
                    --eval.train_algorithm='VICReg' \
                    --training.lambd=$lambd --training.mu=$mu \
                    --training.projector_dim=$pdim \
                    --eval.num_augmentations_pretrain=$augs --eval.epoch=$eval_i \
                    --training.num_augmentations=16 --training.seed=$sidx \
                    --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                    --training.batch_size=$batch_size --training.model=$model \
                    --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

    # run linear eval on precached features from model
    python scripts/train_model_multiPatch.py --config-file configs/cc_classifier.yaml \
                    --eval.train_algorithm='VICReg' \
                    --training.lambd=$lambd --training.mu=$mu \
                    --training.projector_dim=$pdim \
                    --eval.num_augmentations_pretrain=$augs --eval.epoch=$eval_i \
                    --training.num_augmentations=16 \
                    --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                    --training.batch_size=$batch_size --training.model=$model \
                    --training.seed=$sidx \
                    --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                    --logging.wandb_project=$wandb_projname \
                    --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath
done
