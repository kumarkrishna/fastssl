#!/usr/bin/env bash
#SBATCH --array=0-11%10
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:2
#SBATCH --mem=32GB
#SBATCH --time=10:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/multiPatch_barlow_fracTrain_imagenet100_results.%A.%a.out
#SBATCH --error=sbatch_err/multiPatch_barlow_fracTrain_imagenet100_results.%A.%a.err
#SBATCH --job-name=multiPatch_barlow_fracTrain_imagenet100_results

. /etc/profile
module load anaconda/3
conda activate ffcv_new
# alias python=$HOME/.conda/envs/ffcv_new/bin/python
WANDB__SERVICE_WAIT=300
which python

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR="127.0.0.1"
echo $SLURM_JOBID, $SLURM_ARRAY_TASK_ID, $MASTER_PORT, $MASTER_ADDR

# lambd_arr=(0.02 0.002 0.004 0.007 0.01 0.02 0.04 0.002 0.004 0.006 0.008 0.01 0.02 0.012 0.014 0.015 0.016 0.018)
# augs_arr=(2 4 4 4 4 4 4 8 8 8 8 8 8 8 8 8 8 8)
# lenL=${#lambd_arr[@]}
# lenA=${#augs_arr[@]}
# aidx=$SLURM_ARRAY_TASK_ID
# lidx=$SLURM_ARRAY_TASK_ID

pdim_arr=(8192   512  512   512)
lambd_arr=(0.001 0.02 0.004 0.01)
augs_arr=(2      2    4     8)
lenL=${#lambd_arr[@]}
lenA=${#augs_arr[@]}
sidx=$((SLURM_ARRAY_TASK_ID/lenL))
cfgidx=$((SLURM_ARRAY_TASK_ID%lenL))
pidx=$cfgidx
lidx=$cfgidx
aidx=$cfgidx

lambd=${lambd_arr[$lidx]}
num_augs=${augs_arr[$aidx]}
pdim=${pdim_arr[$pidx]}
# dataset='imagenet'
dataset='imagenet100'
# if [ $num_augs = 8 ]
# then
#     batch_size=125
# else
#     batch_size=128
# fi
batch_size=128

wandb_group='eigengroup'
wandb_projname='multiPatch_Barlow_fractrain-resnet18-hparam-imagenet100'

checkpt_dir=$SCRATCH/fastssl/checkpoints_imagenet100_barlow_mp_fracTrain

if [ $num_augs = 4 ]
then
    train_dpath=$SCRATCH/ffcv/ffcv_datasets/$dataset/train_frac_0.50_500_0.50_90.ffcv
elif [ $num_augs = 8 ]
then
    train_dpath=$SCRATCH/ffcv/ffcv_datasets/$dataset/train_frac_0.25_500_0.50_90.ffcv
else
    train_dpath=$SCRATCH/ffcv/ffcv_datasets/$dataset/train_no10k_500_0.50_90.ffcv
fi
val_dpath=$SCRATCH/ffcv/ffcv_datasets/$dataset/val_500_0.50_90.ffcv
num_workers=4

model=resnet18proj
# epochs=40
epochs=100
log_epochs=5

# Let's train a SSL (BarlowTwins) model with the above hyperparams
$HOME/.conda/envs/ffcv_new/bin/python -m torch.distributed.run \
                --nproc_per_node 2 --nnodes=1 \
                --rdzv_backend c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
                scripts/train_model_imagenet_distributed.py \
                --config-file configs/cc_barlow_twins.yaml \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.batch_size=$batch_size --training.seed=$sidx \
                --training.ckpt_dir=$checkpt_dir --training.num_augmentations=$num_augs \
                --training.epochs=$epochs --training.log_interval=$log_epochs \
                --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                --logging.wandb_project=$wandb_projname \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath \
                --training.num_workers=$num_workers \
                --eval.num_augmentations_pretrain=$num_augs --training.distributed=True

model=resnet18feat
train_dpath=$SCRATCH/ffcv/ffcv_datasets/$dataset/train_no10k_500_0.50_90.ffcv
val_dpath=$SCRATCH/ffcv/ffcv_datasets/$dataset/val_500_0.50_90.ffcv

for ep in {5..100..5}
# for ep in {40..40..5}
do
    echo "Running eval for checkpoint at epoch " $ep
    # Let's precache features
    $HOME/.conda/envs/ffcv_new/bin/python scripts/train_model.py \
                    --config-file configs/cc_precache.yaml \
                    --training.model=$model --training.dataset=$dataset \
                    --training.lambd=$lambd --training.projector_dim=$pdim \
                    --training.batch_size=$batch_size --training.seed=$sidx \
                    --training.ckpt_dir=$checkpt_dir --eval.num_augmentations_pretrain=$num_augs \
                    --eval.epoch=$ep --training.num_workers=$num_workers \
                    --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

    # run linear eval on precached features from model: using default seed 42
    $HOME/.conda/envs/ffcv_new/bin/python scripts/train_model.py \
                    --config-file configs/cc_classifier.yaml \
                    --training.model=$model --training.dataset=$dataset \
                    --training.lambd=$lambd --training.projector_dim=$pdim \
                    --training.batch_size=$batch_size --training.seed=$sidx \
                    --training.ckpt_dir=$checkpt_dir --eval.num_augmentations_pretrain=$num_augs \
                    --eval.epoch=$ep \
                    --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                    --logging.wandb_project=$wandb_projname \
                    --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath
done
