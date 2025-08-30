#!/usr/bin/env bash
###SBATCH --array=1-4
#SBATCH --array=0-49%10
#SBATCH --partition=long
####SBATCH --gres=gpu:a100l:2
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=32GB
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=sbatch_out/barlow_finetune_sweep.%A.%a.out
#SBATCH --error=sbatch_err/barlow_finetune_sweep.%A.%a.err
#SBATCH --job-name=barlow_finetune_sweep

. /etc/profile
module load anaconda/3
conda activate fastssl_micromamba
# alias python=$HOME/.conda/envs/ffcv_new/bin/python
WANDB__SERVICE_WAIT=300

lr_head_arr=(1e-5 5e-5 1e-4 5e-4 1e-3)
lr_backbone_arr=(1e-5 5e-5 1e-4 5e-4 1e-3)
wd_arr=(0 1e-6)

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR="127.0.0.1"
echo $SLURM_JOBID, $SLURM_ARRAY_TASK_ID, $MASTER_PORT, $MASTER_ADDR

lambd=0.001
pdim=8192
# dataset='imagenet'
dataset='imagenet100'
batch_size=256
##sidx=$SLURM_ARRAY_TASK_ID
sidx=0

checkpt_dir=$SCRATCH/SSL_RSA/imagenet100_ep100_ckpts/seed_$sidx
train_dpath=$SCRATCH/ffcv/ffcv_datasets/$dataset/train_no10k_500_0.50_90.ffcv
val_dpath=$SCRATCH/ffcv/ffcv_datasets/$dataset/val_500_0.50_90.ffcv
num_workers=4

wandb_group='blake-richards'
wandb_projname='Barlow-resnet18-imagenet100-finetune-hparam'

model=resnet18feat
num_augs=2
epochs=100
eval_epoch=100
log_epochs=50


len1=${#lr_head_arr[@]}
len2=${#lr_backbone_arr[@]}
len12=$((len1*len2))
wdidx=$((SLURM_ARRAY_TASK_ID/len12))
lridx=$((SLURM_ARRAY_TASK_ID%len12))
lr2idx=$((lridx/len1))
lr1idx=$((lridx%len1))
lr_head=${lr_head_arr[$lr1idx]}
lr_backbone=${lr_backbone_arr[$lr2idx]}
wd=${wd_arr[$wdidx]}
# lr_head=1e-4
# lr_backbone=1e-4
# wd=1e-6

# copy pretrained checkpt to hparam directory for finetuning
curr_wd=$(pwd)
cd $checkpt_dir
fname='*exp_ssl_'$eval_epoch'_seed*.pth'
hparam_dir=$SCRATCH'/SSL_RSA/finetune_hparam_search/lr_head_'$lr_head'_lr_backbone_'$lr_backbone'_wd_'$wd
mkdir -p $hparam_dir
find . -name $fname -exec cp --parents \{\} $hparam_dir \;
checkpt_dir=$hparam_dir
cd $curr_wd

python -m torch.distributed.run \
                --nproc_per_node 2 --nnodes=1 \
                --rdzv_backend c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
                scripts/train_model_imagenet_distributed_v2.py \
                --config-file configs/cc_finetune.yaml \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.batch_size=$batch_size --training.seed=$sidx \
                --eval.epoch=$eval_epoch --eval.num_augmentations_pretrain=$num_augs \
                --training.ckpt_dir=$checkpt_dir \
                --finetune.lr_head=$lr_head --finetune.lr_backbone=$lr_backbone \
                --finetune.weight_decay=$wd \
                --training.epochs=$epochs --training.log_interval=$log_epochs \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath \
                --training.num_workers=$num_workers --training.distributed=True \
                --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                --logging.wandb_project=$wandb_projname

# single GPU training for debugging:
# epochs=2
# log_epochs=1
# python -m torch.distributed.run --nproc_per_node 1 --nnodes=1 --rdzv_backend c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT scripts/train_model_imagenet_distributed_v2.py --config-file configs/cc_finetune.yaml --training.model=$model --training.dataset=$dataset --training.lambd=$lambd --training.projector_dim=$pdim --training.batch_size=$batch_size --training.seed=$sidx --eval.epoch=$eval_epoch --eval.num_augmentations_pretrain=$num_augs --training.ckpt_dir=$checkpt_dir --training.epochs=$epochs --training.log_interval=$log_epochs --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath --training.num_workers=$num_workers
