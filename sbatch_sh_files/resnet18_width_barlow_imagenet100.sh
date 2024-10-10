#!/usr/bin/env bash
#SBATCH --array=0-14%10
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/rn18_barlow_imagenet_widthVary.%A.%a.out
#SBATCH --error=sbatch_err/rn18_barlow_imagenet_widthVary.%A.%a.err
#SBATCH --job-name=rn18_barlow_imagenet_widthVary

. /etc/profile
module load anaconda/3
# conda activate ffcv_new
conda activate ffcv_ssl_fastssl
# alias python=$HOME/.conda/envs/ffcv_new/bin/python
WANDB__SERVICE_WAIT=300
which python

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR="127.0.0.1"
echo $SLURM_JOBID, $SLURM_ARRAY_TASK_ID, $MASTER_PORT, $MASTER_ADDR

width_arr=(8 12 16 20 24 28 32 36 40 44 48 52 56 60 64)
sidx=0
width=${width_arr[$SLURM_ARRAY_TASK_ID]}

lambd=0.001
pdim=8192
# dataset='imagenet'
dataset='imagenet100'
batch_size=256

wandb_group='eigengroup'
wandb_projname='Barlow-rn18-imagenet100-WidthScaling-ep100'

# checkpt_dir=$SCRATCH/fastssl/checkpoints_imagenet96_barlow
checkpt_dir=$SCRATCH/fastssl/checkpoints_matteo_imagenet100_ep100
train_dpath=$SCRATCH/ffcv/ffcv_datasets/$dataset/train_no10k_500_0.50_90.ffcv
val_dpath=$SCRATCH/ffcv/ffcv_datasets/$dataset/val_500_0.50_90.ffcv
num_workers=4

model=resnet18proj_width${width}
num_augs=2
# epochs=30
epochs=100
log_epochs=10

# Let's train a SSL (BarlowTwins) model with the above hyperparams
python -m torch.distributed.run --nproc_per_node 2 --nnodes=1 \
                --rdzv_backend c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
                scripts/train_model_imagenet_distributed_widthVary.py \
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

model=resnet18feat_width${width}

# Let's precache features, should take ~27 seconds (rtx8000)
python scripts/train_model_widthVary.py \
                --config-file configs/cc_precache.yaml \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.batch_size=$batch_size --training.seed=$sidx \
                --training.ckpt_dir=$checkpt_dir --eval.num_augmentations_pretrain=$num_augs \
                --eval.epoch=$epochs --training.num_workers=$num_workers \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

# run linear eval on precached features from model: using default seed 42
python scripts/train_model_widthVary.py \
                --config-file configs/cc_classifier.yaml \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.batch_size=$batch_size --training.seed=$sidx \
                --training.ckpt_dir=$checkpt_dir --eval.num_augmentations_pretrain=$num_augs \
                --eval.epoch=$epochs \
                --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                --logging.wandb_project=$wandb_projname \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath
