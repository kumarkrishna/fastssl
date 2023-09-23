#!/usr/bin/env bash
#SBATCH --array=0-41%20
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=2:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/resnet50_barlowtwins_lambda_pdim.%A.%a.out
#SBATCH --error=sbatch_err/resnet50_barlowtwins_lambda_pdim.%A.%a.err
#SBATCH --job-name=resnet50_barlowtwins_lambda_pdim

. /etc/profile
module load anaconda/3
conda activate ffcv_new
WANDB__SERVICE_WAIT=300

lambd_arr=(0.0004 0.0004 0.0004 0.0004 0.0004 0.0004 0.0004 0.0004 0.0004 0.0004 0.0004 0.0004 0.0004 0.0004)
# lambd_arr=(0.04 0.04 0.02 0.01 0.002 0.002 0.001 0.001 0.001 0.0008 0.001 0.0004 0.001 0.0008)
pdim_arr=(64 128 256 512 768 1024 1536 2048 2304 2560 3072 4096 6144 8192)
dataset='cifar10'
if [ $dataset = 'stl10' ]
then
    batch_size=256
else
    batch_size=512
fi

lenL=${#lambd_arr[@]}
sidx=$((SLURM_ARRAY_TASK_ID/lenL))
cfgidx=$((SLURM_ARRAY_TASK_ID%lenL))
lambd=${lambd_arr[$cfgidx]}
pdim=${pdim_arr[$cfgidx]}

wandb_group='eigengroup'
wandb_projname='BarlowTwins-pdim-lambda-result'

width=64
model=resnet50proj_width${width}
checkpt_dir=$SCRATCH/fastssl/checkpoints_BarlowTwins_cifar10


# Let's train a SSL (BarlowTwins) model with the above hyperparams
python scripts/train_model_widthVary.py --config-file configs/cc_barlow_twins.yaml \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                --training.batch_size=$batch_size --training.model=$model \
                --training.seed=$sidx \
                --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                --logging.wandb_project=$wandb_projname

model=resnet50feat_width${width}
# Let's precache features, should take ~35 seconds (rtx8000)
python scripts/train_model_widthVary.py --config-file configs/cc_precache.yaml \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                --training.batch_size=$batch_size --training.model=$model \
                --training.seed=$sidx 

# run linear eval on precached features from model: using default seed 42
python scripts/train_model_widthVary.py --config-file configs/cc_classifier.yaml \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                --training.batch_size=$batch_size --training.model=$model \
                --training.seed=$sidx \
                --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                --logging.wandb_project=$wandb_projname

# dataset='stl10'

# cp $SLURM_TMPDIR/*.pth $checkpt_dir/resnet50_checkpoints/