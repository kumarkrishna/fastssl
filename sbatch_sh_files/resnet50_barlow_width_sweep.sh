#!/usr/bin/env bash
#SBATCH --array=0-239%20
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/resnet50_barlowtwins_width_sweep.%A.%a.out
#SBATCH --error=sbatch_err/resnet50_barlowtwins_width_sweep.%A.%a.err
#SBATCH --job-name=resnet50_barlowtwins_width_sweep

. /etc/profile
module load anaconda/3
conda activate ffcv_new

width_arr=(8 16 32 48 64)
lambd_arr=(0.0002 0.0004 0.0008 0.001 0.002 0.004 0.008 0.01)
pdim_arr=(256 512 1024 2048 3072 4096)
dataset='cifar10'
if [ $dataset = 'stl10' ]
then
    batch_size=256
else
    batch_size=512
fi

lenW=${#width_arr[@]}
lenL=${#lambd_arr[@]}
lenP=${#pdim_arr[@]}
lenLP=$((lenL*lenP))
widx=$((SLURM_ARRAY_TASK_ID/lenLP))
lpidx=$((SLURM_ARRAY_TASK_ID%lenLP))
pidx=$((lpidx/lenL))
lidx=$((lpidx%lenL))
width=${width_arr[$widx]}
lambd=${lambd_arr[$lidx]}
pdim=${pdim_arr[$pidx]}

wandb_group='blake-richards'
wandb_projname='BarlowTwins-resnet50-width-hparam'


model=resnet50proj_width${width}
checkpt_dir=$SCRATCH/fastssl/checkpoints_matteo


# Let's train a SSL (BarlowTwins) model with the above hyperparams
python scripts/train_model_widthVary.py --config-file configs/cc_barlow_twins.yaml \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                --training.batch_size=$batch_size --training.model=$model \
                --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                --logging.wandb_project=$wandb_projname

model=resnet50feat_width${width}
# Let's precache features, should take ~35 seconds (rtx8000)
python scripts/train_model_widthVary.py --config-file configs/cc_precache.yaml \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                --training.batch_size=$batch_size --training.model=$model

# run linear eval on precached features from model: using default seed 42
python scripts/train_model_widthVary.py --config-file configs/cc_classifier.yaml \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                --training.batch_size=$batch_size --training.model=$model \
                --training.seed=42 \
                --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                --logging.wandb_project=$wandb_projname

# dataset='stl10'

# cp $SLURM_TMPDIR/*.pth $checkpt_dir/resnet50_checkpoints/
