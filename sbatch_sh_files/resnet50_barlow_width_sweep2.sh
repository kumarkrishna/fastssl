#!/usr/bin/env bash
#SBATCH --array=45-959%20
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

lambd_arr=(0.0001 0.0002 0.0004 0.0008 0.001 0.002 0.004 0.008 0.01 0.02 0.04 0.08 0.1 0.2 0.4)
dataset='cifar10'
if [ $dataset = 'stl10' ]
then
    batch_size=256
else
    batch_size=512
fi

lenL=${#lambd_arr[@]}
width=$((1+SLURM_ARRAY_TASK_ID/lenL))
lidx=$((SLURM_ARRAY_TASK_ID%lenL))
lambd=${lambd_arr[$lidx]}
pdim=$((64*width))

wandb_group='blake-richards'
wandb_projname='BarlowTwins-resnet50-widthScaling-hparam'


model=resnet50proj_width${width}
checkpt_dir=$SCRATCH/fastssl/checkpoints_matteo_width_hparam


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
