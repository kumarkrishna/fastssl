#!/usr/bin/env bash
#SBATCH --array=0-14%15
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/resnet18_barlowtwins.%A.%a.out
#SBATCH --error=sbatch_err/resnet18_barlowtwins.%A.%a.err
#SBATCH --job-name=resnet18_barlowtwins

. /etc/profile
module load anaconda/3
conda activate ffcv

lambd_arr=(0.001 0.005 0.01 0.05 0.1)
pdim_arr=(512 1024 2048)
dataset='cifar10'
if [ $dataset = 'stl10' ]
then
    batch_size=256
else
    batch_size=512
fi

lenL=${#lambd_arr[@]}
pidx=$((SLURM_ARRAY_TASK_ID/lenL))
lidx=$((SLURM_ARRAY_TASK_ID%lenL))
lambd=${lambd_arr[$lidx]}
pdim=${pdim_arr[$pidx]}

model=resnet18proj
checkpt_dir=$SCRATCH/fastssl/checkpoints

# Let's train a SSL (BarlowTwins) model with the above hyperparams
python scripts/train_model.py --config-file configs/cc_barlow_twins.yaml \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                --training.batch_size=$batch_size --training.model=$model

model=resnet18feat
# Let's precache features, should take ~35 seconds (rtx8000)
python scripts/train_model.py --config-file configs/cc_precache.yaml \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                --training.batch_size=$batch_size --training.model=$model
# Let's precache embeddings, should take ~35 seconds (rtx8000)
# python scripts/train_model.py --config-file configs/cc_precache.yaml --training.lambd=$lambd --training.projector_dim=$pdim --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.batch_size=$batch_size --training.model=resnet50proj

# run linear eval on precached features from model: using default seed 42
python scripts/train_model.py --config-file configs/cc_classifier.yaml \
                --training.lambd=$lambd --training.projector_dim=$pdim \
                --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                --training.seed=42 --training.model=$model

# dataset='stl10'

# cp $SLURM_TMPDIR/*.pth $checkpt_dir/resnet50_checkpoints/
