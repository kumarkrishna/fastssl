#!/usr/bin/env bash
#SBATCH --array=0-191%64
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/resnet18_width_barlowtwins.%A.%a.out
#SBATCH --error=sbatch_err/resnet18_width_barlowtwins.%A.%a.err
#SBATCH --job-name=resnet18_width_barlowtwins

. /etc/profile
module load anaconda/3
conda activate ffcv

dataset='cifar10'
if [ $dataset = 'stl10' ]
then
    batch_size=256
else
    batch_size=512
fi

SEEDS=3

width=$((1+SLURM_ARRAY_TASK_ID/SEEDS))
seed=$((SLURM_ARRAY_TASK_ID%SEEDS))
lambd=0.005
pdim=2048
noise_level=15

model=resnet18proj_width${width}
if [ $noise_level = 15 ]
then
    checkpt_dir=$SCRATCH/fastssl/checkpoints_matteo_Noise15
else
    checkpt_dir=$SCRATCH/fastssl/checkpoints_matteo
fi

# Let's train a SSL (BarlowTwins) model with the above hyperparams
python scripts/train_model_widthVary.py --config-file configs/cc_barlow_twins.yaml \
                    --training.lambd=$lambd --training.projector_dim=$pdim \
                    --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                    --training.batch_size=$batch_size --training.model=$model \
                    --training.seed=$seed

model=resnet18feat_width${width}
if [ $noise_level = 15 ]
then
    trainset=/network/projects/_groups/linclab_users/ffcv/ffcv_datasets/{dataset}/Noise_15
    testset=/network/projects/_groups/linclab_users/ffcv/ffcv_datasets/{dataset}/Noise_15
else
    trainset=/network/projects/_groups/linclab_users/ffcv/ffcv_datasets/{dataset}
    testset=/network/projects/_groups/linclab_users/ffcv/ffcv_datasets/{dataset}
fi

# Let's precache features, should take ~35 seconds (rtx8000)
python scripts/train_model_widthVary.py --config-file configs/cc_precache.yaml \
                    --training.lambd=$lambd --training.projector_dim=$pdim \
                    --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                    --training.batch_size=$batch_size --training.model=$model \
                    --training.seed=$seed \
                    --training.train_dataset=${trainset}/train.beton \
                    --training.val_dataset=${testset}/test.beton

# run linear eval on precached features from model: using default seed 42
python scripts/train_model_widthVary.py --config-file configs/cc_classifier.yaml \
                    --training.lambd=$lambd --training.projector_dim=$pdim \
                    --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                    --training.batch_size=$batch_size --training.model=$model \
                    --training.seed=$seed \
                    --training.train_dataset=${trainset}/train.beton \
                    --training.val_dataset=${testset}/test.beton

# cp $SLURM_TMPDIR/*.pth $checkpt_dir/resnet18_checkpoints/
