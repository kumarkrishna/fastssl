#!/usr/bin/env bash
####SBATCH --array=0-47%16
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/verify_vicreg.out
#SBATCH --error=sbatch_err/verify_vicreg.err
#SBATCH --job-name=verify_vicreg

. /etc/profile
module load anaconda/3
conda activate ffcv_new

lambd=25.0
mu=25.0
pdim=2048
dataset='cifar10'
if [ $dataset = 'stl10' ]
then
    batch_size=256
else
    batch_size=512
fi

checkpt_dir=$SCRATCH/fastssl/checkpoints
train_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/train.beton
val_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/test.beton

model=resnet50proj
# Let's train a SSL (BarlowTwins) model with the above hyperparams
python scripts/train_model.py --config-file configs/cc_VICReg.yaml \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.mu=$mu \
                --training.projector_dim=$pdim  --training.ckpt_dir=$checkpt_dir \
                --training.batch_size=$batch_size \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

cp $SLURM_TMPDIR/*.pth $checkpt_dir/resnet50_checkpoints/

model=resnet50feat
# Let's precache features, should take ~35 seconds (rtx8000)
python scripts/train_model.py --config-file configs/cc_precache.yaml \
                --eval.train_algorithm='VICReg' \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.mu=$mu \
                --training.projector_dim=$pdim --training.ckpt_dir=$checkpt_dir \
                --training.batch_size=$batch_size \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath
# Let's precache embeddings, should take ~35 seconds (rtx8000)
# python scripts/train_model.py --config-file configs/cc_precache.yaml --training.lambd=$lambd --training.projector_dim=$pdim --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.batch_size=$batch_size --training.model=resnet50proj

# run linear eval on precached features from model: using default seed 42
python scripts/train_model.py --config-file configs/cc_classifier.yaml \
                --eval.train_algorithm='VICReg' \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.mu=$mu \
                --training.projector_dim=$pdim --training.ckpt_dir=$checkpt_dir \
                --training.seed=42 \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath
