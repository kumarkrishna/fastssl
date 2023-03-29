#!/usr/bin/env bash
####SBATCH --array=0-47%16
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:1
#SBATCH --reservation=DGXA100
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/exp_track_alpha_barlowtwins.out
#SBATCH --error=sbatch_err/exp_track_alpha_barlowtwins.err
#SBATCH --job-name=exp_track_alpha_barlowtwins

. /etc/profile
module load anaconda/3
conda activate ffcv

lambd=0.00397897
pdim=3072
dataset='cifar10'
if [ $dataset = 'stl10' ]
then
    batch_size=256
else
    batch_size=512
fi

checkpt_dir=$SCRATCH/fastssl/checkpoints

python scripts/train_model.py --config-file configs/cc_barlow_twins.yaml --training.lambd=$lambd --training.projector_dim=$pdim --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.batch_size=$batch_size --training.track_alpha=True --training.log_interval=2

# precache features, should take ~35 seconds
python scripts/train_model.py --config-file configs/cc_precache.yaml --training.lambd=$lambd --training.projector_dim=$pdim --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.batch_size=$batch_size --training.model=resnet50feat
# precache embeddings, should take ~35 seconds
python scripts/train_model.py --config-file configs/cc_precache.yaml --training.lambd=$lambd --training.projector_dim=$pdim --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.batch_size=$batch_size --training.model=resnet50proj

# run linear eval on model
# TODO: make linear eval compatible with precached features
python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=$lambd --training.projector_dim=$pdim --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=1
python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=$lambd --training.projector_dim=$pdim --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=2
python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=$lambd --training.projector_dim=$pdim --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=3
# dataset='stl10'

cp $SLURM_TMPDIR/*.pth $checkpt_dir/resnet50_checkpoints/
