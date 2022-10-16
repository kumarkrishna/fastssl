#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:1
#SBATCH --reservation=DGXA100
#SBATCH --mem=24GB
#SBATCH --time=4:00:00
#SBATCH --cpus-per-gpu=4
##SBATCH --output=sbatch_out/exp_run_fastssl_nocast.out
##SBATCH --error=sbatch_err/exp_run_fastssl_nocast.err
#SBATCH --job-name=exp_run_fastssl

. /etc/profile
module load anaconda/3
conda activate ffcv
export LD_PRELOAD=~/Projects/SSL_alpha/fastssl/configs/hack.so 	# Olexa's hack to avoid INTERNAL ASSERT ERROR on Pytorch 1.10
# Do some more exports to avoid getting stuck at PCA
export MKL_THREADING_LAYER=TBB

lamda=0.01
proj_dim=1024

checkpt_dir='checkpoints'

dataset='stl10'
batch_size=256
python scripts/train_model.py --config-file configs/cc_barlow_twins.yaml --training.lambd=$lamda --training.projector_dim=$proj_dim --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.batch_size=$batch_size

python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=$lamda --training.projector_dim=$proj_dim --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=1 --training.batch_size=$batch_size
python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=$lamda --training.projector_dim=$proj_dim --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=2 --training.batch_size=$batch_size

# python scripts/train_model.py --config-file configs/cc_barlow_twins.yaml
# python scripts/train_model.py --config-file configs/cc_classifier.yaml
# python train_ffcv.py --config-file default_config.yaml
# python train_classifier.py --dataset cifar10 --epochs 200
# python train_classifier.py --dataset cifar10 --epochs 200 --use_projector True
# cp $SLURM_TMPDIR/*.csv saved_models/