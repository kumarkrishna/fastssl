#!/usr/bin/env bash
#SBATCH --array=0-79%80
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=15:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/simclr_stl10_fastssl_hparam_sweep.%A.%a.out
#SBATCH --error=sbatch_err/simclr_stl10_fastssl_hparam_sweep.%A.%a.err
#SBATCH --job-name=simclr_stl10_fastssl_hparam_sweep

. /etc/profile
module load anaconda/3
conda activate ffcv
export LD_PRELOAD=~/Projects/SSL_alpha/fastssl/configs/hack.so 	# Olexa's hack to avoid INTERNAL ASSERT ERROR on Pytorch 1.10
# Do some more exports to avoid getting stuck at PCA
export MKL_THREADING_LAYER=TBB


# temp_arr=(0.005 0.01 0.02 0.05 0.1 0.2 0.5)
temp_arr=(0.01 0.05 0.1 0.2 0.5)
# proj_arr=(128 256 512 768 1024 2048)
proj_arr=(128 256 512 768)
# bsz_arr=(128 256 512 1024)  # for cifar10
bsz_arr=(128 192 256 384)  # for stl10


len1=${#temp_arr[@]}
len2=${#proj_arr[@]}
len3=${#bsz_arr[@]}
lenMul23=$((len2*len3))
idx1=$((SLURM_ARRAY_TASK_ID/lenMul23))
idx23=$((SLURM_ARRAY_TASK_ID%lenMul23))
idx2=$((idx23/len3))
idx3=$((idx23%len3))

dataset='stl10'
temp=${temp_arr[$idx1]}
projector_dim=${proj_arr[$idx2]}
batch_size=${bsz_arr[$idx3]}

checkpt_dir='simclr_checkpoints_arch_hparams_'$dataset
model_name=resnet50proj 
python scripts/train_model.py --config-file configs/cc_SimCLR.yaml --training.temperature=$temp --training.projector_dim=$projector_dim --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.batch_size=$batch_size

model_name=resnet50feat
dataset='cifar10'
python scripts/train_model.py --config-file configs/cc_classifier.yaml --eval.train_algorithm='SimCLR' --training.temperature=$temp --training.projector_dim=$projector_dim --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=1 --training.batch_size=$batch_size
python scripts/train_model.py --config-file configs/cc_classifier.yaml --eval.train_algorithm='SimCLR' --training.temperature=$temp --training.projector_dim=$projector_dim --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=2 --training.batch_size=$batch_size
python scripts/train_model.py --config-file configs/cc_classifier.yaml --eval.train_algorithm='SimCLR' --training.temperature=$temp --training.projector_dim=$projector_dim --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=3 --training.batch_size=$batch_size

dataset='stl10'
python scripts/train_model.py --config-file configs/cc_classifier.yaml --eval.train_algorithm='SimCLR' --training.temperature=$temp --training.projector_dim=$projector_dim --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=1 --training.batch_size=$batch_size
python scripts/train_model.py --config-file configs/cc_classifier.yaml --eval.train_algorithm='SimCLR' --training.temperature=$temp --training.projector_dim=$projector_dim --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=2 --training.batch_size=$batch_size
python scripts/train_model.py --config-file configs/cc_classifier.yaml --eval.train_algorithm='SimCLR' --training.temperature=$temp --training.projector_dim=$projector_dim --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=3 --training.batch_size=$batch_size
