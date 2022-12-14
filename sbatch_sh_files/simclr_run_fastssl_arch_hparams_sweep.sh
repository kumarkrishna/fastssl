#!/usr/bin/env bash
#SBATCH --array=0-53%60
#SBATCH --partition=long
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --mem=16GB
#SBATCH --time=3:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/simclr_cifar10_fastssl_arch_hparam_sweep.%A.%a.out
#SBATCH --error=sbatch_err/simclr_cifar10_fastssl_arch_hparam_sweep.%A.%a.err
#SBATCH --job-name=simclr_cifar10_fastssl_arch_hparam_sweep

. /etc/profile
module load anaconda/3
conda activate ffcv
export LD_PRELOAD=~/Projects/SSL_alpha/fastssl/configs/hack.so 	# Olexa's hack to avoid INTERNAL ASSERT ERROR on Pytorch 1.10
# Do some more exports to avoid getting stuck at PCA
export MKL_THREADING_LAYER=TBB


# temp_arr=(0.005 0.01 0.02 0.05 0.1 0.2 0.5)
# proj_arr=(128 256 512 768 1024 2048)
temp_arr=(0.05 0.2 0.5)
bsz_arr=(64 128 256 512 768 1024)

arch_layers_arr=('2' '4' '6')
# arch_layers_arr=('8')

lenA=${#arch_layers_arr[@]}
len1=${#temp_arr[@]}
# len2=${#proj_arr[@]}
len2=${#bsz_arr[@]}
lenMul=$((len1*len2))
aidx=$((SLURM_ARRAY_TASK_ID/lenMul))
idx12=$((SLURM_ARRAY_TASK_ID%lenMul))
idx1=$((idx12%len1))
idx2=$((idx12/len1))

dataset='cifar10'
# batch_size=512
# temp=${temp_arr[$idx1]}
# projector_dim=${proj_arr[$idx2]}
batch_size=${bsz_arr[$idx2]}
temp=${temp_arr[$idx1]}
projector_dim=512
checkpt_dir='simclr_checkpoints_arch_hparams_'$dataset
model_name=shallowConvproj_${arch_layers_arr[$aidx]} 
python scripts/train_model.py --config-file configs/cc_SimCLR.yaml --training.temperature=$temp --training.projector_dim=$projector_dim --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.batch_size=$batch_size

model_name=shallowConvfeat_${arch_layers_arr[$aidx]} 
dataset='cifar10'
python scripts/train_model.py --config-file configs/cc_classifier.yaml --eval.train_algorithm='SimCLR' --training.temperature=$temp --training.projector_dim=$projector_dim --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=1 --training.batch_size=$batch_size
python scripts/train_model.py --config-file configs/cc_classifier.yaml --eval.train_algorithm='SimCLR' --training.temperature=$temp --training.projector_dim=$projector_dim --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=2 --training.batch_size=$batch_size
python scripts/train_model.py --config-file configs/cc_classifier.yaml --eval.train_algorithm='SimCLR' --training.temperature=$temp --training.projector_dim=$projector_dim --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=3 --training.batch_size=$batch_size

# dataset='stl10'
# python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.temperature=${temp_arr[$idx1]} --training.projector_dim=${proj_arr[$idx2]} --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=1 --training.batch_size=$batch_size
# python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.temperature=${temp_arr[$idx1]} --training.projector_dim=${proj_arr[$idx2]} --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=2 --training.batch_size=$batch_size
# python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.temperature=${temp_arr[$idx1]} --training.projector_dim=${proj_arr[$idx2]} --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=3 --training.batch_size=$batch_size
