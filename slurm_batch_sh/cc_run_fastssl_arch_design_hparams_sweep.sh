#!/usr/bin/env bash
#SBATCH --array=0-119%50
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=7:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/exp_stl10_fastssl_arch_design_hparam_sweep.%A.%a.out
#SBATCH --error=sbatch_err/exp_stl10_fastssl_arch_design_hparam_sweep.%A.%a.err
#SBATCH --job-name=exp_stl10_fastssl_arch_design_hparam_sweep

. /etc/profile
module load anaconda/3
conda activate ffcv
export LD_PRELOAD=~/Projects/SSL_alpha/fastssl/configs/hack.so 	# Olexa's hack to avoid INTERNAL ASSERT ERROR on Pytorch 1.10
# Do some more exports to avoid getting stuck at PCA
export MKL_THREADING_LAYER=TBB


# lamda_arr=(0.0 0.000001 0.000004 0.000018 0.000079 0.000341 0.001466 0.0063 0.027 0.116346 0.5 1.0)
lamda_arr=(0.005 0.011 0.022 0.044 0.1)
proj_arr=(128 256 512 768 1024 2048 3072 4096)
# lamda_arr=(0.001 0.004 0.016 0.064 0.256 0.5)
# lamda_arr=(1e-5 4e-5 1e-4 4e-4)
# lamda_arr=(4e-6)
# lamda_arr=(0.016 0.064 0.256 1e-3 1e-3 1e-5 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 4e-3 4e-4)
# lamda_arr=(0.016)
# proj_arr=(1024 2048 3072 4096)
# proj_arr=(4096 3072 2048 1024)
# proj_arr=(128 256 512 768)
# proj_arr=(128 128 128)
# proj_arr=(2048 128 128 128 2048 128 1024 128 2048 256 3072 4096 512 768 2048 1024)
# proj_arr=(768)

arch_layers_arr=('2' '4' '6')
# arch_layers_arr=('8')

lenA=${#arch_layers_arr[@]}
lenL=${#lamda_arr[@]}
lenP=${#proj_arr[@]}
lenMul=$((lenL*lenP))
aidx=$((SLURM_ARRAY_TASK_ID/lenMul))
lpidx=$((SLURM_ARRAY_TASK_ID%lenMul))
lidx=$((lpidx%lenL))
pidx=$((lpidx/lenL))

dataset='stl10'
batch_size=256
checkpt_dir='checkpoints_arch_design_hparams_'$dataset
model_name=shallowConvproj_${arch_layers_arr[$aidx]} 
python ../scripts/train_model.py --config-file configs/cc_barlow_twins.yaml --training.lambd=${lamda_arr[$lidx]} --training.projector_dim=${proj_arr[$pidx]} --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.batch_size=$batch_size

model_name=shallowConvfeat_${arch_layers_arr[$aidx]} 
dataset='cifar10'
batch_size=512
python ../scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=${lamda_arr[$lidx]} --training.projector_dim=${proj_arr[$pidx]} --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=1 --training.batch_size=$batch_size
python ../scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=${lamda_arr[$lidx]} --training.projector_dim=${proj_arr[$pidx]} --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=2 --training.batch_size=$batch_size
python ../scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=${lamda_arr[$lidx]} --training.projector_dim=${proj_arr[$pidx]} --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=3 --training.batch_size=$batch_size

dataset='stl10'
batch_size=256
python ../scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=${lamda_arr[$lidx]} --training.projector_dim=${proj_arr[$pidx]} --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=1 --training.batch_size=$batch_size
python ../scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=${lamda_arr[$lidx]} --training.projector_dim=${proj_arr[$pidx]} --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=2 --training.batch_size=$batch_size
python ../scripts/train_model.py --config-file configs/cc_classifier.yaml --training.lambd=${lamda_arr[$lidx]} --training.projector_dim=${proj_arr[$pidx]} --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=3 --training.batch_size=$batch_size
