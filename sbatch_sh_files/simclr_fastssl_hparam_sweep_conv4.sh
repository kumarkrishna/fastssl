#!/usr/bin/env bash
#SBATCH --array=0-432%50
#SBATCH --partition=long
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --mem=20GB
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/simclr_mini-imagenet_hparam_sweep.%A.%a.out
#SBATCH --error=sbatch_err/simclr_mini-imagenet_hparam_sweep.%A.%a.err
#SBATCH --job-name=simclr_mini-imagenet_hparam_sweep_dual
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec

. /etc/profile
module load anaconda/3
conda activate ffcv
export LD_PRELOAD=~/fastssl/configs/hack.so 	# Olexa's hack to avoid INTERNAL ASSERT ERROR on Pytorch 1.10
# Do some more exports to avoid getting stuck at PCA
export MKL_THREADING_LAYER=TBB


temp_arr=(0.05 0.1 0.5)
proj_arr=(128 256 512 1024)
bsz_arr=(32 64 128)
lr_arr=(0.00005 0.0001 0.001 0.005)
wd_arr=(1e-9 1e-6 1e-4)

lenT=${#temp_arr[@]}
lenP=${#proj_arr[@]}
lenB=${#bsz_arr[@]}
lenLR=${#lr_arr[@]}
lenWD=${#wd_arr[@]}

lenMul12=$((lenLR*lenWD))
lenMul123=$((lenLR*lenWD*lenB))
lenMul1234=$((lenLR*lenWD*lenB*lenP))

tidx=$((SLURM_ARRAY_TASK_ID/lenMul1234))
idx1234=$((SLURM_ARRAY_TASK_ID%lenMul1234))
pidx=$((idx1234/lenMul123))
idx123=$((idx1234%lenMul123))
bidx=$((idx123/lenMul12))
idx12=$((idx123%lenMul12))
lridx=$((idx12/lenWD))
wdidx=$((idx12%lenWD))

dataset='mini-imagenet'
temp=${temp_arr[$tidx]}
projector_dim=${proj_arr[$pidx]}
batch_size=${bsz_arr[$bidx]}
lr=${lr_arr[$lridx]}
wd=${wd_arr[$wdidx]}

checkpt_dir='simclr_checkpoints_hparams_'$dataset
model_name=shallowConvprojdualstream_4 
python scripts/train_model.py --config-file configs/mila-imagenet-simCLR.yaml --training.temperature=$temp --training.projector_dim=$projector_dim --training.batch_size=$batch_size --training.lr=$lr --training.weight_decay=$wd --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.epochs=100 --training.log_interval=50

model_name=shallowConvfeatdualstream_4
python scripts/train_model.py --config-file configs/cc_classifier.yaml --eval.train_algorithm='SimCLR' --training.datadir=/network/scratch/l/lindongy/mini-imagenet --training.temperature=$temp --training.projector_dim=$projector_dim --training.batch_size=$batch_size --training.lr=$lr --training.weight_decay=$wd --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=1 --eval.epoch=100

python scripts/train_model.py --config-file configs/cc_classifier.yaml --eval.train_algorithm='SimCLR' --training.datadir=/network/scratch/l/lindongy/mini-imagenet --training.temperature=$temp --training.projector_dim=$projector_dim --training.batch_size=$batch_size --training.lr=$lr --training.weight_decay=$wd --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=2 --eval.epoch=100

python scripts/train_model.py --config-file configs/cc_classifier.yaml --eval.train_algorithm='SimCLR' --training.datadir=/network/scratch/l/lindongy/mini-imagenet --training.temperature=$temp --training.projector_dim=$projector_dim --training.batch_size=$batch_size --training.lr=$lr --training.weight_decay=$wd --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=3 --eval.epoch=100

# dataset='stl10'
# python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.temperature=${temp_arr[$idx1]} --training.projector_dim=${proj_arr[$idx2]} --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=1 --training.batch_size=$batch_size
# python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.temperature=${temp_arr[$idx1]} --training.projector_dim=${proj_arr[$idx2]} --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=2 --training.batch_size=$batch_size
# python scripts/train_model.py --config-file configs/cc_classifier.yaml --training.temperature=${temp_arr[$idx1]} --training.projector_dim=${proj_arr[$idx2]} --training.model=$model_name --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir --training.seed=3 --training.batch_size=$batch_size
