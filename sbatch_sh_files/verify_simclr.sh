#!/usr/bin/env bash
###SBATCH --array=0-79%20
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=2:00:00  # cifar10
#####SBATCH --time=6:30:00  # stl10
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/exp_track_alpha_simclr.%A.out
#SBATCH --error=sbatch_err/exp_track_alpha_simclr.%A.err
#SBATCH --job-name=exp_track_alpha_simclr

. /etc/profile
module load anaconda/3
conda activate ffcv

temp=0.1
projector_dim=128

dataset='cifar10'
if [ $dataset = 'stl10' ]
then
    batch_size=256
else
    batch_size=512
fi

checkpt_dir=$SCRATCH/fastssl/checkpoints_simclr
model_name=resnet50proj 
python scripts/train_model.py --config-file configs/cc_SimCLR.yaml \
                            --training.temperature=$temp \
                            --training.projector_dim=$projector_dim \
                            --training.model=$model_name \
                            --training.dataset=$dataset \
                            --training.ckpt_dir=$checkpt_dir \
                            --training.batch_size=$batch_size \
			    --training.seed=42

model_name=resnet50feat
# Let's precache features, should take ~35 seconds (rtx8000)
python scripts/train_model.py --config-file configs/cc_precache.yaml \
                            --eval.train_algorithm='SimCLR' \
                            --training.model=$model_name \
                            --training.temperature=$temp \
                            --training.projector_dim=$projector_dim \
                            --training.dataset=$dataset \
                            --training.ckpt_dir=$checkpt_dir \
                            --training.batch_size=$batch_size \
                            --training.seed=42
                            

python scripts/train_model.py --config-file configs/cc_classifier.yaml \
                            --eval.train_algorithm='SimCLR' \
                            --training.model=$model_name \
                            --training.temperature=$temp \
                            --training.projector_dim=$projector_dim \
                            --training.dataset=$dataset \
                            --training.ckpt_dir=$checkpt_dir \
                            --training.batch_size=$batch_size \
                            --training.seed=42 
