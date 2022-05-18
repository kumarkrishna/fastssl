#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:1
#SBATCH --reservation=DGXA100
#SBATCH --mem=10GB
#SBATCH --time=4:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/exp_run_fastssl_nocast.out
#SBATCH --error=sbatch_err/exp_run_fastssl_nocast.err
#SBATCH --job-name=exp_run_fastssl_nocast

. /etc/profile
module load anaconda/3
conda activate ffcv

python scripts/train_model.py --config-file configs/cc_barlow_twins.yaml
python scripts/train_model.py --config-file configs/cc_classifier.yaml
# python train_ffcv.py --config-file default_config.yaml
# python train_classifier.py --dataset cifar10 --epochs 200
# python train_classifier.py --dataset cifar10 --epochs 200 --use_projector True
# cp $SLURM_TMPDIR/*.csv saved_models/
# cp $SLURM_TMPDIR/*.pth saved_models/