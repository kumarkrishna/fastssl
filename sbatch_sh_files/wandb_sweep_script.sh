#!/usr/bin/env bash
#SBATCH --array=0-5
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:1
#SBATCH --reservation=DGXA100
#SBATCH --mem=16GB
#SBATCH --time=5:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/exp_track_alpha_fastssl_design_hparam_sweep.%A.%a.out
#SBATCH --error=sbatch_err/exp_track_alpha_fastssl_design_hparam_sweep.%A.%a.err
#SBATCH --job-name=exp_track_alpha_fastssl_design_hparam_sweep

cd ..
module load anaconda/3
conda activate ffcv


wandb agent eigengroup/fastssl/w04a4pys


#wandb sweep --project fastssl -e eigengroup wandb_sweep_configs/design_hyperparams_sweep.yaml