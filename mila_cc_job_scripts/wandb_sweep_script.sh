#!/usr/bin/env bash
#SBATCH --array=0-10
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:1
#SBATCH --reservation=DGXA100
#SBATCH --mem=32GB
#SBATCH --time=20:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/fastssl_design_hparam_sweep_spectral.%A.%a.out
#SBATCH --error=sbatch_err/fastssl_design_hparam_sweep_spectral.%A.%a.err
#SBATCH --job-name=fastssl_design_hparam_sweep_spectral

cd ..
module load anaconda/3
conda activate ffcv



export FFCV_DATA_DIR=/home/mila/a/arnab.mondal/scratch/ffcv/ffcv_datasets
export CHECKPOINT_DIR=/home/mila/a/arnab.mondal/scratch/fastssl/checkpoints

wandb agent eigengroup/fastssl/co7pkvcm
#wandb agent eigengroup/fastssl/755bw1ry

# export FFCV_DATA_DIR=/network/projects/_groups/linclab_users/ffcv/ffcv_datasets
# export TORCHVISION_DATA_DIR=/network/datasets
# export CHECKPOINT_DIR=/network/projects/_groups/linclab_users/fastssl/checkpoints


#wandb sweep --project fastssl -e eigengroup wandb_sweep_configs/design_hyperparams_sweep.yaml