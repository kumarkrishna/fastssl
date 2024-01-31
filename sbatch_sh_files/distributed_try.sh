#!/bin/bash
#SBATCH --gpus-per-task=rtx8000:1
####SBATCH --gpus=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=2
#SBATCH --mem=16G
#SBATCH --time=00:02:00
#SBATCH --partition=main

module load anaconda/3
conda activate ffcv_new

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR="127.0.0.1"

srun python scripts/distributed_test.py