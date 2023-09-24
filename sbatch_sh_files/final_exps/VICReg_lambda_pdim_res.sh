#!/usr/bin/env bash
#SBATCH --array=0-23%25
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=2:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/resnet50_vicreg_lambda_pdim.%A.%a.out
#SBATCH --error=sbatch_err/resnet50_vicreg_lambda_pdim.%A.%a.err
#SBATCH --job-name=resnet50_vicreg_lambda_pdim

. /etc/profile
module load anaconda/3
conda activate ffcv_new
WANDB__SERVICE_WAIT=300

# lambd_arr=(750.0 750.0 750.0 750.0 750.0 750.0 750.0 750.0)
# lambd_arr=(250.0 400.0 500.0 600.0 750.0)
# lambd_arr=(25.0 50.0 75.0 100.0 150.0 200.0)
# lambd_arr=(20.0 20.0 25.0 25.0 25.0 75.0 200.0 800.0)
# pdim_arr=(64 128 256 512 1024 2048 4096 8192)
lambd_arr=(750.0 750.0 750.0 750.0 750.0 750.0 750.0 750.0)
pdim_arr=(64 128 256 512 1024 2048 4096 8192)
dataset='cifar10'
if [ $dataset = 'stl10' ]
then
    batch_size=256
else
    batch_size=512
fi

lenL=${#lambd_arr[@]}
sidx=$((SLURM_ARRAY_TASK_ID/lenL))
cfgidx=$((SLURM_ARRAY_TASK_ID%lenL))
lambd=${lambd_arr[$cfgidx]}
mu=${lambd_arr[$cfgidx]}
pdim=${pdim_arr[$cfgidx]}

wandb_group='eigengroup'
wandb_projname='VICReg-pdim-ortho-result'

model=resnet50proj
checkpt_dir=$SCRATCH/fastssl/checkpoints_VICReg_cifar10
train_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/train.beton
val_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/test.beton

# Let's train a SSL (BarlowTwins) model with the above hyperparams
python scripts/train_model.py --config-file configs/cc_VICReg.yaml \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.mu=$mu \
                --training.projector_dim=$pdim  --training.ckpt_dir=$checkpt_dir \
                --training.batch_size=$batch_size --training.seed=$sidx \
                --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                --logging.wandb_project=$wandb_projname \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

model=resnet50feat
# Let's precache features, should take ~35 seconds (rtx8000)
python scripts/train_model.py --config-file configs/cc_precache.yaml \
                --eval.train_algorithm='VICReg' \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.mu=$mu \
                --training.projector_dim=$pdim  --training.ckpt_dir=$checkpt_dir \
                --training.batch_size=$batch_size --training.seed=$sidx \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

# run linear eval on precached features from model: using default seed 42
python scripts/train_model.py --config-file configs/cc_classifier.yaml \
                --eval.train_algorithm='VICReg' \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.mu=$mu \
                --training.projector_dim=$pdim  --training.ckpt_dir=$checkpt_dir \
                --training.batch_size=$batch_size --training.seed=$sidx \
                --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                --logging.wandb_project=$wandb_projname \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

# dataset='stl10'

# cp $SLURM_TMPDIR/*.pth $checkpt_dir/resnet50_checkpoints/
