#!/usr/bin/env bash
#SBATCH --array=0-242%25
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=2:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/resnet50_vicreg_sweep.%A.%a.out
#SBATCH --error=sbatch_err/resnet50_vicreg_sweep.%A.%a.err
#SBATCH --job-name=resnet50_vicreg_sweep

. /etc/profile
module load anaconda/3
conda activate ffcv_new
WANDB__SERVICE_WAIT=300

lambd_arr=(0.01 0.1 1.0 5.0 10.0 25.0 50.0 75.0 100.0)
mu_arr=(0.01 0.1 1.0 5.0 10.0 25.0 50.0 75.0 100.0)
pdim_arr=(256 1024 8192)
augs_arr=(2)
dataset='cifar10'
if [ $dataset = 'stl10' ]
then
    batch_size=256
else
    batch_size=512
fi

lenL=${#lambd_arr[@]}
lenM=${#mu_arr[@]}
lenP=${#pdim_arr[@]}
lenA=${#augs_arr[@]}
lenLMP=$((lenL*lenM*lenP))
lenLM=$((lenL*lenM))
aidx=$((SLURM_ARRAY_TASK_ID/lenLMP))
pmlidx=$((SLURM_ARRAY_TASK_ID%lenLMP))
pidx=$((pmlidx/lenLM))
mlidx=$((pmlidx%lenLM))
midx=$((mlidx/lenL))
lidx=$((mlidx%lenL))
lambd=${lambd_arr[$lidx]}
mu=${mu_arr[$midx]}
pdim=${pdim_arr[$pidx]}
augs=${augs_arr[$aidx]}

wandb_group='blake-richards'
wandb_projname='VICReg-resnet50-hparam'

checkpt_dir=$SCRATCH/fastssl/checkpoints_vicreg
train_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/train.beton
val_dpath=$SCRATCH/ffcv/ffcv_datasets/{dataset}/test.beton

model=resnet50proj
# Let's train a SSL (VICReg) model with the above hyperparams
python scripts/train_model.py --config-file configs/cc_VICReg.yaml \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.mu=$mu \
                --training.projector_dim=$pdim  --training.ckpt_dir=$checkpt_dir \
                --training.batch_size=$batch_size \
                --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                --logging.wandb_project=$wandb_projname \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

model=resnet50feat
# Let's precache features, should take ~35 seconds (rtx8000)
python scripts/train_model.py --config-file configs/cc_precache.yaml \
                --eval.train_algorithm='VICReg' \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.mu=$mu \
                --training.projector_dim=$pdim --training.ckpt_dir=$checkpt_dir \
                --training.batch_size=$batch_size \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

# run linear eval on precached features from model: using default seed 42
python scripts/train_model.py --config-file configs/cc_classifier.yaml \
                --eval.train_algorithm='VICReg' \
                --training.model=$model --training.dataset=$dataset \
                --training.lambd=$lambd --training.mu=$mu \
                --training.projector_dim=$pdim --training.ckpt_dir=$checkpt_dir \
                --training.seed=42 \
                --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                --logging.wandb_project=$wandb_projname \
                --training.train_dataset=$train_dpath --training.val_dataset=$val_dpath

# dataset='stl10'

# cp $SLURM_TMPDIR/*.pth $checkpt_dir/resnet50_checkpoints/
