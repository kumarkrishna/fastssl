#!/usr/bin/env bash

#SBATCH -A berzelius-2023-229
#SBATCH --gpus=1
#SBATCH -t 6:00:00
#SBATCH -C fat
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user mgamba@kth.se
#SBATCH --output /proj/memorization/logs/%A_%a.out
#SBATCH --error /proj/memorization/logs/%A_%a.err
#SBATCH --array=0-19

NAME="ssl_sweep"

# load env
source scripts/setup_env

export SLURM_TMPDIR="/scratch/local/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
if [ ! -d "$SLURM_TMPDIR" ]; then
    mkdir -p "$SLURM_TMPDIR"
fi

WANDB__SERVICE_WAIT=300

algorithms=(
    "barlow_twins"
    "byol"
    "SimCLR"
    "VICReg"
)

datasets=(
    "cifar10"
    "stl10"
)

algorithm=${algorithms[0]}
dataset=${datasets[0]}

## WARNING, THIS SCRIPT ASSUMES THE SAME NUMBER OF HP CONFIGURATIONS IS SET FOR EACH ALGORITHM

# Barlow Twins
bt_lambd=(0.0005 0.001 0.005 0.01)

# BYOL
byol_tau=(0.7 0.8 0.9 0.99)

# VICReg
vicreg_lambd=(5 15 25 35)

# SimCLR
simcrl_temp=(0.01 0.05 0.1 0.5)

#dataset='stl10'
dataset='cifar10'
if [ $dataset = 'stl10' ]
then
    batch_size=256
    jac_batch_size=4
    proj_str="$algorithm""-stl10-"
    ckpt_str="-stl10"
else
    batch_size=512
    jac_batch_size=4
    proj_str="$algorithm""-cifar10-"
    ckpt_str="-cifar10"
fi

num_workers=16

SEEDS=5
CONFIGS=${#bt_lambd[@]}

seed=$((SLURM_ARRAY_TASK_ID/CONFIGS))
config=$((SLURM_ARRAY_TASK_ID%CONFIGS))

wandb_group='smoothness'

width=64
pdim=2048
model=resnet18proj_width${width}
noise=40

wandb_projname="$proj_str"'ssl-sweep'

# dataset locations
trainset="${DATA_DIR}"/$dataset
testset="${DATA_DIR}"/$dataset

if [ "$algorithm" = "barlow_twins" ]; then
    lambd=${bt_lambd[$config]}
    args="--training.lambd=$lambd"
    config_file="configs/cc_barlow_twins.yaml"
    destdir=$checkpt_dir/resnet18/width${width}/2_augs/lambd_"$lambd"000_pdim_"$pdim"_lr_0.001_wd_1e-05/
elif [ "$algorithm" = "byol" ]; then
    tau=${byol_tau[$config]}
    args="--training.momentum_tau=$tau"
    config_file="configs/cc_byol.yaml"
    destdir=$checkpt_dir/resnet18/width${width}/2_augs/lambd_0.007812_pdim_32_lr_0.001_wd_1e-05/
elif [ "$algorithm" = "SimCLR" ]; then
    temperature=${simcrl_temp[$config]}
    args="--training.temperature=$temperature"
    config_file="configs/cc_SimCLR.yaml"
    destdir=$checkpt_dir/resnet18/width$width/2_augs/temp_"$temperature"00_pdim_"$pdim"_bsz_"$batch_size"_lr_0.001_wd_1e-05/
elif [ "$algorithm" = "VICReg" ]; then
    lambd=${vicreg_lambd[$config]}
    mu=25
    args="--training.lambd=$lambd --training.mu=$mu"
    config_file="configs/cc_VICReg.yaml"
    destdir=$checkpt_dir/resnet18/width${width}/2_augs/lambd_"$lambd".000_mu_"$mu".000_pdim_"$pdim"_bsz_"$batch_size"_lr_0.001_wd_1e-05/
else
    echo "Algorithm not supported: $algorithm"
    exit 1
fi

checkpt_dir="${SAVE_DIR}"/"$NAME"_"$algorithm$ckpt_str"

if [ ! -d "$checkpt_dir" ]
then
    mkdir -p "$checkpt_dir"
fi

# Let's train a SSL model with the above hyperparams
python scripts/train_model_widthVary.py --config-file $config_file \
                    --training.projector_dim=$pdim \
                    --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                    --training.batch_size=$batch_size --training.model=$model \
                    --training.seed=$seed \
                    --training.epochs=200 \
                    --training.lr=1e-3 \
                    --training.train_dataset=${trainset}_train.beton \
                    --training.val_dataset=${testset}_test.beton \
                    --training.num_workers=$num_workers \
                    --training.log_interval=50 \
                    --training.track_alpha=True \
                    --training.track_jacobian=True \
                    --training.jacobian_batch_size=$jac_batch_size \
                    --training.weight_decay=1e-5 \
                    --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                    --logging.wandb_project=$wandb_projname \
                    $args

status=$?

# let's save the model checkpoints to persistent storage
if [ ! -d $destdir ]; then
    mkdir -p $destdir
fi
cp -v "$SLURM_TMPDIR/exp_ssl_50.pth" "$destdir/exp_ssl_50_seed_"$seed".pt"
cp -v "$SLURM_TMPDIR/exp_ssl_100.pth" "$destdir/exp_ssl_100_seed_"$seed".pt"
cp -v "$SLURM_TMPDIR/exp_ssl_200.pth" "$destdir/exp_ssl_200_seed_"$seed".pt"

new_status=$?
status=$((status|new_status))

echo "Done with training. Running linear evaluation"

model=resnet18feat_width${width}

for epoch in 50 100 200; do

    src_checkpt="$destdir/exp_ssl_"$epoch"_seed_"$seed".pt"

    if [ ! -f "$src_checkpt" ];
    then
        echo "Error: no file not found $src_checkpt"
        exit 1
    else
        echo "Copying SSL features to local storage"
        cp -v "$src_checkpt" "$SLURM_TMPDIR/exp_ssl_100.pth"
    fi
    
    # dataset locations
    trainset="${DATA_DIR}"/$dataset
    testset="${DATA_DIR}"/$dataset

    # running eval for 0 label noise
    # Let's precache features, should take ~35 seconds (rtx8000)
    python scripts/train_model_widthVary.py --config-file configs/cc_precache.yaml \
                        --training.projector_dim=$pdim \
                        --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                        --training.batch_size=$batch_size --training.model=$model \
                        --training.seed=$seed \
                        --training.num_workers=$num_workers \
                        --training.train_dataset=${trainset}_train.beton \
                        --training.val_dataset=${testset}_test.beton \
                        --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                        --logging.wandb_project=$wandb_projname \
                        $args
    status=$?

    # run linear eval on precached features from model: using default seed 42
    python scripts/train_model_widthVary.py --config-file configs/cc_classifier.yaml \
                        --training.projector_dim=$pdim \
                        --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                        --training.batch_size=$batch_size --training.model=$model \
                        --training.seed=$seed \
                        --training.num_workers=$num_workers \
                        --training.train_dataset=${trainset}_train.beton \
                        --training.val_dataset=${testset}_test.beton \
                        --training.log_interval=10 \
                        --training.track_jacobian=True \
                        --training.jacobian_batch_size=32 \
                        --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                        --logging.wandb_project=$wandb_projname \
                        $args
    new_status=$?
    status=$((status|new_status))

    # running eval with label noise
    wandb_projname="$proj_str"'ssl-sweep-noise'$noise
    checkpt_dir="${SAVE_DIR}"/"$NAME""_"$algorithm"_noise"$noise"$ckpt_str"

    if [ ! -d "$checkpt_dir" ]
    then
        mkdir -p "$checkpt_dir"
    fi

    # dataset locations
    trainset="${DATA_DIR}"/$dataset"-Noise_"$noise
    testset="${DATA_DIR}"/$dataset

    # Let's precache features, should take ~35 seconds (rtx8000)
    python scripts/train_model_widthVary.py --config-file configs/cc_precache.yaml \
                        --training.projector_dim=$pdim \
                        --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                        --training.batch_size=$batch_size --training.model=$model \
                        --training.seed=$seed \
                        --training.num_workers=$num_workers \
                        --training.train_dataset=${trainset}/train.beton \
                        --training.val_dataset=${testset}_test.beton \
                        --training.label_noise=$noise \
                        --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                        --logging.wandb_project=$wandb_projname \
                        $args
    status=$?

    # run linear eval on precached features from model: using default seed 42
    python scripts/train_model_widthVary.py --config-file configs/cc_classifier.yaml \
                        --training.projector_dim=$pdim \
                        --training.dataset=$dataset --training.ckpt_dir=$checkpt_dir \
                        --training.batch_size=$batch_size --training.model=$model \
                        --training.seed=$seed \
                        --training.num_workers=$num_workers \
                        --training.train_dataset=${trainset}/train.beton \
                        --training.val_dataset=${testset}_test.beton \
                        --training.log_interval=10 \
                        --training.label_noise=$noise \
                        --training.track_jacobian=True \
                        --training.jacobian_batch_size=32 \
                        --logging.use_wandb=True --logging.wandb_group=$wandb_group \
                        --logging.wandb_project=$wandb_projname \
                        $args

    new_status=$?
    status=$((status|new_status))

    # save precached features to checkpt_dir/feats
    if [ ! -d $checkpt_dir/feats ]
    then
        mkdir $checkpt_dir/feats
    fi

    cp -r $SLURM_TMPDIR/feats/* $checkpt_dir/feats/

done

exit $status

