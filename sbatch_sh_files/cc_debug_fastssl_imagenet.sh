#!/usr/bin/env bash

cluster_name=beluga
compute_node_data_dir=$SLURM_TMPDIR/ImageNet

if [[ $cluster_name == "beluga" || $cluster_name == "narval" ]]; then
    module load gcc python/3.9 cuda/11.4 opencv/4.5.5
    source ~/ffcvenv/bin/activate
    if [ $cluster_name == "narval" ]; then
        ffcv_data_dir=~/scratch/imagenet/ffcv                 #for narval
        echo 'In narval'
    elif [ $cluster_name == "beluga" ]; then
        ffcv_data_dir=~/scratch/ffcv_dataset/imagenet/ffcv    #for beluga
        echo 'In beluga'
    fi
elif [ $cluster_name == "mila" ]; then
    module load anaconda/3
    conda activate ffcv
    ffcv_data_dir=~/scratch/Imagenet/ffcv                 #for mila
    echo 'In mila'
fi

if [ ! -d $compute_node_data_dir ]; then
    echo 'Copying Imagenet'
    mkdir $compute_node_data_dir
    cp -r $ffcv_data_dir $compute_node_data_dir
    echo 'Imagenet Copied ..'
fi

lamda=5e-3
proj_dim=4096
checkpt_dir='checkpoints_design_hparams_imagenet'
imagenet_train=$compute_node_data_dir/ffcv/train_256_0.5_90.ffcv
imagenet_val=$compute_node_data_dir/ffcv/val_256_0.5_90.ffcv
dataset=imagenet
batch_size=8

python scripts/train_model.py \
            --config-file configs/cc_barlow_twins.yaml \
            --training.lambd=$lamda \
            --training.projector_dim=$proj_dim \
            --training.dataset=$dataset \
            --training.train_dataset=$imagenet_train \
            --training.val_dataset=$imagenet_val \
            --training.ckpt_dir=$checkpt_dir \
            --training.batch_size=$batch_size \
            --training.num_workers=3 \

mkdir ~/scratch/bt_checkpoints/ckpt_lamda_proj_dim
cp -r $SLURM_TMPDIR/* ~/scratch/bt_checkpoints/ckpt_lamda_proj_dim