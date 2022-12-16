#!/usr/bin/env bash

cluster_name=mila
compute_node_data_dir=$SLURM_TMPDIR/ImageNet

if [[ $cluster_name == "beluga" || $cluster_name == "narval" ]]; then
    module load gcc python/3.9 cuda/11.4 opencv/4.5.5
    #module load cuda/11.1 opencv/4.5.1
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
    ffcv_data_dir=~/scratch/ImageNet/ffcv                 #for mila
    echo 'In mila'
fi

if [ ! -d $compute_node_data_dir ]; then
    echo 'Copying Imagenet'
    mkdir $compute_node_data_dir
    cp -r $ffcv_data_dir $compute_node_data_dir
    echo 'Imagenet Copied ..'
fi

lamda=0.005
proj_dim=1024
hidden_dim=$proj_dim
ckpt_dir=~/scratch/bt_checkpoints/ckpt_$lamda-$proj_dim
echo "checkpoint dir: $ckpt_dir"
datadir=$compute_node_data_dir
imagenet_train=$compute_node_data_dir/ffcv/train_256_0.5_90.ffcv
imagenet_val=$compute_node_data_dir/ffcv/val_256_0.5_90.ffcv
dataset=imagenet
batch_size=48
log_interval=1

python -m scripts.train_model_distributed \
            --config-file configs/cc_barlow_twins.yaml \
            --training.lambd=$lamda \
            --training.projector_dim=$proj_dim \
            --training.hidden_dim=$hidden_dim\
            --training.train_dataset=$imagenet_train\
            --training.val_dataset=$imagenet_val\
            --training.dataset=$dataset \
            --training.datadir=$datadir \
            --training.ckpt_dir=$ckpt_dir \
            --training.batch_size=$batch_size \
            --training.num_workers=3\
            --training.log_interval=$log_interval