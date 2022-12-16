#!/usr/bin/env bash

#export LD_PRELOAD=~/Projects/SSL_alpha/fastssl/configs/hack.so 	# Olexa's hack to avoid INTERNAL ASSERT ERROR on Pytorch 1.10
## Do some more exports to avoid getting stuck at PCA
#export MKL_THREADING_LAYER=TBB

cluster_name=mila
compute_node_data_dir=$SLURM_TMPDIR/ImageNet

if [[ $cluster_name == "beluga" || $cluster_name == "narval" ]]; then
    module load gcc python/3.9 cuda/11.4 opencv/4.5.5
    source ~/ffcvenv/bin/activate
    if [ $cluster_name == "narval" ]; then
        local_data_dir=/home/arnab39/projects/rrg-bengioy-ad/data/curated/imagenet                 #for narval
        echo 'In narval'
    elif [ $cluster_name == "beluga" ]; then
        local_data_dir=/home/arnab39/projects/rrg-bengioy-ad/data/curated/imagenet                 #for beluga
        echo 'In beluga'
    fi
elif [ $cluster_name == "mila" ]; then
    module load anaconda/3
    conda activate ffcv
    local_data_dir=/network/datasets/imagenet                #for mila
    echo 'In mila'
fi

if [ ! -d $compute_node_data_dir ]; then
    mkdir -p $compute_node_data_dir/train $compute_node_data_dir/val/ > /dev/null
    echo 'Copying Imagenet'
    tar -xvf $local_data_dir/ILSVRC2012_img_train.tar -C $compute_node_data_dir/train/ > /dev/null
    find $compute_node_data_dir/train/ -name *.tar | while read NAME ; do mkdir -p ${NAME%.tar}; tar -xvf $NAME -C ${NAME%.tar}; rm -f $NAME; done > /dev/null
    cp $local_data_dir/ILSVRC2012_img_val.tar $compute_node_data_dir
    cp $local_data_dir/ILSVRC2012_devkit_t12.tar.gz $compute_node_data_dir
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
batch_size=92
log_interval=1

python -m scripts.train_model_distributed \
            --config-file configs/cc_barlow_twins.yaml \
            --training.lambd=$lamda \
            --training.projector_dim=$proj_dim \
            --training.hidden_dim=$hidden_dim\
            --training.dataset=$dataset \
            --training.datadir=$datadir \
            --training.ckpt_dir=$ckpt_dir \
            --training.batch_size=$batch_size \
            --training.num_workers=3\
            --training.log_interval=$log_interval
