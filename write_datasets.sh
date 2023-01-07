#!/bin/bash
export IMAGENET_DIR=/network/datasets/imagenet.var/imagenet_torchvision/
export export WRITE_DIR=/network/scratch/l/lindongy/ffcv_datasets/imagenet_256

module load anaconda/3 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/testenv/bin/activate

write_dataset () {
    write_path=$WRITE_DIR/${1}_${2}_${3}_${4}.ffcv
    echo "Writing ImageNet ${1} dataset to ${write_path}"
    python write_datasets.py \
        --cfg.dataset=imagenet \
        --cfg.split=${1} \
        --cfg.data_dir=$IMAGENET_DIR/${1} \
        --cfg.write_path=$write_path \
        --cfg.max_resolution=${2} \
        --cfg.write_mode=proportion \
        --cfg.compress_probability=${3} \
        --cfg.jpeg_quality=$4
}

write_dataset train $1 $2 $3
write_dataset val $1 $2 $3