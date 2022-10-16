#!/bin/bash

#This part of the script is for copying and unpacking the ImageNet dataset in the compute nodes.
dataset_home_in_compute_node=$SLURM_TMPDIR/ImageNet
dataset_home_in_scratch=/home/arnab39/projects/rrg-bengioy-ad/data/curated/imagenet
write_dir_in_scratch=/home/arnab39/scratch/ffcv_datasets/imagenet

mkdir -p $dataset_home_in_compute_node/train $dataset_home_in_compute_node/val/ > /dev/null
echo 'Copying Imagenet'
tar -xvf $dataset_home_in_scratch/ILSVRC2012_img_train.tar -C $dataset_home_in_compute_node/train/ > /dev/null
find $dataset_home_in_compute_node/train/ -name *.tar | while read NAME ; do mkdir -p ${NAME%.tar}; tar -xvf $NAME -C ${NAME%.tar}; rm -f $NAME; done > /dev/null
cp $dataset_home_in_scratch/ILSVRC2012_img_val.tar $dataset_home_in_compute_node
cp $dataset_home_in_scratch/ILSVRC2012_devkit_t12.tar.gz $dataset_home_in_compute_node
echo 'Imagenet Copied ..'


#Use this directly if using the ImageNet dataset in the same node
dataset_home=$dataset_home_in_compute_node

export DATA_DIR=$dataset_home
export WRITE_DIR=$dataset_home/ffcv
mkdir $WRITE_DIR
image_size=256

write_dataset () {
    write_path=$WRITE_DIR/${1}_${2}_${3}_${4}.ffcv
    echo "Writing ImageNet ${1} dataset to ${write_path}"
    python write_imagenet.py \
        --cfg.dataset=imagenet \
        --cfg.split=${1} \
        --cfg.data_dir=$IMAGENET_DIR \
        --cfg.write_path=$write_path \
        --cfg.max_resolution=${2} \
        --cfg.write_mode=proportion \
        --cfg.compress_probability=${3} \
        --cfg.jpeg_quality=$4
}

write_dataset train $image_size 0.5 90
write_dataset val $image_size 0.5 90

cp -r $WRITE_DIR/ffcv $write_dir_in_scratch