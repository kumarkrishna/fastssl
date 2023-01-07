#!/bin/bash

write_dataset () {
    write_path=$WRITE_DIR/doubleImage_${1}_${2}_${3}.beton
    echo "Writing ImageNet doubleImage dataset to ${write_path}"
    python write_double_datasets.py \
        --cfg.dataset=imagenet \
        --cfg.data_dir=$IMAGENET_DIR/train \
        --cfg.write_path=$write_path \
        --cfg.max_resolution=${1} \
        --cfg.write_mode=proportion \
        --cfg.compress_probability=${2} \
        --cfg.jpeg_quality=$3
}

write_dataset $1 $2 $3
