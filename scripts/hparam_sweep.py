"""
Perform hparam sweep for fastssl, using Ray Tune.

Usage:

python hparam_sweep.py --model resnet50M --dataset cifar10 --algorithm ssl --num_epochs 100 --batch_size 128 --num_workers 2 --lr 0.1 --momentum 0.9 --weight_decay 0.0001 --warmup_epochs 5 --checkpoint_dir /data/krishna/research/results/0313/003/checkpoints
"""
import argparse
import os
import time

import numpy as np
import ray
from ray import tune


def build_parser():

def build_experiment():

def build_hparam_sweep():
    



if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    ray.init()
    experiment = build_experiment(args)
    tune.run(experiment, **args)
