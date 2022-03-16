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

from train_model import bt_trainer

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=1, metavar='N',
                        help='samples')
    parser.add_argument('--num_epochs', type=int, default=10, metavar='N',
                        help="num_epochs")
    
    # lambda sweep 
    parser.add_argument('--lambd_start', type=float, default=-5.0,
                        metavar='N', help="lambda start for BT loss")
    parser.add_argument('--lambd_end', type=float, default=0,
                        metavar='N', help="lambda end for BT loss")
    
    # projector sweep
    parser.add_argument('--projector_start', type=float, default=7,
                        metavar='N', help="projector start for BT loss")    
    parser.add_argument('--projector_end', type=float, default=10,
                        metavar='N', help="projector end for BT loss")  

    # experiment details 
    parser.add_argument('--exp', type=str, default="bt",
                        metavar='N', help="specify experiment")
    parser.add_argument('--hkey', type=str, default="lambd",
                        metavar='N', help="specify hyperparameter key") 
    parser.add_argument('--name', type=str, default="bt_lambd_sweep",
                        metavar='N', help="name of experiment") 

    # resource allocation
    parser.add_argument('--gpu_per_trial', type=int, default=1,
                        metavar='N', help="num gpus per trial")
    parser.add_argument('--num_workers', type=int, default=1,
                        metavar='N', help="num cpus per trial")
        
    args = parser.parse_args()
    return args


def build_projector_sweep(args):
    """
    Build hyperparam sweep for projector in loss_fn of BT.
    """
    # setup config for ray
    projector = np.logspace(args.projector_start, args.projector_end, args.num_samples, base=2).astype(int)
    config = {
        "projector_fim": tune.grid_search(projector),
    }
    return config


def build_lambd_sweep(args):
    """
    Build hyperparam sweep for lambd in loss_fn of BT.
    """
    # setup config for ray
    lambd = list(np.logspace(args.lambd_start, args.lambd_end, args.num_samples))
    config = {
        "lambd": tune.grid_search(lambd),
    }
    return config 

def run_hparam_sweep(trainer, config):
    analysis = tune.run(
        trainer,
        config=config,
        name=args.name,
        resources_per_trial={
            "cpu": 2,
            "gpu": args.gpu_per_trial,
        },
        verbose=1,
    )
    return analysis

def gen_config(args):
    """
    Generate config for Ray Tune.
    """
    if args.hkey == "lambd":
        config = build_lambd_sweep(args)
    elif args.hkey == "projector":
        config = build_projector_sweep(args)
    return config

def build_trainer(args):
    """
    Build trainer from arguments.
    """
    if args.exp == "bt":
        trainer = bt_trainer
    
    print("Built Trainer ... ")
    return trainer


if __name__ == "__main__":
    args = build_parser()

    # initialize ray with some options
    ray.init()

    # run the hyperparam sweep
    config = gen_config(args)
    # print(config)

    # generate the trainer for ray to run 
    trainer = build_trainer(args)

    # # run the hyperparam sweep
    analysis = run_hparam_sweep(trainer, config)