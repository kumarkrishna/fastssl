from argparse import ArgumentParser
import numpy as np
import os

from fastargs import get_current_config
from pathlib import Path

import torch


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_args_from_config():
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    args = config.get()
    return args

def merge_with_args(config):
    base_config = get_current_config()
    args = base_config.get()
    for key, val in config.items():
        if key in args.training.__dict__.keys():
            setattr(args.training, key, val)
    return args


def gen_ckpt_path(args, train_algorithm='ssl', epoch=100, prefix='exp', suffix='pth'):
    ckpt_dir = os.path.join(
        args.ckpt_dir, 'lambd_{:.6f}_pdim_{}'.format(args.lambd, args.projector_dim))
    # create directory if it doesn't exist
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    # create ckpt file name
    ckpt_path = os.path.join(ckpt_dir, '{}_{}_{}.{}'.format(
        prefix, train_algorithm, epoch, suffix))
    return ckpt_path