from argparse import ArgumentParser
import torch
import numpy as np

from fastargs import get_current_config

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