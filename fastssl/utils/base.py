from argparse import ArgumentParser
import torch
import numpy as np
import os

from fastargs import get_current_config


def get_args_from_config():
    breakpoint()
    config = get_current_config()
    parser = ArgumentParser(description="Fast CIFAR-10 training")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")

    config_update_dict = dict()
    if "FFCV_DATA_DIR" in os.environ:
        config_update_dict["training.train_dataset"] = os.path.join(
            os.environ["FFCV_DATA_DIR"],
            f"{config['training.dataset']}/{config['training.train_data_fname']}",
        )
        config_update_dict["training.val_dataset"] = os.path.join(
            os.environ["FFCV_DATA_DIR"],
            f"{config['training.dataset']}/{config['training.val_data_fname']}",
        )
    else:
        config_update_dict["training.train_dataset"] = os.path.join(
            f"{config['training.ffcv_datadir']}",
            f"{config['training.dataset']}/{config['training.train_data_fname']}",
        )
        config_update_dict["training.val_dataset"] = os.path.join(
            f"{config['training.ffcv_datadir']}",
            f"{config['training.dataset']}/{config['training.val_data_fname']}",
        )
    if "CHECKPOINT_DIR" in os.environ:
        config_update_dict["training.ckpt_dir"] = os.environ["CHECKPOINT_DIR"]

    config.collect(config_update_dict)
    config.summary()
    args = config.get()
    return args


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def merge_with_args(config):
    base_config = get_current_config()
    args = base_config.get()
    for key, val in config.items():
        if key in args.training.__dict__.keys():
            setattr(args.training, key, val)
    return args
