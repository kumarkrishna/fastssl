"""
Implment trainer, evaluation pipelines for SSL and linear models.

Usage:
    [local]
    python trainer.py --config-file configs/barlow_twins.yaml

    [CC cluster]
    python trainer.py --config-file configs/cc_barlow_twins.yaml
"""


from argparse import ArgumentParser
from functools import partial
from typing import List

import numpy as np
import os, glob

import time, copy

from fastargs import Section, Param

from fastssl.utils.base import get_args_from_config
from fastssl.utils.experiment import Experiment

Section("train", "Fast CIFAR-10 training").params(
    dataset=Param(str, "dataset", default="cifar10"),
    datadir=Param(str, "train data dir", default="/data/krishna/data/cifar"),
    train_dataset=Param(
        str, "train-dataset", default="/data/krishna/data/ffcv/cifar_train.beton"
    ),
    valid_dataset=Param(
        str, "valid-dataset", default="/data/krishna/data/ffcv/cifar_test.beton"
    ),
    batch_size=Param(int, "batch-size", default=512),
    epochs=Param(int, "epochs", default=100),
    lr=Param(float, "learning-rate", default=1e-3),
    weight_decay=Param(float, "weight_decay", default=1e-6),
    lambd=Param(float, "lambd for BarlowTwins", default=1 / 128),
    label_smoothing_coeff=Param(float, "label smoothing coeff", default=0.1),
    momentum_tau=Param(float, "momentum_tau for BYOL", default=0.01),
    temperature=Param(float, "temperature for SimCLR", default=0.01),
    seed=Param(int, "seed", default=1),
    mode=Param(str, "mode", default="train"),
    algorithm=Param(str, "learning algorithm", default="ssl"),
    model=Param(str, "model to train", default="resnet50proj"),
    model_type=Param(str, "model type", default="BarlowTwins"),
    loss_fn_type=Param(str, "loss function type", default="BarlowTwinsLoss"),
    num_workers=Param(int, "num of CPU workers", default=4),
    optim_type=Param(str, "optimizer", default="adam"),
    projector_dim=Param(int, "projector dimension", default=128),
    hidden_dim=Param(int, "hidden dimension for BYOL projector", default=128),
    log_interval=Param(int, "log-interval in terms of epochs", default=20),
    ckpt_dir=Param(
        str, "ckpt-dir", default="/data/krishna/research/results/0319/001/checkpoints"
    ),
    use_autocast=Param(bool, "autocast fp16", default=True),
    track_alpha=Param(bool, "Track evolution of alpha", default=False),
    precache=Param(bool, "Precache outputs of network", default=False),
    adaptive_ssl=Param(bool, "Use alpha to regularize SSL loss", default=False),
    num_augmentations=Param(int, "Number of augmentations to use per image", default=2),
)

Section("valid", "Fast CIFAR-10 evaluation").params(
    train_algorithm=Param(str, "pretrain algo", default="ssl"),
    algorithm=Param(str, "eval algo", default="linear"),
    epoch=Param(int, "epoch", default=24),
    use_precache=Param(bool, "Use Precached outputs of network", default=False),
    num_augmentations_pretrain=Param(
        int, "Number of augmentations used for pretraining", default=2
    ),
)


def main(args):
    # build an experiment
    experiment = Experiment(args)
    experiment.run(mode="train")


if __name__ == "__main__":
    args = get_args_from_config()
    args.train.datadir = args.train.datadir.format(dataset=args.train.dataset)
    args.train.train_dataset = args.train.train_dataset.format(
        dataset=args.train.dataset
    )
    args.train.valid_dataset = args.train.valid_dataset.format(
        dataset=args.train.dataset
    )
    breakpoint()
    main(args)
