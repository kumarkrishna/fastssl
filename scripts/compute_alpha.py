"""
Given a model, compute the statistic for activations for list of layers.

Usage:

python compute_alpha.py --model_path=<model_path> --layer_names=<layer_names> --output_path=<output_path>
"""
from argparse import ArgumentParser

import os
import torch
from torch.cuda.amp import autocast

from fastargs import get_current_config
from fastargs import Section, Param

from fastssl.data import cifar_classifier_ffcv
from fastssl.models import barlow_twins as bt

import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm

from fastssl.utils.base import get_args_from_config, gen_ckpt_path
from fastssl.utils.powerlaw import PowerLaw


Section('training', 'Fast CIFAR-10 training').params(
    dataset=Param(
        str, 'dataset', default='cifar10'),
    train_dataset=Param(
        str, 'train-dataset', default='/data/krishna/data/ffcv/cifar_train.beton'),
    val_dataset=Param(
        str, 'valid-dataset', default='/data/krishna/data/ffcv/cifar_test.beton'),
    batch_size=Param(
        int, 'batch-size', default=128),
    epoch=Param(
        int, 'epoch', default=100), 
    lr=Param(
        float, 'learning-rate', default=1e-3),
    weight_decay=Param(
        float, 'weight_decay', default=1e-6),
    lambd=Param(
        float, 'lambd', default=1/128),
    seed=Param(
        int, 'seed', default=1),
    algorithm=Param(
        str, 'learning algorithm', default='ssl'),
    model=Param(
        str, 'model to train', default='resnet50M'),
    num_workers=Param(
        int, 'num of CPU workers', default=4),
    projector_dim=Param(
        int, 'projector dimension', default=128),
    log_interval=Param(
        int, 'log-interval in terms of epochs', default=10),
    ckpt_dir=Param(
        str, 'ckpt-dir', default='/data/krishna/research/results/0319/001/checkpoints')
)

Section('eval', 'Fast CIFAR-10 evaluation').params(
    train_algorithm=Param(
        str, 'pretrain algo', default='ssl'),
    epoch=Param(
        int, 'epoch', default=24)
)


def get_eigenspectrum(activations, max_eigvals=2048):
    """
    Given a (n x n) covariance matrix, compute the 
    eigenspectrum and the coefficient of decay by 
    fitting to the log-spectrum.
    """
    act_shape = activations.shape
    feats = activations.reshape(act_shape[0], -1)

    # batchsize x featdim
    bsz, feat_dim = feats.shape

    centered_ft = feats - feats.mean(axis=0)
    pca = PCA(n_components=min(max_eigvals, bsz, feat_dim), svd_solver='full')
    pca.fit(centered_ft)
    eigenspectrum = pca.explained_variance_ratio_
    return eigenspectrum


def get_ckpt_path(args, prefix='exp', suffix='pth'):
    ckpt_dir = os.path.join(
        args.ckpt_dir, 'lambd_{:.6f}_pdim_{}'.format(args.lambd, args.projector_dim))
    ckpt_path = os.path.join(ckpt_dir, '{}_{}_{}.{}'.format(
        prefix, args.algorithm, args.epoch, suffix))
    return ckpt_path


def build_model(args):
    training = args.training

    ckpt_path = gen_ckpt_path(
        training, 
        train_algorithm=args.eval.train_algorithm,
        epoch=args.eval.epoch)

    model_args = {
        'bkey' : training.model,
        'ckpt_path' : ckpt_path,
        'dataset' : training.dataset,
        'feat_dim' : 2048,
    }

    model_cls = bt.Featurize

    model = model_cls(**model_args)
    model = model.to(memory_format=torch.channels_last).cuda()
    return model
        

def save_data(data, args, prefix='feats'):
    filename = gen_ckpt_path(args, prefix=prefix, suffix='npy')
    print('Saving {} to ... {}'.format(prefix, filename))
    with open(filename, 'wb') as f:
        np.save(f, data)


def merge_with_args(config):
    base_config = get_current_config()
    args = base_config.get().training
    for key, val in config.items():
        if key in args.__dict__.keys():
            setattr(args, key, val)
    return args
    

def alpha_trainer(config):
    """
    Trainer clas compatible with the ray api.
    """
    args = merge_with_args(config)
    store_eigenspectrum(args)


def run_experiment(args):
    training = args.training

    model = build_model(args)
    loaders = cifar_classifier_ffcv(
        training.train_dataset, training.val_dataset,
        training.batch_size, training.num_workers)

    trainbar = tqdm(loaders['test'])
    img_feats = []

    for inp in trainbar:
        # import pdb; pdb.set_trace()
        img, _ = inp
        # img = img.cuda(non_blocking=True)
        # with autocast():
        feats = model(img).cpu().detach().numpy()
            # feats = model.projs(img).cpu().detach().numpy()
        img_feats.append(feats)
    
    img_feats = np.vstack(img_feats)
    print("Extracted feats {}".format(img_feats.shape))

    eigenspec = get_eigenspectrum(np.array(img_feats))
    print("Computed eigenspectrum as {}".format(eigenspec.shape))

    plaw = PowerLaw(variant='stringer')
    alpha, _ = plaw.fit(eigenspec, (10, 100))
    print("Computed alpha as {}".format(alpha))

    # saves the eigenspectrum and features
    save_data(img_feats, args.training, prefix='feats')
    save_data(eigenspec, args.training, prefix='eigenspec_feats')


if __name__ == "__main__":
    args = get_args_from_config()
    run_experiment(args)