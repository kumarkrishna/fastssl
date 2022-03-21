"""
Implment trainer, evaluation pipelines for SSL and linear models.

NOTE: Please update the hparams to best known configuration (ensures good defaults).

Usage:
python train_model.py --model resnet50M --dataset cifar10 --algorithm ssl --lr 0.1 --lambd 0.1 --epochs 10 --batch_size 128 --weight_decay 0.0 --seed 0
"""


from argparse import ArgumentParser
from functools import partial
from typing import List

import numpy as np
import os

from pathlib import Path
from ray import tune

import time
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torchvision
from tqdm import tqdm

from fastargs import get_current_config
from fastargs import Section, Param

from fastssl.data import cifar_ffcv, cifar_classifier_ffcv
from fastssl.models import barlow_twins as bt
# import ResNet50Modified, BarlowTwinLoss


Section('training', 'Fast CIFAR-10 training').params(
    dataset=Param(
        str, 'dataset', default='cifar10'),
    train_dataset=Param(
        # str, 'train-dataset', default='/data/krishna/data/ffcv/cifar_train.beton'),
        str, 'train-dataset', default='../ffcv_datasets/CIFAR/train.beton'),
    val_dataset=Param(
        # str, 'valid-dataset', default='/data/krishna/data/ffcv/cifar_test.beton'),
        str, 'valid-dataset', default='../ffcv_datasets/CIFAR/test.beton'),
    batch_size=Param(
        int, 'batch-size', default=512),
    epochs=Param(
        int, 'epochs', default=24), 
    lr=Param(
        float, 'learning-rate', default=1e-3),
    weight_decay=Param(
        float, 'weight_decay', default=1e-6),
    lambd=Param(
        float, 'lambd', default=1/128),
    seed=Param(
        int, 'seed', default=1),
    algorithm=Param(
        str, 'learning algorithm', default='linear'),
    model=Param(
        str, 'model to train', default='linear'),
    num_workers=Param(
        int, 'num of CPU workers', default=4),
    projector_dim=Param(
        int, 'projector dimension', default=128),
    log_interval=Param(
        int, 'log-interval in terms of epochs', default=20),
    ckpt_dir=Param(
        # str, 'ckpt-dir', default='/data/krishna/research/results/0317/001/checkpoints')
        str, 'ckpt-dir', default='checkpoints')
)


def build_dataloaders(
    algorithm='ssl',
    train_dataset=None,
    val_dataset=None,
    batch_size=128,
    num_workers=2):

    if algorithm == 'ssl':
        return cifar_ffcv(
            train_dataset, val_dataset,
            batch_size, num_workers)
    elif algorithm == 'linear':
        return cifar_classifier_ffcv(
            train_dataset, val_dataset,
            batch_size, num_workers)


def build_model(args=None):
    """
    Returns:
        model : model to train
    """
    if args.model == 'resnet50M':
        model_args = {
            'projector_dim': args.projector_dim,
            'dataset': args.dataset,
            }
        model_cls = bt.ResNet50Modified
    
    elif args.model == 'linear':
        ckpt_path = gen_ckpt_path(args, algorithm='ssl', epoch=100)
        model_args = {
            'ckpt_path': ckpt_path,
            'num_classes': 10,
            'dataset': args.dataset,
            'feat_dim': args.projector_dim
        }
        model_cls = bt.LinearClassifier

    model = model_cls(**model_args)
    model = model.to(memory_format=torch.channels_last).cuda()
    return model


def build_loss_fn(args=None):
    if args.algorithm == 'ssl':
        return partial(bt.BarlowTwinLoss, _lambda=args.lambd)
    elif args.algorithm == 'linear':
        def classifier_xent(model, inp):
            x, y = inp
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True).squeeze(1)
            logits = model(x)
            return CrossEntropyLoss()(logits, y)
        return classifier_xent


def build_optimizer(model, args=None):
    """
    Build optimizer for training model.

    Args:
        model : model parameters to train
        args : dict with all relevant parameters
    Returns:
        optimizer : optimizer for training model
    """
    if args.algorithm == 'ssl':
        return Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.algorithm == 'linear':
        return Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)



def train_step(model, dataloader, optimizer=None, loss_fn=None, epoch=None, epochs=None):
    """
    Generic trainer.

    Args:
        model : 
        dataloader :
        optimizer :
        loss_fn:
    """

    total_loss, total_num, num_batches= 0.0, 0, 0

    ## setiup dataloader + tqdm 
    train_bar = tqdm(dataloader, desc='Train')

    ## set model in train mode
    model.train()
    scaler = GradScaler()   
    
    # for inp in dataloader:
    for inp in train_bar:
        ## backward
        optimizer.zero_grad()
        with autocast():
            loss = loss_fn(model, inp)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ## update loss
        total_loss += loss.item() 
        num_batches += 1

        # tune.report(epoch=epoch, loss=total_loss/num_batches)
        train_bar.set_description(
            'Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / num_batches))
    return total_loss / num_batches

def eval_step(model, dataloader, epoch=None, epochs=None):
    model.eval()
    total_correct_1, total_correct_5, total_samples = 0.0, 0.0, 0
    test_bar = tqdm(dataloader, desc='Test')
    for data, target in test_bar:
        total_samples += data.shape[0]
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        with autocast():
            logits = model(data)
            preds = torch.argsort(logits, dim=1, descending=True)
            total_correct_1 += torch.sum((preds[:, 0:1] == target).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((preds[:, 0:5] == target).any(dim=-1).float()).item()

        test_bar.set_description(
            '{} Epoch: [{}/{}] ACC@1: {:.2f}% ACC@5: {:.2f}%'.format(
                'Test', epoch, epochs,
                total_correct_1 / total_samples * 100,
                total_correct_5 / total_samples * 100)
        )


def train(model, loaders, optimizer, loss_fn, args):
    results = {'train_loss': []}
    EPOCHS = args.epochs
    for epoch in range(1, EPOCHS+1):
        train_loss = train_step(
            model=model,
            dataloader=loaders['train'],
            optimizer=optimizer,
            loss_fn=loss_fn, epoch=epoch, epochs=EPOCHS)
        
        results['train_loss'].append(train_loss)

        if args.algorithm == 'linear':
            eval_step(model, loaders['test'], epoch=epoch, epochs=EPOCHS)
        elif epoch % args.log_interval == 0:
            ckpt_path = gen_ckpt_path(args, epoch)
            torch.save(
                model.state_dict(),
                ckpt_path)

def gen_ckpt_path(args, algorithm='ssl', epoch=100, prefix='exp', suffix='pth'):
    ckpt_dir = os.path.join(
        args.ckpt_dir, 'lambd_{:.6f}_pdim_{}'.format(args.lambd, args.projector_dim))
    # create directory if it doesn't exist
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    # create ckpt file name
    ckpt_path = os.path.join(ckpt_dir, '{}_{}_{}.{}'.format(
        prefix, algorithm, epoch, suffix))
    return ckpt_path


def get_arguments():
    parser = ArgumentParser(description='Fast CIFAR-10 training', exit_on_error=False)
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='DS',
                        help='dataset')
    parser.add_argument('--train-dataset', type=str, 
        default='/data/krishna/data/ffcv/cifar_train.beton', metavar='DS', help='train-dataset')
    parser.add_argument('--val-dataset', type=str, 
        default='/data/krishna/data/ffcv/cifar_test.beton', metavar='DS', help='valid-dataset')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 128)')


    ## optimization arguments
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=1e-6, metavar='WD',
                        help='weight decay (default: 0.0)')
    parser.add_argument('--lambd', type=float, default=0.1, metavar='L',
                        help='lambda for barlow twins loss')    

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    ## model arguments    
    parser.add_argument('--algorithm', type=str, default='ssl', metavar='ALG',
                    help='For Saving the current Model')
    parser.add_argument('--model', type=str, default='resnet50M', metavar='MODEL',
                    help="model")
    parser.add_argument('--num-workers', type=int,  default=4,
                        metavar='N', help='num workers')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status') 
    args = parser.parse_args()
    return args

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.ssed(seed)
 

def run_experiment(args):
    # set_seeds(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## Use FFCV to build dataloaders 
    loaders = build_dataloaders(
        args.algorithm,
        args.train_dataset,
        args.val_dataset,
        args.batch_size,
        args.num_workers)
    print("CONSTRUCTED DATA LOADERS")

    # build model from SSL library
    model = build_model(args)
    print("CONSTRUCTED MODEL")

    # build optimizer
    optimizer = build_optimizer(model, args)
    print("CONSTRUCTED OPTIMIZER")

    # get loss function
    loss_fn = build_loss_fn(args)
    print("CONSTRUCTED LOSS FUNCTION")

    # train the model with default=BT
    train(model, loaders, optimizer, loss_fn, args)


def merge_with_args(config):
    base_config = get_current_config()
    args = base_config.get().training
    for key, val in config.items():
        if key in args.__dict__.keys():
            setattr(args, key, val)
    return args
    

def bt_trainer(config):
    """
    Trainer clas compatible with the ray api.
    """
    args = merge_with_args(config)
    run_experiment(args)


if __name__ == "__main__":
    # gather arguments 
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    args = config.get()
    
    start_time = time.time()
    # train model 
    run_experiment(args.training)

    # wrapup experiments with logging key variables
    print(f'Total time: {time.time() - start_time:.5f}')
    print(f'Models saved to {args.training.ckpt_dir}')