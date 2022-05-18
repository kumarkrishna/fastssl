"""
Implment trainer, evaluation pipelines for SSL and linear models.

NOTE: Please update the hparams to best known configuration (ensures good defaults).

Usage:
[local]
python train_model.py --config-file configs/barlow_twins.yaml

[CC cluster]
python train_model.py --config-file configs/cc_barlow_twins.yaml
"""


from argparse import ArgumentParser
from functools import partial
from typing import List

import numpy as np
import os

from pathlib import Path
import pickle
from ray import tune

import time
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD, lr_scheduler

import torchvision
from tqdm import tqdm

from fastargs import Section, Param

from fastssl.data import cifar_ffcv, cifar_classifier_ffcv, cifar_pt, stl10_pt, stl_classifier_ffcv
from fastssl.models import barlow_twins as bt
from fastssl.utils.base import set_seeds, get_args_from_config, merge_with_args
import fastssl.utils.powerlaw as powerlaw

Section('training', 'Fast CIFAR-10 training').params(
    dataset=Param(
        str, 'dataset', default='cifar10'),
    datadir=Param(
        str, 'train data dir', default='/data/krishna/data/cifar'),
    train_dataset=Param(
        str, 'train-dataset', default='/data/krishna/data/ffcv/cifar_train.beton'),
    val_dataset=Param(
        str, 'valid-dataset', default='/data/krishna/data/ffcv/cifar_test.beton'),
    batch_size=Param(
        int, 'batch-size', default=512),
    epochs=Param(
        int, 'epochs', default=100), 
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
        str, 'model to train', default='resnet50proj'),
    num_workers=Param(
        int, 'num of CPU workers', default=4),
    projector_dim=Param(
        int, 'projector dimension', default=128),
    log_interval=Param(
        int, 'log-interval in terms of epochs', default=20),
    ckpt_dir=Param(
        str, 'ckpt-dir', default='/data/krishna/research/results/0319/001/checkpoints'),
    use_autocast=Param(
        bool, 'autocast fp16', default=True),
)

Section('eval', 'Fast CIFAR-10 evaluation').params(
    train_algorithm=Param(
        str, 'pretrain algo', default='ssl'),
    epoch=Param(
        int, 'epoch', default=24)
)


def build_dataloaders(
    dataset='cifar10',
    algorithm='ssl',
    datadir='data/',
    train_dataset=None,
    val_dataset=None,
    batch_size=128,
    num_workers=2):
    if 'cifar' in dataset:
        if algorithm == 'ssl':
            return cifar_pt(
                datadir, batch_size=batch_size, num_workers=num_workers)
            # for ffcv cifar10 dataloader
            # return cifar_ffcv(
            #     train_dataset, val_dataset, batch_size, num_workers)
        elif algorithm == 'linear':
            # dataloader for classifier
            return cifar_classifier_ffcv(
                train_dataset, val_dataset, batch_size, num_workers)
    elif dataset == 'stl10':
        if algorithm == 'ssl':
            return stl10_pt(
                datadir,
                splits=["unlabeled"],
                batch_size=batch_size,
                num_workers=num_workers)
        elif algorithm == 'linear':
            return stl_classifier_ffcv(
                train_dataset, val_dataset, batch_size, num_workers)
                # datadir,
                # splits=["train", "test"],
                # batch_size=batch_size,
                # num_workers=num_workers)


def gen_ckpt_path(args, train_algorithm='ssl', epoch=100, prefix='exp', suffix='pth'):
    if suffix=='pth':
        main_dir = os.environ['SLURM_TMPDIR']
        ckpt_dir = main_dir
    else:
        main_dir = args.ckpt_dir
        ckpt_dir = os.path.join(
            main_dir, 'lambd_{:.6f}_pdim_{}{}_lr_{}_wd_{}'.format(args.lambd, args.projector_dim,'_no_autocast' if not args.use_autocast else '',
                                                                args.lr, args.weight_decay))
    # create directory if it doesn't exist
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    # create ckpt file name
    ckpt_path = os.path.join(ckpt_dir, '{}_{}_{}{}.{}'.format(
        prefix, train_algorithm, epoch, '_seed_{}'.format(args.seed) if 'linear' in train_algorithm else '',suffix))
    return ckpt_path 


def build_model(args=None):
    """
    Returns:
        model : model to train
    """
    training = args.training

    if training.algorithm == 'ssl':
        model_args = {
            'bkey': training.model,
            'dataset': training.dataset,
            'projector_dim': training.projector_dim,
            }
        model_cls = bt.BarlowTwins
    
    elif training.algorithm == 'linear':
        ckpt_path = gen_ckpt_path(
            training,
            train_algorithm=args.eval.train_algorithm,
            epoch=args.eval.epoch)
        model_args = {
            'bkey': training.model, # supports : resnet50feat, resnet50proj
            'ckpt_path': ckpt_path,
            'dataset': training.dataset,
            'feat_dim': 2048,  # args.projector_dim
            'num_classes': 10 if training.dataset in 'cifar10' else 100,
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
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logits = model(x)
            return CrossEntropyLoss(label_smoothing=0.1)(logits, y)
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
        default_lr = 1e-3
        default_weight_decay = 1e-6
        return Adam(model.parameters(), lr=default_lr, weight_decay=default_weight_decay)

def train_step(model, dataloader, optimizer=None, loss_fn=None, scaler=None, epoch=None, epochs=None):
    """
    Generic trainer.

    Args:
        model : 
        dataloader :
        optimizer :
        loss_fn:
    """

    total_loss, total_num, num_batches= 0.0, 0, 0

    ## setup dataloader + tqdm 
    train_bar = tqdm(dataloader, desc='Train')
    
    ## set model in train mode
    model.train()

    # for inp in dataloader:
    for inp in train_bar:
        ## backward
        optimizer.zero_grad()

        ## forward   
        if scaler:
            with autocast():
                loss = loss_fn(model, inp)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss_fn(model, inp)
            loss.backward()
            optimizer.step()
        
        ## update loss
        total_loss += loss.item() 
        num_batches += 1

        import ray
        if ray.tune.is_session_enabled():
            tune.report(epoch=epoch, loss=total_loss/num_batches)
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
            total_correct_1 += torch.sum((preds[:, 0:1] == target[:, None]).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((preds[:, 0:5] == target[:, None]).any(dim=-1).float()).item()

        acc_1 = total_correct_1 / total_samples * 100
        acc_5 = total_correct_5 / total_samples * 100
        test_bar.set_description(
            '{} Epoch: [{}/{}] ACC@1: {:.2f}% ACC@5: {:.2f}%'.format(
                'Test', epoch, epochs,
                acc_1, acc_5)
        )
    return acc_1, acc_5


def debug_plot(activations_eigen,alpha,ypred,R2,R2_100,figname):
    import matplotlib.pyplot as plt 
    plt.loglog(np.arange(1,1+len(activations_eigen)),activations_eigen)
    plt.loglog(np.arange(1,1+len(ypred)),ypred)
    plt.title(r'$\alpha$={:.3f} R2={:.3f}, R_100={:.3f}'.format(alpha,R2,R2_100))
    plt.savefig(figname)
    plt.close('all')


def train(model, loaders, optimizer, loss_fn, args):
    results = {'train_loss': [], 'test_acc_1': [], 'test_acc_5': []}
    EPOCHS = args.epochs
    
    if args.algorithm == 'linear':
        if args.use_autocast:
            with autocast():
                # alpha_arr, R2_arr, R2_100_arr = powerlaw.stringer_get_powerlaw_batch(net=model,layer=model.fc,
                #                                                                     # data_loader=loaders['test'],trange=np.arange(50,200),
                #                                                                     data_loader=loaders['test'],trange=np.arange(5,50),
                #                                                                     use_cuda=True)
                activations = powerlaw.generate_activations_prelayer(net=model,layer=model.fc,data_loader=loaders['test'],use_cuda=True)
                activations_eigen = powerlaw.get_eigenspectrum(activations)
                # alpha,ypred,R2,R2_100 = powerlaw.stringer_get_powerlaw(activations_eigen,trange=np.arange(3,500))
                alpha,ypred,R2,R2_100 = powerlaw.stringer_get_powerlaw(activations_eigen,trange=np.arange(3,100))
                # debug_plot(activations_eigen,alpha,ypred,R2,R2_100,'test_full_early_{:.4f}.png'.format(args.lambd))
                # save_path = gen_ckpt_path(args, args.algorithm, args.epochs, 'results_{}_full_early_alpha'.format(args.dataset), 'npy')
                # np.save(save_path,dict(alpha=alpha,R2=R2,R2_100=R2_100))
                # breakpoint()
                results['eigenspectrum'] = activations_eigen
                results['alpha'] = alpha
                results['R2'] = R2
                results['R2_100'] = R2_100
        else:
            alpha_arr, R2_arr, R2_100_arr = powerlaw.stringer_get_powerlaw_batch(net=model,layer=model.fc,
                                                                                # data_loader=loaders['test'],trange=np.arange(50,200),
                                                                                data_loader=loaders['test'],trange=np.arange(5,50),
                                                                                use_cuda=True)

            results['alpha_arr'] = alpha_arr
            results['R2_arr'] = R2_arr
            results['R2_100_arr'] = R2_100_arr
    if args.use_autocast:
        scaler = GradScaler()
    else:
        scaler = None
    
    for epoch in range(1, EPOCHS+1):
        train_loss = train_step(
            model=model,
            dataloader=loaders['train'],
            optimizer=optimizer,
            scaler=scaler,
            loss_fn=loss_fn, epoch=epoch, epochs=EPOCHS)
        
        results['train_loss'].append(train_loss)

        if args.algorithm == 'linear':
            acc_1, acc_5 = eval_step(model, loaders['test'], epoch=epoch, epochs=EPOCHS)
            results['test_acc_1'].append(acc_1)
            results['test_acc_5'].append(acc_5)
        elif epoch % args.log_interval == 0:
            ckpt_path = gen_ckpt_path(args, epoch=epoch)
            torch.save(
                model.state_dict(),
                ckpt_path)
    return results



def run_experiment(args):
    # import ray
    # num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK'))
    # ray.init(num_cpus=2)
    training = args.training

    set_seeds(training.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## Use FFCV to build dataloaders 
    loaders = build_dataloaders(
        training.dataset,
        training.algorithm,
        training.datadir,
        training.train_dataset,
        training.val_dataset,
        training.batch_size,
        training.num_workers)
    print("CONSTRUCTED DATA LOADERS")

    # build model from SSL library
    model = build_model(args)
    print("CONSTRUCTED MODEL")

    # build optimizer
    optimizer = build_optimizer(model, training)
    print("CONSTRUCTED OPTIMIZER")

    # get loss function
    loss_fn = build_loss_fn(training)
    print("CONSTRUCTED LOSS FUNCTION")

    # train the model with default=BT
    results = train(model, loaders, optimizer, loss_fn, training)

    # save results
    save_path = gen_ckpt_path(training, training.algorithm, training.epochs, 'results_{}_early_alpha'.format(training.dataset), 'npy')
    # with open(save_path, 'wb') as f:
        # pickle.dump(results, f)
    np.save(save_path,results)


def bt_trainer(config):
    """
    Trainer class compatible with the ray api.
    """
    args = merge_with_args(config)
    run_experiment(args)


if __name__ == "__main__":
    # gather arguments 
    args = get_args_from_config()
    args.training.datadir = args.training.datadir.format(dataset=args.training.dataset)
    args.training.train_dataset = args.training.train_dataset.format(dataset=args.training.dataset)
    args.training.val_dataset = args.training.val_dataset.format(dataset=args.training.dataset)

    # train model 
    start_time = time.time()
    run_experiment(args)

    # wrapup experiments with logging key variables
    print(f'Total time: {time.time() - start_time:.5f}')
    print(f'Results saved to {args.training.ckpt_dir}')