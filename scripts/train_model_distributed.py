"""
Implment trainer, evaluation pipelines for SSL and linear models.

NOTE: Please update the hparams to best known configuration (ensures good defaults).

Usage:
[local]
python train_model.py --config-file configs/barlow_twins.yaml

[CC cluster]
python train_model.py --config-file configs/cc_barlow_twins.yaml
"""


from functools import partial
import numpy as np
import os
import signal
import subprocess
from argparse import Namespace
import time
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss

from tqdm import tqdm
import sys
import math
from torch import nn, optim

from fastargs import Section, Param

from fastssl.data import get_ssltrain_imagenet_pytorch_dataloaders_distributed, get_sseval_imagenet_ffcv_dataloaders
import fastssl.models.barlow_twins as bt
from fastssl.utils.base import set_seeds, get_args_from_config
import fastssl.utils.powerlaw as powerlaw
from fastssl.models.barlow_twins import off_diagonal

Section('training', 'Fast distributed imagenet training').params(
    dataset=Param(
        str, 'dataset', default='imagenet'),
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
    learning_rate_weights=Param(
        float, 'learning-rate weights', default=0.2),
    learning_rate_biases=Param(
        float, 'learning-rate biases', default=0.0048),
    weight_decay=Param(
        float, 'weight_decay', default=1e-6),
    lambd=Param(
        float, 'lambd', default=5e-3),
    seed=Param(
        int, 'seed', default=1),
    algorithm=Param(
        str, 'learning algorithm', default='ssl'),
    model=Param(
        str, 'model to train', default='resnet50proj'),
    num_workers=Param(
        int, 'num of CPU workers', default=3),
    projector_dim=Param(
        int, 'projector dimension', default=4096),
    hidden_dim=Param(
        int, 'hidden dimension', default=4096),
    log_interval=Param(
        int, 'log-interval in terms of epochs', default=1),
    ckpt_dir=Param(
        str, 'ckpt-dir', default='/data/krishna/research/results/0319/001/checkpoints'),
    use_autocast=Param(
        bool, 'autocast fp16', default=True),
)

Section('eval', 'Fast Imagenet evaluation').params(
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
        num_workers=2,
        world_size=None
):
        if algorithm == 'ssl':
            return get_ssltrain_imagenet_pytorch_dataloaders_distributed(
                datadir, batch_size, num_workers, world_size
            )
        elif algorithm == 'linear':
            return get_sseval_imagenet_ffcv_dataloaders(
                train_dataset, val_dataset, batch_size, num_workers
            )



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
            'hidden_dim': training.hidden_dim
        }
        model_cls = bt.BarlowTwins

    elif training.algorithm == 'linear':
        if training.dataset == 'cifar10':
            num_classes = 10
        elif training.dataset == 'stl10':
            num_classes = 100
        elif training.dataset == 'imagenet':
            num_classes = 1000
        model_args = {
            'bkey': training.model,  # supports : resnet50feat, resnet50proj
            'ckpt_path': args.ckpt_dir + f'/checkpoint_ssl.pth',
            'dataset': training.dataset,
            'feat_dim': 2048,  # args.projector_dim
            'num_classes': num_classes,
        }
        model_cls = bt.LinearClassifier

    print(model_args, model_cls)
    model = model_cls(**model_args)
    model = model.to(memory_format=torch.channels_last)
    return model


def build_loss_fn(args=None):
    if args.algorithm == 'ssl':
        return partial(BarlowTwinLoss, _lambda=args.lambd)
    elif args.algorithm == 'linear':
        def classifier_xent(model, inp):
            x, y = inp
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logits = model(x)
            return CrossEntropyLoss(label_smoothing=0.1)(logits, y)
        return classifier_xent


def train_step(model, dataloader, optimizer=None, loss_fn=None, scaler=None, epoch=None, epochs=None, gpu=None):
    """
    Generic trainer.

    Args:
        model :
        dataloader :
        optimizer :
        loss_fn:
    """

    total_loss, total_num, num_batches = 0.0, 0, 0

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
                loss = loss_fn(model, inp, gpu)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss_fn(model, inp, gpu)
            loss.backward()
            optimizer.step()

        ## update loss
        total_loss += loss.item()
        num_batches += 1
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


def train(model, loaders, optimizer, loss_fn, args, gpu, sampler, start_epoch):
    results = {'train_loss': [], 'test_acc_1': [], 'test_acc_5': []}

    if args.algorithm == 'linear':
        if args.use_autocast:
            with autocast():
                activations = powerlaw.generate_activations_prelayer(net=model, layer=model.fc,
                                                                     data_loader=loaders['test'], use_cuda=True)
                activations_eigen = powerlaw.get_eigenspectrum(activations)
                alpha, ypred, R2, R2_100 = powerlaw.stringer_get_powerlaw(activations_eigen, trange=np.arange(3, 100))
                results['eigenspectrum'] = activations_eigen
                results['alpha'] = alpha
                results['R2'] = R2
                results['R2_100'] = R2_100
        else:
            alpha_arr, R2_arr, R2_100_arr = powerlaw.stringer_get_powerlaw_batch(net=model, layer=model.fc,
                                                                                 data_loader=loaders['test'],
                                                                                 trange=np.arange(5, 50),
                                                                                 use_cuda=True)

            results['alpha_arr'] = alpha_arr
            results['R2_arr'] = R2_arr
            results['R2_100_arr'] = R2_100_arr

    if args.use_autocast:
        scaler = GradScaler()
    else:
        scaler = None

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        train_loss = train_step(
            model=model,
            dataloader=loaders['train'],
            optimizer=optimizer,
            scaler=scaler,
            loss_fn=loss_fn,
            epoch=epoch,
            epochs=args.epochs,
            gpu=gpu)

        results['train_loss'].append(train_loss)

        if args.algorithm == 'linear':
            acc_1, acc_5 = eval_step(model, loaders['test'], epoch=epoch, epochs=args.epochs)
            results['test_acc_1'].append(acc_1)
            results['test_acc_5'].append(acc_5)

        if epoch % args.log_interval == 0:
            if args.rank == 0:
                # save checkpoint
                print('Checkpoint saved ..')
                state = dict(epoch=epoch + 1, model=model.state_dict(),
                             optimizer=optimizer.state_dict())
                torch.save(state, args.ckpt_dir + f'/checkpoint_{args.algorithm}.pth')

    return results


def main_worker(gpu, train_args, eval_args):
    args = Namespace()
    args.training = train_args
    args.eval = eval_args
    args.training.rank += gpu

    torch.distributed.init_process_group(
        backend='nccl', init_method=args.training.dist_url,
        world_size=args.training.world_size, rank=args.training.rank
    )

    training = args.training

    if training.rank == 0:
        os.makedirs(training.ckpt_dir, exist_ok=True)
        stats_file = open(training.ckpt_dir + '/stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    set_seeds(training.seed)
    # build model from SSL library

    model = build_model(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(parameters, lr=0, weight_decay=1e-6,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)
    print("CONSTRUCTED MODEL AND OPTIMIZER")

    # automatically resume from checkpoint if it exists
    if os.path.isfile(training.ckpt_dir + f'/checkpoint_{training.algorithm}.pth'):
        ckpt = torch.load(training.ckpt_dir + f'/checkpoint_{training.algorithm}.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print('Training checkpoint loaded ..')
    else:
        start_epoch = 0

    ## Use FFCV to build dataloaders
    loaders, sampler = build_dataloaders(
        training.dataset,
        training.algorithm,
        training.datadir,
        training.train_dataset,
        training.val_dataset,
        training.batch_size,
        training.num_workers,
        training.world_size
    )
    print("CONSTRUCTED DATA LOADERS")

    # get loss function
    loss_fn = build_loss_fn(training)
    print("CONSTRUCTED LOSS FUNCTION")

    # train the model with default=BT
    results = train(model, loaders, optimizer, loss_fn, training, gpu, sampler, start_epoch)



def BarlowTwinLoss(model, inp, gpu, _lambda=None):

    # generate samples from tuple
    (x1, x2), _ = inp
    x1, x2 = x1.cuda(gpu, non_blocking=True), x2.cuda(gpu, non_blocking=True)
    bsz = x1.shape[0]

    # forward pass
    z1 = model(x1)
    z2 = model(x2)

    z1_norm = (z1 - z1.mean(0)) / z1.std(0)  # NxD
    z2_norm = (z2 - z2.mean(0)) / z2.std(0)  # NxD

    c = torch.mm(z1_norm.T, z2_norm) / bsz  # DxD

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()

    loss = on_diag + _lambda * off_diag
    return loss

class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0., torch.where(update_norm > 0,
                                 (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == "__main__":
    # gather arguments
    start_time = time.time()
    args = get_args_from_config()
    args.training.datadir = args.training.datadir.format(dataset=args.training.dataset)
    args.training.train_dataset = args.training.train_dataset.format(dataset=args.training.dataset)
    args.training.val_dataset = args.training.val_dataset.format(dataset=args.training.dataset)
    args.training.ngpus_per_node = torch.cuda.device_count()
    print(f"GPUs per node: {args.training.ngpus_per_node}")
    #if 'SLURM_JOB_ID' in os.environ:
    if args.training.ngpus_per_node > 1:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.training.rank = int(os.getenv('SLURM_NODEID')) * args.training.ngpus_per_node
        args.training.world_size = int(os.getenv('SLURM_NNODES')) * args.training.ngpus_per_node
        args.training.dist_url = f'tcp://{host_name}:58472'
    else:
        # single-node distributed training
        args.training.rank = 0
        args.training.dist_url = 'tcp://localhost:58472'
        args.training.world_size = args.training.ngpus_per_node

    train_args = Namespace(**args.__dict__['training'].__dict__)
    eval_args = Namespace(**args.__dict__['eval'].__dict__)
    torch.multiprocessing.spawn(main_worker, args=(train_args, eval_args), nprocs=args.training.ngpus_per_node)

    # wrapup experiments with logging key variables
    print(f'Total time: {time.time() - start_time:.5f}')
    print(f'Results saved to {args.training.ckpt_dir}')