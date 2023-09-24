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
import os, glob

from pathlib import Path
import pickle

# from ray import tune

import time, copy
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD, lr_scheduler

import torchvision
from tqdm import tqdm

from fastargs import Section, Param

from fastssl.data import (
    cifar_ffcv,
    cifar_classifier_ffcv,
    cifar_pt,
    stl_ffcv,
    stl10_pt,
    stl_classifier_ffcv,
    simple_dataloader,
)
from fastssl.models import barlow_twins as bt
from fastssl.models import linear, byol, simclr, vicreg

from fastssl.utils.base import (
    set_seeds, 
    get_args_from_config, 
    merge_with_args,
    start_wandb_server,
    stop_wandb_server,
    log_wandb
)
import fastssl.utils.powerlaw as powerlaw

Section("training", "Fast CIFAR-10 training").params(
    dataset=Param(str, "dataset", default="cifar10"),
    datadir=Param(str, "train data dir", default="/data/krishna/data/cifar"),
    train_dataset=Param(
        str, "train-dataset", default="/data/krishna/data/ffcv/cifar_train.beton"
    ),
    val_dataset=Param(
        str, "valid-dataset", default="/data/krishna/data/ffcv/cifar_test.beton"
    ),
    batch_size=Param(int, "batch-size", default=512),
    epochs=Param(int, "epochs", default=100),
    lr=Param(float, "learning-rate", default=1e-3),
    weight_decay=Param(float, "weight_decay", default=1e-6),
    lambd=Param(float, "lambd for BarlowTwins/VICReg", default=1 / 128),
    mu=Param(float, "mu for VICReg", default=25.0),
    momentum_tau=Param(float, "momentum_tau for BYOL", default=0.01),
    temperature=Param(float, "temperature for SimCLR", default=0.01),
    seed=Param(int, "seed", default=1),
    algorithm=Param(str, "learning algorithm", default="ssl"),
    model=Param(str, "model to train", default="resnet50proj"),
    num_workers=Param(int, "num of CPU workers", default=4),
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

Section("eval", "Fast CIFAR-10 evaluation").params(
    train_algorithm=Param(str, "pretrain algo", default="ssl"),
    epoch=Param(int, "epoch", default=24),
    use_precache=Param(bool, "Use Precached outputs of network", default=False),
    num_augmentations_pretrain=Param(
        int, "Number of augmentations used for pretraining", default=2
    ),
)

Section("logging", "Fast CIFAR-10 logging options").params(
    use_wandb=Param(bool, "Use wandb to log results", default=False),
    wandb_group=Param(str, "Wandb team to log run", default="eigengroup"),
    wandb_project=Param(str, "Wandb project to log run", default="temp-proj"),
)

def build_dataloaders(
    dataset="cifar10",
    algorithm="ssl",
    datadir="data/",
    train_dataset=None,
    val_dataset=None,
    batch_size=128,
    num_workers=2,
    num_augmentations=2,
):
    if os.path.splitext(train_dataset)[-1] == ".npy":
        # using precached features!!
        print("Using simple dataloader")
        return simple_dataloader(
            train_dataset, val_dataset, batch_size=batch_size, num_workers=num_workers
        )
    if "cifar" in dataset:
        if algorithm in ("BarlowTwins", "SimCLR", "ssl", "byol", "VICReg"):
            # return cifar_pt(
            #     datadir, batch_size=batch_size, num_workers=num_workers)
            # for ffcv cifar10 dataloader
            return cifar_ffcv(
                train_dataset,
                val_dataset,
                batch_size,
                num_workers,
                num_augmentations=num_augmentations,
            )
        elif algorithm == "linear":
            default_linear_bsz = 512
            # dataloader for classifier
            return cifar_classifier_ffcv(
                train_dataset,
                val_dataset,
                default_linear_bsz,
                num_workers,
                num_augmentations=num_augmentations,
            )
        else:
            raise Exception("Algorithm not implemented")
    elif dataset == "stl10":
        if algorithm in ("BarlowTwins", "SimCLR", "ssl", "byol", "VICReg"):
            # return stl10_pt(
            #     datadir,
            #     splits=["unlabeled"],
            #     batch_size=batch_size,
            #     num_workers=num_workers)
            # return stl_ffcv(train_dataset, val_dataset, batch_size, num_workers)
            return stl_ffcv(
                train_dataset,
                val_dataset,
                batch_size,
                num_workers,
                num_augmentations=num_augmentations,
            )
        elif algorithm == "linear":
            default_linear_bsz = 256
            # return stl_classifier_ffcv(
            #     train_dataset, val_dataset, default_linear_bsz, num_workers
            # )
            return stl_classifier_ffcv(
                train_dataset,
                val_dataset,
                default_linear_bsz,
                num_workers,
                num_augmentations=num_augmentations,
            )
        else:
            raise Exception("Algorithm not implemented")
    else:
        raise Exception("Dataset {} not supported".format(dataset))


def gen_ckpt_path(args, eval_args, epoch=100, prefix="exp", suffix="pth"):
    if suffix == "pth":
        main_dir = os.environ["SLURM_TMPDIR"]
        ckpt_dir = main_dir
        ckpt_path = os.path.join(
            ckpt_dir,
            "{}_{}_{}{}.{}".format(
                prefix,
                eval_args.train_algorithm
                if "linear" in args.algorithm
                else args.algorithm,
                epoch,
                "_seed_{}".format(args.seed)
                if "linear" in eval_args.train_algorithm
                else "",
                suffix,
            ),
        )
    else:
        if "precache" in prefix:
            # save precache features/embeddings in $SLURM_TMPDIR
            main_dir = os.environ["SLURM_TMPDIR"]
        else:
            main_dir = args.ckpt_dir
        model_name = args.model
        model_name = model_name.replace("proj", "")
        model_name = model_name.replace("feat", "")
        main_dir = os.path.join(main_dir, model_name)
        # dir for augs during SSL pretraining
        if args.algorithm == "linear":
            dir_algorithm = eval_args.train_algorithm
            main_dir = os.path.join(
                main_dir, "{}_augs".format(eval_args.num_augmentations_pretrain)
            )
        else:
            dir_algorithm = args.algorithm
            main_dir = os.path.join(main_dir, "{}_augs".format(args.num_augmentations))

        # dir for SSL hparams
        if dir_algorithm in ["ssl", "BarlowTwins"]:
            ckpt_dir = os.path.join(
                main_dir,
                "lambd_{:.6f}_pdim_{}{}_lr_{}_wd_{}".format(
                    args.lambd,
                    args.projector_dim,
                    "_no_autocast" if not args.use_autocast else "",
                    args.lr,
                    args.weight_decay,
                ),
            )
        elif dir_algorithm in ["SimCLR"]:
            ckpt_dir = os.path.join(
                main_dir,
                "temp_{:.3f}_pdim_{}{}_bsz_{}_lr_{}_wd_{}".format(
                    args.temperature,
                    args.projector_dim,
                    "_no_autocast" if not args.use_autocast else "",
                    args.batch_size,
                    args.lr,
                    args.weight_decay,
                ),
            )
        elif dir_algorithm in ["VICReg"]:
            ckpt_dir = os.path.join(
                main_dir,
                "lambd_{:.3f}_mu_{:.3f}_pdim_{}{}_bsz_{}_lr_{}_wd_{}".format(
                    args.lambd,
                    args.mu,
                    args.projector_dim,
                    "_no_autocast" if not args.use_autocast else "",
                    args.batch_size,
                    args.lr,
                    args.weight_decay,
                ),
            )

        # dir for augs during linear eval
        if args.algorithm == "linear":
            ckpt_dir = os.path.join(
                ckpt_dir, "{}_augs_eval".format(args.num_augmentations)
            )
        # create ckpt file name
        ckpt_path = os.path.join(
            ckpt_dir,
            "{}{}{}.{}".format(
                prefix,
                "" if "precache" in prefix else "_{}_{}".format(args.algorithm, epoch),
                "_seed_{}".format(args.seed),
                suffix,
            ),
        )
    # create directory if it doesn't exist
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    return ckpt_path


def build_model(args=None):
    """
    Returns:
        model : model to train
    """
    training = args.training
    eval = args.eval

    if training.algorithm in ("BarlowTwins", "SimCLR", "ssl", "byol", "VICReg"):
        model_args = {
            "bkey": training.model,
            "dataset": training.dataset,
            "projector_dim": training.projector_dim,
        }

        if training.algorithm in ("byol"):
            model_args["hidden_dim"] = training.hidden_dim
            model_cls = byol.BYOL
        elif training.algorithm in ("SimCLR"):
            # setting projector dim and hidden dim the same for SimCLR projector
            model_args["hidden_dim"] = training.projector_dim
            model_cls = simclr.SimCLR
        elif training.algorithm in ("VICReg"):
            # setting projector dim and hidden dim the same for VICReg projector
            model_args["hidden_dim"] = training.projector_dim
            model_cls = vicreg.VICReg
        else:
            model_args["hidden_dim"] = training.projector_dim
            model_cls = bt.BarlowTwins

    elif training.algorithm == "linear":
        ckpt_path = gen_ckpt_path(training, eval, epoch=args.eval.epoch)
        if eval.use_precache:
            model_type = ""
        else:
            model_type = training.model  # supports : resnet<18/50><feat/proj>
        if "proj" in training.model:
            feat_dim = training.projector_dim
        else:
            if "resnet18" in training.model:
                feat_dim = 512
            elif "resnet50" in training.model:
                feat_dim = 2048
            else:
                feat_dim = 2048
        model_args = {
            "bkey": model_type,
            "ckpt_path": ckpt_path,
            "dataset": training.dataset,
            # "feat_dim": training.projector_dim if "proj" in training.model else 2048,
            "feat_dim": feat_dim,
            "proj_hidden_dim": training.hidden_dim
            if eval.train_algorithm in ("byol")
            else training.projector_dim,
            "num_classes": 10 if training.dataset in ["cifar10", "stl10"] else 100,
        }
        model_cls = linear.LinearClassifier

    model = model_cls(**model_args)
    model = model.to(memory_format=torch.channels_last).cuda()
    return model


def build_loss_fn(args=None):
    if args.algorithm in ("BarlowTwins", "ssl"):
        return partial(bt.BarlowTwinLoss, _lambda=args.lambd)
    elif args.algorithm == "byol":
        return byol.BYOLLoss
    elif args.algorithm == "SimCLR":
        return partial(simclr.SimCLRLoss, _temperature=args.temperature)
    elif args.algorithm == "VICReg":
        return partial(vicreg.VICRegLoss, _lambda=args.lambd, _mu=args.mu)
    elif args.algorithm == "linear":

        def classifier_xent(model, inp):
            inp = list(inp)
            # WARNING: every epoch could have different augmentations of images
            y = inp.pop(1)
            num_augs = len(inp)
            # x, y = inp
            for x in inp:
                x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            # x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logits = model(x)
            return CrossEntropyLoss(label_smoothing=0.1)(logits, y)

        return classifier_xent
    else:
        raise Exception("Algorithm {} not implemented".format(args.algorithm))


def build_optimizer(model, args=None):
    """
    Build optimizer for training model.

    Args:
        model : model parameters to train
        args : dict with all relevant parameters
    Returns:
        optimizer : optimizer for training model
    """
    if args.algorithm in ("BarlowTwins", "SimCLR", "ssl", "byol", "VICReg"):
        return Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.algorithm == "linear":
        default_lr = 1e-3
        default_weight_decay = 1e-6
        return Adam(
            model.parameters(), lr=default_lr, weight_decay=default_weight_decay
        )
    else:
        raise Exception("Algorithm not implemented")


# def save_images(img1,img2,name):
#     import matplotlib.pyplot as plt
#     plt.close('all')
#     plt.imsave('test_imgs/{}1.png'.format(name),img1/255.)
#     plt.close('all')
#     plt.imsave('test_imgs/{}2.png'.format(name),img2/255.)
#     plt.close('all')


def train_step(
    model,
    dataloader,
    args,
    target_model=None,
    optimizer=None,
    loss_fn=None,
    scaler=None,
    epoch=None,
):
    """
    Generic trainer.

    Args:
        model :
        target_model: Not None if BYOL
        dataloader :
        optimizer :
        loss_fn:
    """

    total_loss, total_num, num_batches = 0.0, 0, 0

    ## setup dataloader + tqdm
    train_bar = tqdm(dataloader, desc="Train")

    ## set model in train mode
    model.train()
    if args.algorithm == "linear":
        # setting backbone to be eval mode for linear evaluation
        model.backbone.eval()

    # for inp in dataloader:
    for inp in train_bar:
        # if num_batches==0:
        #     save_images(img1=inp[0][0].detach().cpu().numpy().transpose([1,2,0]),img2=inp[1][0].detach().cpu().numpy().transpose([1,2,0]),name='epoch_{}_img_'.format(epoch))
        # breakpoint()
        if type(inp[0]) == type(inp[1]) and inp[0].shape == inp[1].shape:
            # inp is a tuple with the two augmentations.
            # This is legacy implementation of ffcv for dual augmentations
            inp = ((inp[0], inp[1]), None)
        ## backward
        optimizer.zero_grad()

        ## forward
        if scaler:
            with autocast():
                if args.algorithm == "byol":
                    loss = loss_fn(model, target_model, inp)
                elif args.algorithm in ("BarlowTwins", "SimCLR", "ssl", "linear", "VICReg"):
                    loss = loss_fn(model, inp)
                else:
                    raise Exception("Algorithm not implemented")
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

        if args.algorithm == "byol":
            byol.update_state_dict(target_model, model.state_dict(), args.momentum_tau)

        # import ray
        # if ray.tune.is_session_enabled():
        #     tune.report(epoch=epoch, loss=total_loss/num_batches)
        train_bar.set_description(
            "Train Epoch: [{}/{}] Loss: {:.4f}".format(
                epoch, args.epochs, total_loss / num_batches
            )
        )
    return total_loss / num_batches


def eval_step(model, dataloader, epoch=None, epochs=None):
    model.eval()
    total_correct_1, total_correct_5, total_samples = 0.0, 0.0, 0
    test_bar = tqdm(dataloader, desc="Test")
    for inp in test_bar:
        # for data, target in test_bar:
        inp = list(inp)
        # WARNING: every epoch could have different augmentations of images
        target = inp.pop(1)
        for x in inp:
            x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        total_samples += inp[0].shape[0]
        # total_samples += data.shape[0]
        # data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        with autocast():
            logits = model(inp)
            preds = torch.argsort(logits, dim=1, descending=True)
            total_correct_1 += torch.sum(
                (preds[:, 0:1] == target[:, None]).any(dim=-1).float()
            ).item()
            total_correct_5 += torch.sum(
                (preds[:, 0:5] == target[:, None]).any(dim=-1).float()
            ).item()

        acc_1 = total_correct_1 / total_samples * 100
        acc_5 = total_correct_5 / total_samples * 100
        test_bar.set_description(
            "{} Epoch: [{}/{}] ACC@1: {:.2f}% ACC@5: {:.2f}%".format(
                "Test", epoch, epochs, acc_1, acc_5
            )
        )
    return acc_1, acc_5


def debug_plot(activations_eigen, alpha, ypred, R2, R2_100, figname):
    import matplotlib.pyplot as plt

    plt.loglog(np.arange(1, 1 + len(activations_eigen)), activations_eigen)
    plt.loglog(np.arange(1, 1 + len(ypred)), ypred)
    plt.title(r"$\alpha$={:.3f} R2={:.3f}, R_100={:.3f}".format(alpha, R2, R2_100))
    plt.savefig(figname)
    plt.close("all")


def precache_outputs(model, loaders, args, eval_args):
    model.eval()
    trainset_outputs = []
    trainset_labels = []
    train_bar = tqdm(loaders["train"], desc="Train set")
    for inp in train_bar:
        # for data, target in train_bar:
        inp = list(inp)
        # WARNING: every epoch could have different augmentations of images
        target = inp.pop(1)
        for x in inp:
            x = x.cuda(non_blocking=True)
        # data = data.cuda(non_blocking=True)
        with autocast():
            with torch.no_grad():
                out_augs = [model(x) for x in inp]
                # mean of features across different augmentations of each image
                out = torch.mean(torch.stack(out_augs), dim=0)
                # out = model(data)
        trainset_outputs.append(out.data.cpu().float())
        trainset_labels.append(target.data.cpu())
    trainset_outputs = torch.cat(trainset_outputs)
    trainset_labels = torch.cat(trainset_labels)

    testset_outputs = []
    testset_labels = []
    test_bar = tqdm(loaders["test"], desc="Test set")
    for inp in test_bar:
        # for data, target in test_bar:
        inp = list(inp)
        # WARNING: every epoch could have different augmentations of images
        target = inp.pop(1)
        for x in inp:
            x = x.cuda(non_blocking=True)
        # data = data.cuda(non_blocking=True)
        with autocast():
            with torch.no_grad():
                out_augs = [model(x) for x in inp]
                # mean of features across different augmentations of each image
                out = torch.mean(torch.stack(out_augs), dim=0)
                # out = model(data)
        testset_outputs.append(out.data.cpu().float())
        testset_labels.append(target.data.cpu())
    testset_outputs = torch.cat(testset_outputs)
    testset_labels = torch.cat(testset_labels)
    output_dict = {
        "train": {
            "activations": trainset_outputs.cpu().numpy(),
            "labels": trainset_labels.cpu().numpy(),
        },
        "test": {
            "activations": testset_outputs.cpu().numpy(),
            "labels": testset_labels.cpu().numpy(),
        },
    }
    return output_dict


def train(model, loaders, optimizer, loss_fn, args, eval_args, use_wandb=False):
    if args.track_alpha:
        results = {
            "train_loss": [],
            "test_acc_1": [],
            "test_acc_5": [],
            "eigenspectrum": [],
            "alpha": [],
            "R2": [],
            "R2_100": [],
        }
    else:
        results = {"train_loss": [], "test_acc_1": [], "test_acc_5": []}

    if args.algorithm == "linear":
        if args.use_autocast:
            with autocast():
                activations = powerlaw.generate_activations_prelayer(
                    net=model,
                    layer=model.fc,
                    data_loader=loaders["test"],
                    use_cuda=True,
                )
                activations_eigen = powerlaw.get_eigenspectrum(activations)
                alpha, ypred, R2, R2_100 = powerlaw.stringer_get_powerlaw(
                    activations_eigen, trange=np.arange(3, 100)
                )
                # debug_plot(activations_eigen,alpha,ypred,R2,R2_100,'test_full_early_{:.4f}.png'.format(args.lambd))
                # save_path = gen_ckpt_path(args, args.algorithm, args.epochs, 'results_{}_full_early_alpha'.format(args.dataset), 'npy')
                # np.save(save_path,dict(alpha=alpha,R2=R2,R2_100=R2_100))
                # breakpoint()
                results["eigenspectrum"] = activations_eigen
                results["alpha"] = alpha
                results["R2"] = R2
                results["R2_100"] = R2_100
                print("Initial alpha", results["alpha"])
        else:
            alpha_arr, R2_arr, R2_100_arr = powerlaw.stringer_get_powerlaw_batch(
                net=model,
                layer=model.fc,
                # data_loader=loaders['test'],trange=np.arange(50,200),
                data_loader=loaders["test"],
                trange=np.arange(5, 50),
                use_cuda=True,
            )

            results["alpha_arr"] = alpha_arr
            results["R2_arr"] = R2_arr
            results["R2_100_arr"] = R2_100_arr

        if use_wandb:
            log_wandb(results, step=0, skip_keys=['eigenspectrum'])

    if args.use_autocast:
        scaler = GradScaler()
    else:
        scaler = None

    if args.algorithm == "byol":
        target_model = copy.deepcopy(model)
        for param in list(target_model.parameters()):
            param.requires_grad = False

    for epoch in range(1, args.epochs + 1):
        if epoch == 1 and args.track_alpha:
            # compute alpha before training starts!
            activations = powerlaw.generate_activations_prelayer(
                net=model,
                layer=model.backbone.proj,
                data_loader=loaders["test"],
                use_cuda=True,
            )
            activations_eigen = powerlaw.get_eigenspectrum(activations)
            alpha, ypred, R2, R2_100 = powerlaw.stringer_get_powerlaw(
                activations_eigen, trange=np.arange(3, 100)
            )
            results["eigenspectrum"].append((epoch - 1, activations_eigen))
            results["alpha"].append((epoch - 1, alpha))
            results["R2"].append((epoch - 1, R2))
            results["R2_100"].append((epoch - 1, R2_100))
            print("Initial alpha", results["alpha"])

        train_loss = train_step(
            model=model,
            dataloader=loaders["train"],
            target_model=target_model if args.algorithm == "byol" else None,
            optimizer=optimizer,
            scaler=scaler,
            loss_fn=loss_fn,
            epoch=epoch,
            args=args,
        )

        results["train_loss"].append(train_loss)

        if args.algorithm == "linear":
            acc_1, acc_5 = eval_step(
                model, loaders["test"], epoch=epoch, epochs=args.epochs
            )
            results["test_acc_1"].append(acc_1)
            results["test_acc_5"].append(acc_5)
        elif epoch % args.log_interval == 0:
            ckpt_path = gen_ckpt_path(args, eval_args, epoch=epoch)
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, ckpt_path)
            if args.track_alpha:
                # compute alpha at intermediate training steps
                activations = powerlaw.generate_activations_prelayer(
                    net=model,
                    layer=model.backbone.proj,
                    data_loader=loaders["test"],
                    use_cuda=True,
                )
                activations_eigen = powerlaw.get_eigenspectrum(activations)
                alpha, ypred, R2, R2_100 = powerlaw.stringer_get_powerlaw(
                    activations_eigen, trange=np.arange(3, 100)
                )
                if args.adaptive_ssl:
                    alpha_gt = max(0, alpha - 1.2)  # check if alpha > 1.2
                    alpha_lt = max(0, 0.8 - alpha)  # check if alpha < 0.8
                    # use alpha_gt to increase lambda
                    # use alpha_lt to decrease lambda
                    curr_lambda = loss_fn.keywords["_lambda"]
                    tqdm.write("Current lamda = {:.6f}".format(curr_lambda))
                    # updated_lambda = curr_lambda*np.exp(alpha_gt-alpha_lt)
                    updated_lambda = curr_lambda + 0.001 * (alpha_gt - alpha_lt)
                    tqdm.write(
                        "alpha = {:.3f}, New lamda = {:.6f}".format(
                            alpha, updated_lambda
                        )
                    )
                    loss_fn = partial(bt.BarlowTwinLoss, _lambda=updated_lambda)
                    if "lambda" not in results.keys():
                        results["lambda"] = []
                    results["lambda"].append((epoch, updated_lambda))
                results["eigenspectrum"].append((epoch, activations_eigen))
                results["alpha"].append((epoch, alpha))
                results["R2"].append((epoch, R2))
                results["R2_100"].append((epoch, R2_100))
                # print(results['alpha'])

        if use_wandb:
            log_wandb(results, step=epoch, skip_keys=['eigenspectrum'])

    return results


def search_precache_file(training, eval):
    # find the appropriate precached npy file
    saved_path = gen_ckpt_path(
        training,
        eval,
        eval.epoch,  # currently not used in the name
        "precache_{}_{}".format(training.dataset, training.model),
        "npy",
    )
    folder = os.path.dirname(saved_path)
    candidate_files = glob.glob(os.path.join(folder, "*.npy"))
    candidate_files = [os.path.basename(f) for f in candidate_files]
    candidate_files = [f for f in candidate_files if training.dataset in f]
    candidate_files = [f for f in candidate_files if training.model in f]
    if len(candidate_files) == 0:
        print("No precached file found! Running linear eval without precaching!")
        setattr(eval, "use_precache", False)
    else:
        try:
            extract_seed_val = lambda x: int(x.split(".npy")[0].split("seed_")[-1])
            seed_files = [
                f for f in candidate_files if training.seed == extract_seed_val(f)
            ]
            assert len(seed_files) == 1
            fname = seed_files[0]
            print("Using precache file {}".format(os.path.join(folder, fname)))
        except:
            fname = candidate_files[0]
            print(
                "Could not find the correct seed value ({}), using {}".format(
                    training.seed, os.path.join(folder, fname)
                )
            )
        setattr(training, "train_dataset", os.path.join(folder, fname))
        setattr(training, "val_dataset", os.path.join(folder, fname))


def run_experiment(args):
    # import ray
    # num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK'))
    # ray.init(num_cpus=2)
    training = args.training
    eval = args.eval

    set_seeds(training.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if eval.use_precache:
        search_precache_file(training, eval)
    ## Use FFCV to build dataloaders
    loaders = build_dataloaders(
        training.dataset,
        training.algorithm,
        training.datadir,
        training.train_dataset,
        training.val_dataset,
        training.batch_size,
        training.num_workers,
        training.num_augmentations,
    )
    print("CONSTRUCTED DATA LOADERS")
    # breakpoint()

    # build model from SSL library
    model = build_model(args)
    print("CONSTRUCTED MODEL")

    # build optimizer
    optimizer = build_optimizer(model, training)
    print("CONSTRUCTED OPTIMIZER")

    if training.precache:
        print("Precaching model outputs, no training")
        # removing the final linear readout layer
        model._modules["fc"] = torch.nn.Identity()
        results = precache_outputs(model, loaders, training, eval)
        # now we save the results to npy file!
        save_path = gen_ckpt_path(
            training,
            eval,
            training.epochs,
            "precache_{}_{}".format(training.dataset, training.model),
            "npy",
        )
        np.save(save_path, results)

    else:
        # get loss function
        loss_fn = build_loss_fn(training)
        print("CONSTRUCTED LOSS FUNCTION")

        # train the model with default=BT
        results = train(model, loaders, optimizer, loss_fn, training, eval,
                        args.logging.use_wandb)

        # save results
        save_path = gen_ckpt_path(
            training,
            eval,
            training.epochs,
            "results_{}_alpha".format(training.dataset),
            "npy",
        )
        np.save(save_path, results)
    return save_path


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
    args.training.train_dataset = args.training.train_dataset.format(
        dataset=args.training.dataset
    )
    args.training.val_dataset = args.training.val_dataset.format(
        dataset=args.training.dataset
    )
    logging_modelname = args.training.model
    logging_modelname = logging_modelname.replace("proj", "")
    logging_modelname = logging_modelname.replace("feat", "")
    logging_jobtype = args.training.algorithm
    if logging_jobtype == 'linear':
        logging_jobtype = f'{args.eval.train_algorithm}_{logging_jobtype}'
    if args.logging.use_wandb:
        start_wandb_server(train_config_dict=args.training.__dict__,
                           eval_config_dict=args.eval.__dict__,
                           wandb_group=args.logging.wandb_group,
                           wandb_project=args.logging.wandb_project,
                           exp_name=f'{logging_modelname}_' +\
                                    f'{args.training.algorithm}_' +\
                                    f'{args.training.seed}',
                           exp_group=f'{logging_modelname}',
                           exp_job_type=f'{logging_jobtype}'
                           )

    # train model
    start_time = time.time()
    save_fname = run_experiment(args)

    # wrapup experiments with logging key variables
    print(f"Total time: {time.time() - start_time}")
    print(f"Results saved to {save_fname}")

    if args.logging.use_wandb: 
        stop_wandb_server()
