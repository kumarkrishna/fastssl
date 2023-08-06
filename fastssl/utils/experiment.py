"""
General Experiment class.

Supports:
    - logging
    - model training
    - model evaluation
    - diagnostics
"""
import glob
import logging
import numpy as np
import os
from pathlib import Path

import time
from typing import Any
from tqdm import tqdm


import torch
from torch import optim
from torch.cuda.amp import autocast

from fastssl.utils.base import set_seeds, get_args_from_config, merge_with_args
from fastssl.utils.base import start_wandb_server, stop_wandb_server, log_wandb
from fastssl.utils import powerlaw, Saver, Timer
from fastssl.data import DATALOADER_REGISTRY, precache_dataloader
from fastssl import models

class Experiment:
    def __init__(self, config):
        self.config = config

        # setup system 
        self.build_system()
        
        # build dataloader
        self.dataloaders = self.build_dataloader()

        # build model
        self.build_model()

        if self.config.train.precache:
            self._build_train_precache()

        # build optimizer
        self.build_optimizer()

        # build loss fn
        self.build_loss_fn()

    def build_system(self):
        # set up seeds
        set_seeds(self.config.train.seed)
        # set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # set up logging
        self.logger = logging.getLogger(__name__)
        # build saver
        self.saver = Saver(self.config)
        # build timer
        self.timer = Timer()

        # setup wandb
        modelidx = self.config.train.model.replace("proj", "").replace("feat", "")
        algoidx = self.config.valid.algorithm if self.config.train.mode == 'eval' else self.config.train.algorithm
        if self.config.train.mode == 'eval':
            exp_job_type = f'{self.config.train.algorithm}_{self.config.valid.algorithm}'
        else:
            exp_job_type = self.config.train.algorithm

        if self.config.logging.use_wandb:
            start_wandb_server(
                    train_config_dict=self.config.train.__dict__,
                    eval_config_dict=self.config.valid.__dict__,
                    wandb_group=self.config.logging.wandb_group,
                    wandb_project=self.config.logging.wandb_project,
                    exp_name='{}_{}_{}'.format(
                        modelidx,
                        algoidx,
                        self.config.train.seed
                        ),
                    exp_group=f'{modelidx}',
                    exp_job_type=f'{exp_job_type}'
                    )


    def maybe_use_precache_features(self):
        if os.path.splitext(self.config.train.train_dataset)[-1] == ".npy":
            self.logger.info("Using pre-cached features for training")
            return precache_dataloader(
                self.config.train.train_dataset,
                self.config.train.valid_dataset,
                self.train.batch_size,
                self.train.num_workers,
            )
        else:
            return None

    def _search_precache_files(self):
        # TODO (krishna): FIXME
        self.logger.info("Searching for pre-cached features")


        ckpt_path = self.get_save_path()
        dirname = os.path.dirname(ckpt_path)
        possible_fnames = glob.glob(os.path.join(dirname, "*.npy"))

        possible_fnames = [
            os.path.basename(fname) for fname in possible_fnames]
        # filter out files that don't have dataset/model in name
        filter_fn = lambda s: self.config.train.dataset not in s and self.config.train.model not in s
        possible_fnames = list(filter(filter_fn, possible_fnames))

        if len(possible_fnames) == 0:
            self.logger.info("No precached file dounf! running linear eval without precache")
            self.config.valid.use_precache = False
            return

        try:
            extract_seed_val = lambda x: int(x.split(".npy")[0].split("seed_")[-1])
            seed_files = [
                f for f in possible_fnames if self.config.train.seed == extract_seed_val(f)
            ]
            assert len(seed_files) == 1
            fname = seed_files[0]
            print("Using precache file {}".format(os.path.join(dirname, fname)))
        except:
            fname = possible_fnames[0]
            print(
                "Could not find the correct seed value ({}), using {}".format(
                    self.config.train.seed, os.path.join(dirname, fname)
                )
            )
        self.config.train.train_dataset = os.path.join(dirname, fname)
        self.config.train.valid_dataset = os.path.join(dirname, fname)

        self.logger.info(f"Train dataset: {self.config.train.train_dataset}")
        self.logger.info(f"Valid dataset: {self.config.train.valid_dataset}")

    
    def _build_train_precache(self):
        self.logger.info("Building pre-cached features for training")
        # remove the readout layer
        self.model._modules["fc"] = torch.nn.Identity()
        precache = self.get_precache()
        ckpt_path = self.get_ckpt_path()
        np.save(ckpt_path, precache)
        return ckpt_path


    def build_dataloader(self):
        # build dataloader
        self.logger.info("Building dataloader")

        if self.config.valid.use_precache:
            self._search_precache_files()

        dataloaders = self.maybe_use_precache_features()
        if dataloaders is not None:
            return dataloaders
        
        stage = "classifier" if "linear" in self. config.train.algorithm else "pretrain"
        dataloader_fn_name = f"{self.config.train.dataset}_{stage}_ffcv"
        dataloader_fn = DATALOADER_REGISTRY.get(dataloader_fn_name, None)
    
        if dataloader_fn is None:
            raise ValueError(f"Unknown dataloader: {dataloader_fn_name}")
        
        self.logger.info(f"Using dataloader: {dataloader_fn_name}")
        dataloaders = dataloader_fn(
            self.config.train.train_dataset,
            self.config.train.valid_dataset,
            self.config.train.batch_size,
            self.config.train.num_workers,
            num_augmentations=self.config.train.num_augmentations
        )
        return dataloaders

    def _build_linear_model_kwargs(self):
        ckpt_path = self.saver.get_ckpt_path()
        model_type = "" if self.config.valid.use_precache else self.config.train.model

        def get_hidden_dim(model_type):
            if "proj" in model_type:
                return self.config.train.projector_dim
            else:
                return 2048

        return {
            "bkey": model_type,
            "ckpt_path": ckpt_path,
            "dataset": self.config.train.dataset,
            # TODO(krishna) : replace feat dim, and assert we use hidden (not projector)
            "hidden_dim": get_hidden_dim(self.config.train.model),
            "proj_hidden_dim": self.config.train.hidden_dim
            if self.config.eval.train_algorithm in ("byol")
            else self.config.train.projector_dim,
            "num_classes": self.config.train.num_classes,
        }


    def build_model(self):
        if self.config.train.algorithm == "linear":
            model_kwargs = self._build_linear_model_kwargs()
        else:
            model_kwargs = {
                "bkey": self.config.train.model,
                "dataset": self.config.train.dataset,
                "hidden_dim": self.config.train.hidden_dim,
                "projector_dim": self.config.train.projector_dim,
            }

        # build the model
        # the algorithm should enforce whether hidden_dim == projector_dim
        self.model = (
            models.__dict__[self.config.train.model_type](**model_kwargs)
            .to(memory_format=torch.channels_last)
            .cuda()
        )

    def build_loss_fn(self):
        # ALGORITHM_TO_LOSS_FN = {
        #     "btwins": BTwinsLoss,
        #     "byol": BYOLLoss,
        #     "simclr": SimCLRLoss,
        #     "smalr": SMALRLoss,
        #     "linear": XentLoss,
        # }
        loss_fn_kwargs = {
            "temperature": self.config.train.temperature,
            "lambd": self.config.train.lambd,
            "label_smoothing_coeff": self.config.train.label_smoothing_coeff
        }
        self.loss_fn = models.__dict__[self.config.train.loss_fn_type](**loss_fn_kwargs)

    def build_optimizer(self):
        """
        Build optimizer.
        """
        # TODO (krishna): add support for AdamW, schedulers
        # TODO (krishna): add default values for lr/wd
        if self.config.train.optim_type == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.train.lr,
                weight_decay=self.config.train.weight_decay,
            )
        else:
            raise NotImplementedError
    
    def forward(self, sample):
        if self.scaler:
            with autocast():
                # TODO (krishna): add support for BYOL
                loss_dict = self.loss_fn(self.model, sample)
                loss = loss_dict["loss"]

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            loss_dict = self.loss_fn(self.model, sample)
            loss = loss_dict["loss"]
            loss.backward()
            self.optimizer.step()
        
        return loss

    def train_step(self, epoch=0):
        total_loss, num_batches = 0.0, 0
        train_bar = tqdm(self.dataloaders["train"], desc="Train")

        self.model.train()
        if self.config.train.mode == "eval" and self.config.valid.algorithm == "linear":
            self.model.backbone.eval()

        self.logger.info("Training epoch {}".format(epoch))
        for sample in train_bar:
            if type(sample[0]) == type(sample[1]) and sample[0].shape == sample[1].shape:
                # inp is a tuple with the two augmentations.
                # This is legacy implementation of ffcv for dual augmentations
                sample = ((sample[0], sample[1]), None)
            
            # clean up for backward pass
            self.optimizer.zero_grad()
            # forward pass
            loss = self.forward(sample)
            # logging
            total_loss += loss.item()
            num_batches += 1

            if self.config.train.algorithm == "BYOL":
                # TODO(krishna): add momentum update
                raise NotImplementedError

            train_bar.set_description(
                "Train Epoch: [{}/{}] Loss: {:.4f}".format(
                epoch, self.config.train.epochs, total_loss / num_batches
                )
            )
        return total_loss / num_batches

    def valid_step(self, epoch=0):
        self.model.eval()
        self.logger.info("Validating epoch {}".format(epoch))
        test_dataloader = self.dataloaders["test"]
        total_correct_1, total_correct_5, total_samples = 0.0, 0.0, 0
        acc_1, acc_5 = 0.0, 0.0
        test_bar = tqdm(test_dataloader, desc="Test")

        for sample in test_bar:
            sample = list(sample)
            sample = [x.cuda(non_blocking=True) for x in sample]
            target = target.cuda(non_blocking=True)
            total_samples += sample[0].shape[0]

            with autocast():
                logits = self.model(sample)
                # use torch metrics to get the accuracy numbers?

            test_bar.set_description(
                "{} Epoch: [{}/{}] ACC@1: {:.2f}% ACC@5: {:.2f}%".format(
                    "Test", epoch, self.config.train.epochs, acc_1, acc_5
                )
            )
        return acc_1, acc_5

    def train(self):
        # how to track metrics
        metrics = {"train_loss": [], "test_acc_1": [], "test_acc_5": []}
        if self.config.train.track_alpha:
            metrics.update({"alpha": [], "eigenspectrum": [], "R2": [], "R2_100": []})
        
        if self.config.train.use_autocast:
            self.logger.info("Using autocast for training")
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.logger.info("Not using autocast for training")
            self.scaler = None
        
        self.timer.start()
        # run training 
        for epoch in range(1, self.config.train.epochs + 1):
            train_loss = self.train_step(epoch)
            metrics["train_loss"].append(train_loss)

            if epoch % self.config.train.log_interval == 0:
                ckpt_path = self.saver.get_save_path(epoch=epoch)
                state = dict(
                    epoch=epoch + 1,
                    model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(),
                )
                torch.save(state, ckpt_path)
            if self.config.logging.use_wandb:
                log_wandb(metrics)
        self.timer.end()

        self.logger.info("Training complete")
        self.logger.info("Total time taken: {}".format(self.timer.total_time))
        self.logger.info("Model saved at: {}".format(ckpt_path))

        return metrics

    def run(self, mode="train"):
        if mode == "train":
            metrics = self.train()
            self.save_info(metrics)

    def stop(self):
        if self.config.logging.use_wandb:
            stop_wandb_server()

    def save_info(self, info):
        save_path = self.saver.get_save_path()
        np.save(save_path, info)
        self.logger.info(f"Saved info to {save_path}")


    def precache_outputs(self):
        self.model.eval()
        trainset_outputs, trainset_labels = [], []
        train_bar = tqdm(self.dataloaders["train"], desc="Precaching trainset outputs")

        for sample in train_bar:
            sample = list(sample)
            target = sample.pop(1)
            sample = [x.cuda(non_blocking=True) for x in sample]
            with autocast():
                with torch.no_grad():
                    embed_augs = [self.model(x) for x in sample]
                    # mean of features across different augmentations of each image
                    embed_mean = torch.mean(torch.stack(embed_augs), dim=0)
                    # out = model(data)
            trainset_outputs.append(embed_mean.data.cpu().float())
            trainset_labels.append(target.data.cpu())
        trainset_outputs = torch.cat(trainset_outputs)
        trainset_labels = torch.cat(trainset_labels)

        testset_outputs = []
        testset_labels = []
        test_bar = tqdm(self.dataloaders["test"], desc="Test set")
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
                    out_augs = [self.model(x) for x in inp]
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
                
