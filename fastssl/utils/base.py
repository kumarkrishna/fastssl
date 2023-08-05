from argparse import ArgumentParser
import random, os
import torch
import numpy as np
import wandb

from fastargs import get_current_config

def set_seeds(seed, use_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)


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

def start_wandb_server(train_config_dict: dict, 
                       eval_config_dict: dict,
                       wandb_group: str,
                       wandb_project: str,
                       exp_name: str = None,
                       exp_group: str = None, 
                       exp_job_type: str = None,
                       ):
    log_config = {}
    for k,v in train_config_dict.items():
        log_config[f"train__{k}"] = v
    for k,v in eval_config_dict.items():
        log_config[f"eval__{k}"] = v

    wandb_dir = os.path.join(log_config['train__ckpt_dir'])

    try:
        wandb.init(entity=wandb_group,
                   project=wandb_project,
                   config=log_config,
                   dir=wandb_dir,
                   name=exp_name,
                   group=exp_group,
                   job_type=exp_job_type,
                   settings=wandb.Settings(_service_wait=150)
                   )
    except Exception as error:
        print(f"Error in wandb init, see: {error}")

def stop_wandb_server():
    wandb.finish()
    
def log_wandb(data_dict: dict, step: int = None, skip_keys: list = None):
    if skip_keys is None: skip_keys = []
    for k,v in data_dict.items():
        if k in skip_keys: continue
        log_step = None
        if isinstance(v, list):
            if len(v) == 0: continue
            v = v[-1]
        if isinstance(v, tuple):
            log_step = v[0]
            v = v[1]
        if log_step is None:
            log_step = step
        try:
            wandb.log({k: v}, step=step, commit=False)
        except:
            print(f"WARNING: Not logging {k} to wandb!")
    wandb.log({}, commit=True)  # finish incremental logging