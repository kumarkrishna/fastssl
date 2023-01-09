# Ffcv dataloaders for imagenet to run barlow twins
import torch
import torchvision
import numpy as np
from imagenet_pipeline.imagenet_transforms import ImageNetClassifierTransform, Transform

DEFAULT_CROP_RATIO = 224 / 256
ImageNet_MEAN = [0.485, 0.456, 0.406]  # official ImageNet mean
ImageNet_STD = [0.229, 0.224, 0.225]  # official ImageNet std
ImageNet_FFCV_MEAN = [0.4820, 0.4538, 0.3998]
ImageNet_FFCV_STD = [0.2208, 0.2165, 0.2157]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255



def get_ssltrain_imagenet_pytorch_dataloaders(
        data_dir=None, batch_size=None, num_workers=None
):
    paths = {
        'train': data_dir + '/train',
    }

    loaders = {}

    for name in ['train']:
        dataset = torchvision.datasets.ImageFolder(paths[name], Transform())
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers,
            pin_memory=True, shuffle=True, drop_last=True
        )
        loaders[name] = loader

    return loaders


def get_ssltrain_imagenet_pytorch_dataloaders_distributed(
        data_dir=None, batch_size=None, num_workers=None, world_size=None
):
    paths = {
        'train': data_dir + '/train',
    }

    loaders = {}

    for name in ['train']:
        dataset = torchvision.datasets.ImageFolder(paths[name], Transform())
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        assert batch_size % world_size == 0
        per_device_batch_size = batch_size // world_size
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=per_device_batch_size, num_workers=num_workers,
            pin_memory=True, sampler=sampler
        )
        loaders[name] = loader

    return loaders, sampler


def get_eval_imagenet_pytorch_dataloaders(
        data_dir=None,
        batch_size=None,
        num_workers=None,
        device="cuda:0"):
    """
    Create pytorch compatible dataloaders for ImageNet.
    """
    loaders = {}
    paths = {
        'train': data_dir + '/train',
        'test': data_dir
    }
    for name in ['train']:
        dataset = torchvision.datasets.ImageFolder(paths[name], ImageNetClassifierTransform())
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers,
            pin_memory=True, shuffle=True, drop_last=True
        )
        loaders[name] = loader

    for name in ['test']:
        dataset = torchvision.datasets.ImageNet(root=data_dir, train='val', download=False,
                                                transform=ImageNetClassifierTransform())
        loaders[name] = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return loaders
