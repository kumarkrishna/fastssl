"""
## NOTE: Supports FFCV and PyTorch dataloaders. 
* linear classifier: 
    * FP16 is sufficient, reasonable speed and little drop in accuracy (Acc@1 is within +/- 0.1)
* SSL : Seems like autocast is important for good performance.
"""
from typing import List 

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
import numpy as np
import torch
import torchvision.transforms as tvt
from ffcv.transforms import RandomResizedCrop, RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze

from fastssl.data.cifar_transforms import CifarTransform, CifarClassifierTransform, SSLPT, ReScale, STL10ClassifierTransform

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def stl10_pt(
    datadir,
    batch_size=None,
    num_workers=None,
    device="cuda:0",
    splits=['train', 'test']):
    """
    Create pytorch compatible dataloaders for CIFAR-10.
    """
    loaders = {}
    for split in splits:
        dataset = torchvision.datasets.STL10(
            root=datadir, split=split, download=True,
            transform=SSLPT())
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loaders


def stl10_classifier_pt(
    datadir,
    batch_size=None,
    num_workers=None,
    device="cuda:0",
    splits=['train', 'test']):
    """
    Create pytorch compatible dataloaders for CIFAR-10.
    """
    loaders = {}
    for split in splits:
        dataset = torchvision.datasets.STL10(
            root=datadir, split=split, download=True,
            transform=STL10ClassifierTransform())
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loaders