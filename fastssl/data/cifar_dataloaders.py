"""
## NOTE: Supports FFCV and PyTorch dataloaders. 
* linear classifier: 
    * FP16 is sufficient, reasonable speed and little drop in accuracy (Acc@1 is within +/- 0.1)
* SSL : Seems like FP32 is important for good performance.
    * 
"""
from typing import List
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze

from fastssl.data.custom_transforms import CifarTransform, CifarClassifierTransform, SSLPT_CIFAR, ReScale

import torch
from torch.utils.data import DataLoader
import torchvision

def to_device(device):
    if device == 'cuda:0':
        return ToDevice(device, non_blocking=True)
    else:
        return ToDevice("cpu")

def gen_image_pipeline(device="cuda:0", transform_cls=None, rescale=False):
    image_pipeline : List[Operation] = [
        SimpleRGBImageDecoder(),
        ToTensor(),
        to_device(device),
        ToTorchImage(),
        Convert(torch.float32),
    ]
    if rescale:
        image_pipeline.append(ReScale(1.0/255.0))
    image_pipeline.append(transform_cls())

    return image_pipeline

def gen_label_pipeline(device="cuda:0", transform_cls=None):
    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        ToDevice("cuda:0"),
        Squeeze()]
    return label_pipeline

def gen_image_label_pipeline(
    train_dataset=None,
    val_dataset=None,
    batch_size=None,
    num_workers=None,
    transform_cls=None,
    rescale=False,
    device='cuda:0'):
    """
    Args:
        train_dataset : path to train dataset
        val_dataset   : path to test dataset 
        batch_size    : batch-size
        num_workers   : number of workers
    Returns 
        loaders       : dict('train': dataloader, 'test': dataloader)
    """

    datadir = {
        'train': train_dataset,
        'test': val_dataset
    }
    
    loaders = {}

    for split in ['train', 'test']:
        label_pipeline  = gen_label_pipeline(device=device)
        image_pipeline = gen_image_pipeline(
            device=device, transform_cls=transform_cls, rescale=rescale)

        # ordering = OrderOption.RANDOM if split == 'train' else OrderOption.SEQUENTIAL
        ordering = OrderOption.RANDOM #if split == 'train' else OrderOption.SEQUENTIAL

        loaders[split] = Loader(
            datadir[split],
            batch_size=batch_size,  
            num_workers=num_workers,
            os_cache=True,
            order=ordering,
            drop_last=False,
            pipelines={'image' : image_pipeline, 'label' : label_pipeline}
           )
    return loaders

def cifar_ffcv(
    train_dataset=None,
    val_dataset=None,
    batch_size=None,
    num_workers=None,
    device="cuda:0"):

    transform_cls = CifarTransform
    return gen_image_label_pipeline(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_cls=transform_cls,
        rescale=True,
        device=device)

def cifar_classifier_ffcv(
    train_dataset=None,
    val_dataset=None,
    batch_size=None,
    num_workers=None,
    device="cuda:0"):
    
    transform_cls = CifarClassifierTransform
    return gen_image_label_pipeline(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_cls=transform_cls,
        device=device)

def cifar_pt(
    datadir,
    batch_size=None,
    num_workers=None,
    device="cuda:0"):
    """
    Create pytorch compatible dataloaders for CIFAR-10.
    """
    loaders = {}
    for split in ['train', 'test']:
        dataset = torchvision.datasets.CIFAR10(
            root=datadir, train=split == 'train', download=True,
            transform=SSLPT_CIFAR())
        loaders[split] = DataLoader(
            # dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return loaders

