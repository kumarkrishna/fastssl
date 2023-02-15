"""
## NOTE: Supports FFCV and PyTorch dataloaders. 
* linear classifier: 
    * FP16 is sufficient, reasonable speed and little drop in accuracy (Acc@1 is within +/- 0.1)
* SSL : Seems like FP32 is important for good performance.
    * 
"""
from typing import List 

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
import numpy as np
import torch
from ffcv.transforms import RandomResizedCrop, RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze

from fastssl.data.cifar_transforms import CifarTransform, CifarClassifierTransform, SSLPT_CIFAR, ReScale, CifarTransformFFCV


import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

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
    # no rescaling required anymore!
    # if rescale:
    #     image_pipeline.append(ReScale(1.0/255.0))
    image_pipeline.append(transform_cls())

    return image_pipeline

def gen_image_pipeline_ffcv_test(device="cuda:0", transform_cls=None, rescale=False):
    # image_pipeline : List[Operation] = [SimpleRGBImageDecoder()]
    image_pipeline : List[Operation] = [RandomResizedCropRGBImageDecoder(
                                        output_size=(transform_cls.dataset_side_length,transform_cls.dataset_side_length),
                                        scale=transform_cls.dataset_resize_scale,ratio=transform_cls.dataset_resize_ratio)]
    if transform_cls:
        image_pipeline.extend(transform_cls.transform_list)

    image_pipeline.extend([
        ToTensor(),
        to_device(device),
        ToTorchImage(),
        Convert(torch.float32),
    ])

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

def gen_image_label_pipeline_ffcv_test(
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

    for split in ['train']:
        image_pipeline1 = gen_image_pipeline_ffcv_test(
            device=device, transform_cls=transform_cls, rescale=rescale)
        image_pipeline2 = gen_image_pipeline_ffcv_test(
            device=device, transform_cls=transform_cls, rescale=rescale)

        ordering = OrderOption.RANDOM #if split == 'train' else OrderOption.SEQUENTIAL
        # ordering = OrderOption.SEQUENTIAL #if split == 'train' else OrderOption.SEQUENTIAL

        # breakpoint()
        loaders[split] = Loader(
            datadir[split],
            batch_size=batch_size,  
            num_workers=num_workers,
            os_cache=True,
            order=ordering,
            drop_last=False,
            pipelines={'image1' : image_pipeline1,
                        'image2': image_pipeline2}
                        # 'label' : label_pipeline}     # The DoubleImage beton files don't have labels
           )

    for split in ['test']:
        label_pipeline  = gen_label_pipeline(device=device)
        image_pipeline = gen_image_pipeline(
            device=device, transform_cls=CifarClassifierTransform, rescale=rescale)

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

    # transform_cls = CifarTransform
    transform_cls = CifarTransformFFCV()
    return gen_image_label_pipeline_ffcv_test(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_cls=transform_cls,
        # rescale=True,
        rescale=False,
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

