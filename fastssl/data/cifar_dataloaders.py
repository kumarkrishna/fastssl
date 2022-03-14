from typing import List 

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation


from ffcv.transforms import RandomHorizontalFlip, Cutout, \
	RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze

from fastssl.data.cifar_transforms import CifarTransform, CifarClassifierTransform

def cifar_ffcv(
    train_dataset=None,
    val_dataset=None,
    batch_size=None,
    num_workers=None):
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
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True), 
            Squeeze()]
        
        image_pipeline : List[Operation] = [
            SimpleRGBImageDecoder(),
            ToTensor(),
            ToTorchImage(),
            ToDevice('cuda:0', non_blocking=True),
            CifarTransform()
        ]

        ordering = OrderOption.RANDOM if split == 'train' else OrderOption.SEQUENTIAL

        loaders[split] = Loader(
            datadir[split], batch_size=batch_size,  
            num_workers=num_workers, os_cache=True,
            order=ordering, drop_last=True,
            pipelines={'image' : image_pipeline, 'labels' : label_pipeline}
           )

    return loaders


def cifar_classifier_ffcv(
    train_dataset=None,
    val_dataset=None,
    batch_size=None,
    num_workers=None):
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
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True), 
            Squeeze()]
        
        image_pipeline : List[Operation] = [
            SimpleRGBImageDecoder(),
            ToTensor(),
            ToTorchImage(),
            ToDevice('cuda:0', non_blocking=True),
            CifarClassifierTransform()
        ]

        ordering = OrderOption.RANDOM if split == 'train' else OrderOption.SEQUENTIAL

        loaders[split] = Loader(
            datadir[split], batch_size=batch_size,  
            num_workers=num_workers, os_cache=True,
            order=ordering, drop_last=True,
            pipelines={'image' : image_pipeline, 'labels' : label_pipeline}
           )

    return loaders