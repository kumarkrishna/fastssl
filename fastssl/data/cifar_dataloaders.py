from typing import List 

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation


from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze

from fastssl.data.cifar_transforms import CifarTransform, CifarClassifierTransform


def to_device(device):
    if device == 'cuda:0':
        return ToDevice(device, non_blocking=True)
    else:
        return ToDevice("cpu")

def gen_image_pipeline(device="cuda:0", transform_cls=None):
    image_pipeline : List[Operation] = [
        SimpleRGBImageDecoder(),
        ToTensor(),
        ToTorchImage(),
        to_device(device),
        transform_cls()
    ]
    return image_pipeline

def gen_label_pipeline(device="cuda:0", transform_cls=None):
    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        to_device(device),
        Squeeze()]
    return label_pipeline

def gen_image_label_pipeline(
    train_dataset=None,
    val_dataset=None,
    batch_size=None,
    num_workers=None,
    transform_cls=None,
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
            device=device, transform_cls=transform_cls)

        ordering = OrderOption.RANDOM if split == 'train' else OrderOption.SEQUENTIAL

        loaders[split] = Loader(
            datadir[split],
            batch_size=batch_size,  
            num_workers=num_workers,
            os_cache=True,
            order=ordering,
            drop_last=True,
            pipelines={'image' : image_pipeline, 'labels' : label_pipeline}
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