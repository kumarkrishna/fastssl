# Ffcv dataloaders for imagenet to run barlow twins
import torch
import torchvision
from typing import List
from ffcv.fields.decoders import IntDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Squeeze, ToDevice, ToTensor, ToTorchImage, Convert
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder, CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
from fastssl.data.custom_transforms import ReScale, GaussianBlur, Solarization
import torchvision.transforms as transforms
from PIL import Image
from fastssl.data.imagenet_transforms import TransformImagenet, ImageNetTransformFFCV, ImageNetClassifierTransform, Transform, TransformGPU

DEFAULT_CROP_RATIO = 224/256

def get_sseval_imagenet_ffcv_dataloaders(
        train_dataset=None, val_dataset=None, batch_size=None, num_workers=None
):
    paths = {
        'train': train_dataset,
        'val': val_dataset
    }

    loaders = {}

    for name in ['train', 'val']:
        image_pipeline: List[Operation] = [
            CenterCropRGBImageDecoder((224, 224), ratio=DEFAULT_CROP_RATIO),
            RandomHorizontalFlip(),
            ToTensor(),
            #ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            torchvision.transforms.ConvertImageDtype(torch.float16),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ]
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            #ToDevice('cuda:0')
        ]
        ordering = OrderOption.QUASI_RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name],
                               batch_size=batch_size,
                               num_workers=num_workers,
                               order=ordering,
                               drop_last=(name == 'train'),
                               pipelines={
                                   'image': image_pipeline,
                                   'label': label_pipeline
                               }
                               )

    return loaders

def get_ssltrain_imagenet_ffcv_dataloaders(
        train_dataset=None, batch_size=None, num_workers=None
):
    paths = {
        'train': train_dataset,
    }

    loaders = {}

    for name in ['train']:
        image_pipeline: List[Operation] = [
            CenterCropRGBImageDecoder((224, 224), ratio=DEFAULT_CROP_RATIO),
            ToTensor(),
            #ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            TransformImagenet()
        ]

        ordering = OrderOption.QUASI_RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name],
                               batch_size=batch_size,
                               num_workers=num_workers,
                               order=ordering,
                               drop_last=(name == 'train'),
                               pipelines={
                                   'image': image_pipeline,
                               }
                               )
    return loaders


def get_simclr_train_imagenet_ffcv_dataloaders(
        train_dataset=None, val_dataset=None, batch_size=None, num_workers=None
):
    paths = {
        'train': train_dataset,
        'test': val_dataset
    }

    loaders = {}

    for name in ['train']:
        image_pipeline1: List[Operation] = [
            CenterCropRGBImageDecoder((224, 224), ratio=DEFAULT_CROP_RATIO),
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            ImageNetTransformFFCV(),
            # TransformImagenet()
            Convert(torch.float32)
        ]
        image_pipeline2 = List[Operation] = [
            CenterCropRGBImageDecoder((224, 224), ratio=DEFAULT_CROP_RATIO),
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            ImageNetTransformFFCV(),
            # TransformImagenet()
            Convert(torch.float32)
        ]

        ordering = OrderOption.QUASI_RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name],
                               batch_size=batch_size,
                               num_workers=num_workers,
                               order=ordering,
                               drop_last=(name == 'train'),
                               pipelines={
                                   'image1': image_pipeline1,
                                   'image2': image_pipeline2
                               }
                               )

    for name in ['test']:
        label_pipeline  = List[Operation] = [
            IntDecoder(),
            ToTensor(),
            ToDevice("cuda:0"),
            Squeeze()]

        image_pipeline: List[Operation] = [
            CenterCropRGBImageDecoder((224, 224), ratio=DEFAULT_CROP_RATIO),
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(torch.float32),
            # ReScale(1.0/255.0),
            ImageNetClassifierTransform
        ]
        ordering = OrderOption.RANDOM #if split == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(
            paths[name],
            batch_size=batch_size,
            num_workers=num_workers,
            os_cache=True,
            order=ordering,
            drop_last=False,
            pipelines={'image' : image_pipeline, 'label' : label_pipeline}
        )
    return loaders


def get_simclr_eval_imagenet_ffcv_dataloaders(
        train_dataset=None, val_dataset=None, batch_size=None, num_workers=None
):
    paths = {
        'train': train_dataset,
        'test': val_dataset
    }

    loaders = {}

    for name in ['train', 'test']:
        image_pipeline: List[Operation] = [
            CenterCropRGBImageDecoder((224, 224), ratio=DEFAULT_CROP_RATIO),
            # RandomHorizontalFlip(),  # maybe get rid of?
            ToTensor(),
            #ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            torchvision.transforms.ConvertImageDtype(torch.float16),
            ImageNetClassifierTransform
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ]
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            #ToDevice('cuda:0')
        ]
        ordering = OrderOption.QUASI_RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name],
                               batch_size=batch_size,
                               num_workers=num_workers,
                               order=ordering,
                               drop_last=(name == 'train'),
                               pipelines={
                                   'image': image_pipeline,
                                   'label': label_pipeline
                               }
                               )

    return loaders


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


