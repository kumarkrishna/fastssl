# Ffcv dataloaders for imagenet to run barlow twins
import torch
import torchvision
from typing import List
import numpy as np
from ffcv.fields.decoders import IntDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Squeeze, ToDevice, ToTensor, ToTorchImage, Convert, NormalizeImage
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder, CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
from fastssl.data.custom_transforms import ReScale, GaussianBlur, Solarization
import torchvision.transforms as transforms
from PIL import Image
from fastssl.data.imagenet_transforms import simclr_imagenet_transform, TransformImagenet, ImageNetTransformFFCV, ImageNetClassifierTransform, Transform, TransformGPU

DEFAULT_CROP_RATIO = 224/256
ImageNet_MEAN = [0.485, 0.456, 0.406]  # official ImageNet mean
ImageNet_STD = [0.229, 0.224, 0.225]  # official ImageNet std
ImageNet_FFCV_MEAN = [0.4820, 0.4538, 0.3998]
ImageNet_FFCV_STD = [0.2208, 0.2165, 0.2157]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255

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

simclr_transforms = [transforms.RandomHorizontalFlip(),
                                        transforms.RandomResizedCrop(size=96),
                                        transforms.RandomApply([
                                            transforms.ColorJitter(brightness=0.5,
                                                                   contrast=0.5,
                                                                   saturation=0.5,
                                                                   hue=0.1)
                                        ], p=0.8),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.GaussianBlur(kernel_size=9),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))
                                        ]

def create_train_loader(train_dataset, num_workers, batch_size,
                        distributed, in_memory):
    this_device = f'cuda:0'
    # paths = train_dataset

    res = 224
    decoder = RandomResizedCropRGBImageDecoder((res, res))
    image_pipeline1: List[Operation] = [decoder, RandomHorizontalFlip()]
    image_pipeline1.extend(simclr_transforms)
    image_pipeline1.extend([ToTensor(),ToDevice(torch.device(this_device), non_blocking=True), ToTorchImage(),
                            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)])

    image_pipeline2: List[Operation] = [decoder, RandomHorizontalFlip()]
    image_pipeline2.extend(simclr_transforms)
    image_pipeline1.extend([ToTensor(),ToDevice(torch.device(this_device), non_blocking=True), ToTorchImage(),
                            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)])
    loaders = {}

    order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
    loaders['train'] = Loader(train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=order,
                    os_cache=in_memory,
                    drop_last=True,
                    pipelines={
                        'image1': image_pipeline1,
                        'image2': image_pipeline2
                    },
                    distributed=distributed)

    return loaders


def create_val_loader(train_dataset, val_dataset, num_workers, batch_size,
                      resolution, distributed):
    this_device = f'cuda:0'
    paths = {
        'train': train_dataset,
        'test': val_dataset
    }

    loaders = {}
    for name in ['train', 'test']:
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(torch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(this_device),
                     non_blocking=True)
        ]

        loaders[name] = Loader(paths[name],
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)
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
            Convert(torch.float32)]
        image_pipeline1.extend(ImageNetTransformFFCV().transform_list)

        image_pipeline2 : List[Operation] = [
            CenterCropRGBImageDecoder((224, 224), ratio=DEFAULT_CROP_RATIO),
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(torch.float32)]
        image_pipeline1.extend(ImageNetTransformFFCV().transform_list)

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
        label_pipeline : List[Operation] = [
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
            ImageNetClassifierTransform()
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
            ImageNetClassifierTransform()
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


