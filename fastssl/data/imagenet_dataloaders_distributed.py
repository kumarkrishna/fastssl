# Ffcv dataloaders for imagenet to run barlow twins
import torch
import torchvision
from typing import List
from ffcv.fields.decoders import (
    IntDecoder,
    SimpleRGBImageDecoder,
)
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
    RandomHorizontalFlip, 
    Squeeze, 
    ToDevice, 
    ToTensor, 
    Convert,
    NormalizeImage,
    ToTorchImage, 
    RandomResizedCrop
)
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
from fastssl.data.custom_transforms import TransformImagenet, GaussianBlur, Solarization
import torchvision.transforms as transforms
from PIL import Image

from fastssl.data.cifar_transforms import (
    ImagenetClassifierTransform,
    ImagenetTransformFFCV
)

import numpy as np

IMG_SIZE = 128 #224
DEFAULT_CROP_RATIO = 128/256 #224/256


def to_device(device):
    if device == "cuda:0":
        return ToDevice(device, non_blocking=True)
    else:
        return ToDevice("cpu")


def gen_image_pipeline(device="cuda:0", transform_cls=None):
    image_pipeline: List[Operation] = [
        CenterCropRGBImageDecoder((IMG_SIZE,IMG_SIZE), ratio=DEFAULT_CROP_RATIO),
        ToTensor(),
        to_device(device),
        ToTorchImage(),
        # Convert(torch.float32),
        # NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
    ]

    if transform_cls:
        image_pipeline.extend(transform_cls().transform_list)

    return image_pipeline


def gen_image_pipeline_ffcv_ssl(device="cuda:0", transform_cls=None):
    if transform_cls:
        image_pipeline: List[Operation] = [
            # RandomResizedCrop((224, 224))
            RandomResizedCrop(
                output_size=(
                    transform_cls.dataset_side_length,
                    transform_cls.dataset_side_length,
                ),
                scale=transform_cls.dataset_resize_scale,
                ratio=transform_cls.dataset_resize_ratio,
            )
        ]

        image_pipeline.extend(transform_cls.transform_list)

    else:
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

    image_pipeline.extend(
        [
            ToTensor(),
            to_device(device),
            ToTorchImage(),
            # Convert(torch.float32),
            # NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
            # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))
        ]
    )

    return image_pipeline


def gen_label_pipeline(device="cuda:0", transform_cls=None):
    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        ToDevice("cuda:0"),
        Squeeze(),
    ]
    return label_pipeline


def gen_image_label_pipeline(
    train_dataset: str = None,
    val_dataset: str = None,
    batch_size: int = None,
    num_workers: int = None,
    transform_cls: ImagenetClassifierTransform = None,
    device: str = "cuda:0",
    num_augmentations: int = 1,
    transform_cls_augs: ImagenetTransformFFCV = None,
):
    """Generate image and label pipelines for supervised classification.

    Args:
        train_dataset (str, optional): path to train dataset. Defaults to None.
        val_dataset (str, optional): path to test dataset. Defaults to None.
        batch_size (int, optional): batch-size. Defaults to None.
        num_workers (int, optional): number of CPU workers. Defaults to None.
        transform_cls (ImagenetClassifierTransform, optional): Transforms to be applied for the original image. Defaults to None.
        device (str, optional): CPU/GPU. Defaults to 'cuda:0'.
        num_augmentations (int, optional): Number of total image augmentations. Defaults to 1.
        transform_cls_augs (ImagenetTransformFFCV, optional): Transforms to be applied to generate other augmentations. Defaults to None.

    Returns:
        loaders : dict('train': dataloader, 'test': dataloader)
    """

    datadir = {"train": train_dataset, "test": val_dataset}
    assert num_augmentations > 0, "Please use at least 1 augmentation for classifier."

    loaders = {}

    for split in ["train", "test"]:
        if datadir[split] is None: continue
        label_pipeline = gen_label_pipeline(device=device)
        image_pipeline = gen_image_pipeline(
            device=device, transform_cls=transform_cls
        )
        if num_augmentations > 1:
            image_pipeline_augs = [
                gen_image_pipeline_ffcv_ssl(
                    device=device, transform_cls=transform_cls_augs
                )
            ] * (num_augmentations - 1)
        else:
            image_pipeline_augs = []
        ordering = OrderOption.RANDOM if split == "train" else OrderOption.SEQUENTIAL
        # ordering = OrderOption.RANDOM #if split == 'train' else OrderOption.SEQUENTIAL

        pipelines = {"image": image_pipeline, "label": label_pipeline}
        custom_field_img_mapper = {}
        for i, aug_pipeline in enumerate(image_pipeline_augs):
            pipelines["image{}".format(i + 1)] = aug_pipeline
            custom_field_img_mapper["image{}".format(i + 1)] = "image"

        loaders[split] = Loader(
            datadir[split],
            batch_size=batch_size,
            num_workers=num_workers,
            os_cache=True,
            order=ordering,
            drop_last=False,
            distributed=True,
            # pipelines={'image' : image_pipeline, 'label' : label_pipeline}
            pipelines=pipelines,
            custom_field_mapper=custom_field_img_mapper,
        )
    return loaders


def gen_image_label_pipeline_ffcv_ssl(
    train_dataset: str = None,
    val_dataset: str = None,
    batch_size: int = None,
    num_workers: int = None,
    transform_cls: ImagenetTransformFFCV = None,
    device: str = "cuda:0",
    num_augmentations: int = 2,
):
    """Function for generating multiple augmentations from each image.

    Args:
        train_dataset (str, optional): Train dataset filename. Defaults to None.
        val_dataset (str, optional): Test dataset filename. Defaults to None.
        batch_size (int, optional): Batch size. Defaults to None.
        num_workers (int, optional): Number of CPU workers. Defaults to None.
        transform_cls (CifarTransformFFCV, optional): Transform object. Defaults to None.
        device (str, optional): CPU/GPU. Defaults to 'cuda:0'.
        num_augmentations (int, optional): Number of patches. Defaults to 2.

    Returns:
        loaders: dict('train': dataloader, 'test': dataloader)
    """

    datadir = {"train": train_dataset, "test": val_dataset}
    assert num_augmentations > 1, "Please use at least 2 augmentations for SSL."

    loaders = {}

    for split in ["train"]:
        if train_dataset is None: continue
        image_pipeline1 = gen_image_pipeline_ffcv_ssl(
            device=device, transform_cls=transform_cls
        )
        label_pipeline = gen_label_pipeline(device=device)
        image_pipeline_augs = [
            gen_image_pipeline_ffcv_ssl(
                device=device, transform_cls=transform_cls
            )
        ] * (
            num_augmentations - 1
        )  # creating other augmentations

        ordering = OrderOption.RANDOM  # if split == 'train' else OrderOption.SEQUENTIAL
        # ordering = OrderOption.SEQUENTIAL #if split == 'train' else OrderOption.SEQUENTIAL

        pipelines = {"image": image_pipeline1, "label": label_pipeline}
        custom_field_img_mapper = {}
        for i, aug_pipeline in enumerate(image_pipeline_augs):
            pipelines["image{}".format(i + 1)] = aug_pipeline
            custom_field_img_mapper["image{}".format(i + 1)] = "image"

        loaders[split] = Loader(
            datadir[split],
            batch_size=batch_size,
            num_workers=num_workers,
            os_cache=True,
            order=ordering,
            drop_last=False,
            pipelines=pipelines,
            distributed=True,
            custom_field_mapper=custom_field_img_mapper,
        )

    for split in ["test"]:
        if val_dataset is None: continue
        label_pipeline = gen_label_pipeline(device=device)
        image_pipeline = gen_image_pipeline(
            device=device, transform_cls=ImagenetClassifierTransform
        )

        ordering = (
            OrderOption.SEQUENTIAL
        )  # if split == 'train' else OrderOption.SEQUENTIAL

        loaders[split] = Loader(
            datadir[split],
            batch_size=batch_size,
            num_workers=num_workers,
            os_cache=True,
            order=ordering,
            drop_last=False,
            distributed=False,
            pipelines={"image": image_pipeline, "label": label_pipeline},
        )

    return loaders


def imagenet_ffcv_dist(
    train_dataset: str = None,
    val_dataset: str = None,
    batch_size: int = None,
    num_workers: int = None,
    device: str = "cuda:0",
    num_augmentations: int = 2,
):
    """Function to return dataloader for Imagenet-1k SSL

    Args:
        train_dataset (str, optional): Train dataset filename. Defaults to None.
        val_dataset (str, optional): Test dataset filename. Defaults to None.
        batch_size (int, optional): Batch size. Defaults to None.
        num_workers (int, optional): Number of CPU workers. Defaults to None.
        device (str, optional): CPU/GPU. Defaults to 'cuda:0'.
        num_augmentations (int, optional): Number of patches. Defaults to 2.

    Returns:
        loaders : dict('train': dataloader, 'test': dataloader)
    """

    # transform_cls = CifarTransform
    transform_cls = ImagenetTransformFFCV()
    gen_img_label_fn = gen_image_label_pipeline_ffcv_ssl
    return gen_img_label_fn(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_cls=transform_cls,
        device=device,
        num_augmentations=num_augmentations,
    )


def imagenet_classifier_ffcv_dist(
    train_dataset: str = None,
    val_dataset: str = None,
    batch_size: int = None,
    num_workers: int = None,
    device: str = "cuda:0",
    num_augmentations: int = 1,
):
    """Function to return dataloader for Imagenet-1k classification

    Args:
        train_dataset (str, optional): Train dataset filename. Defaults to None.
        val_dataset (str, optional): Test dataset filename. Defaults to None.
        batch_size (int, optional): Batch size. Defaults to None.
        num_workers (int, optional): Number of CPU workers. Defaults to None.
        device (str, optional): CPU/GPU. Defaults to 'cuda:0'.
        num_augmentations (int, optional): Number of patches. Defaults to 1.

    Returns:
        loaders : dict('train': dataloader, 'test': dataloader)
    """

    transform_cls = ImagenetClassifierTransform
    transform_cls_extra_augs = ImagenetTransformFFCV()
    return gen_image_label_pipeline(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_cls=transform_cls,
        device=device,
        num_augmentations=num_augmentations,
        transform_cls_augs=transform_cls_extra_augs,
    )