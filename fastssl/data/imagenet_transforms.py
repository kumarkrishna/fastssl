import numpy as np
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from ffcv.transforms import RandomHorizontalFlip, RandomResizedCrop, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage, NormalizeImage
from fastssl.data.custom_ffcv_transforms import ColorJitter, RandomGrayscale
from fastssl.data.custom_transforms import ReScale, GaussianBlur, Solarization
from PIL import Image

# mean, std for normalized dataset
CIFAR_MEAN = [0.485, 0.456, 0.406]  # official ImageNet mean
CIFAR_STD = [0.229, 0.224, 0.225]  # official ImageNet std
# write_ffcv_datasets.py --> ffcv dataset stats... [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
CIFAR_FFCV_MEAN = [125.307, 122.961,
                   113.8575]  # write_ffcv_double_datasets.py or write_ffcv_datasets.py --> ffcv dataset stats...
CIFAR_FFCV_STD = [51.5865, 50.847,
                  51.255]  # write_ffcv_double_datasets.py or write_ffcv_datasets.py --> ffcv dataset stats...
STL_MEAN = [0.4467, 0.4398, 0.4066]  # write_ffcv_datasets.py --> ffcv dataset stats... (Train)
STL_STD = [0.2242, 0.2215, 0.2239]  # write_ffcv_datasets.py --> ffcv dataset stats... (Train)
STL_FFCV_MEAN = [113.9112, 112.1515, 103.6948]  # write_ffcv_datasets.py --> ffcv dataset stats... (Train)
STL_FFCV_STD = [57.1603, 56.4828, 57.0975]  # write_ffcv_datasets.py --> ffcv dataset stats... (Train)
ImageNet_MEAN = [0.485, 0.456, 0.406]  # official ImageNet mean
ImageNet_STD = [0.229, 0.224, 0.225]  # official ImageNet std
ImageNet_FFCV_MEAN = [0.4820, 0.4538, 0.3998]
ImageNet_FFCV_STD = [0.2208, 0.2165, 0.2157]


class ImageNetTransformFFCV():
    """
    Defines a list of FFCV transforms for SimCLR on STL
    """

    def __init__(self):
        self.transform_list = [RandomResizedCrop(224, interpolation=Image.BICUBIC),
                               RandomHorizontalFlip(flip_prob=0.5),
                               ColorJitter(jitter_prob=0.8,
                                           brightness=0.4,
                                           contrast=0.4,
                                           saturation=0.2,
                                           hue=0.1),
                               RandomGrayscale(p=0.2),
                               GaussianBlur(p=0.1),
                               NormalizeImage(mean=np.array(ImageNet_FFCV_MEAN),
                                              std=np.array(ImageNet_FFCV_STD), type=np.float32)]
        self.dataset_side_length = 224  # only matters if using RandomResizedCropRGBImageDecoder
        self.dataset_resize_scale = (0.1, 0.1)  # only matters if using RandomResizedCropRGBImageDecoder
        self.dataset_resize_ratio = (0.75, 4 / 3)  # only matters if using RandomResizedCropRGBImageDecoder


class ImageNetClassifierTransform(nn.Module):
    """
    Generates transformed images, primarily for image classification.
    """

    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=ImageNet_MEAN,
                                 std=ImageNet_STD)
        ])

    def forward(self, x):
        return self.transform(x)


class simclr_imagenet_transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),
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
                                             ])

    def forward(self, x):
        return self.transform(x)

# transform for datasampler
class TransformImagenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            # Solarization(p=0.0),
            transforms.ConvertImageDtype(torch.float16),
            # transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            # Solarization(p=0.2),
            transforms.ConvertImageDtype(torch.float16),
            # transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            # Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            # Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


class TransformGPU:  # Used in BarlowTwins
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x1, x2):
        y1 = self.transform(x1)
        y2 = self.transform_prime(x2)
        return y1, y2
