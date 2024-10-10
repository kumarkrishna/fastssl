import numpy as np
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from ffcv.transforms import (
    RandomHorizontalFlip, 
    RandomColorJitter, 
    RandomResizedCrop, 
    RandomSolarization, 
    Cutout, 
	RandomTranslate, 
    Convert, 
    ToDevice, 
    ToTensor, 
    ToTorchImage, 
    NormalizeImage
)
from fastssl.data.custom_ffcv_transforms import ColorJitter, RandomGrayscale

# mean, std for normalized dataset
CIFAR_MEAN = [0.485, 0.456, 0.406]
CIFAR_STD = [0.229, 0.224, 0.225]
CIFAR_FFCV_MEAN = [125.307, 122.961, 113.8575]
CIFAR_FFCV_STD = [51.5865, 50.847, 51.255]
STL_MEAN = [0.4467, 0.4398, 0.4066]
STL_STD = [0.2242, 0.2215, 0.2239]
STL_FFCV_MEAN = [113.9112, 112.1515, 103.6948]
STL_FFCV_STD = [57.1603, 56.4828, 57.0975]
IMAGENET_FFCV_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_FFCV_STD = np.array([0.229, 0.224, 0.225]) * 255
# IMAGENET_FFCV_MEAN = np.array([0.0, 0.0, 0.0]) * 255
# IMAGENET_FFCV_STD = np.array([1.0, 1.0, 1.0]) * 255

class ReScale(nn.Module):
    def __init__(self, scale):
        super(ReScale, self).__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

class CifarTransformFFCV():
    """
    Defines a list of FFCV transforms for SSL on CIFAR
    """
    def __init__(self):
        self.transform_list = [
                                RandomHorizontalFlip(flip_prob=0.5),
                                ColorJitter(jitter_prob=0.8,
                                            brightness=0.4, 
                                            contrast=0.4,
                                            saturation=0.4, 
                                            hue=0.0),
                                RandomGrayscale(p=0.2),
                                NormalizeImage(mean=np.array(CIFAR_FFCV_MEAN),
                                        std=np.array(CIFAR_FFCV_STD),type=np.float32)
                                ]
        self.dataset_side_length = 32
        self.dataset_resize_scale = (0.08,1.0)
        self.dataset_resize_ratio = (0.75,4/3)

class CifarTransform(nn.Module):
    """
    Generates pair of transformed images, primarily for SSL.
    """
    def __init__(self):
        super().__init__()        
        self.transform = transforms.Compose([
            # transforms.ConvertImageDtype(torch.float32),
            # ReScale(1/255.),
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=0.4, contrast=0.4,
                    saturation=0.4, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.Normalize(mean=CIFAR_MEAN,
                                 std=CIFAR_STD),
            
        ])

    def forward(self, x):
        y1 = self.transform(x)
        y2 = self.transform(x)
        return (y1, y2)

class STLTransformFFCV():
    """
    Defines a list of FFCV transforms for SSL on STL
    """
    def __init__(self):
        self.transform_list = [
                                RandomHorizontalFlip(flip_prob=0.5),
                                ColorJitter(jitter_prob=0.8,
                                            brightness=0.4, 
                                            contrast=0.4,
                                            saturation=0.4, 
                                            hue=0.0),
                                RandomGrayscale(p=0.1),
                                NormalizeImage(mean=np.array(STL_FFCV_MEAN),
                                        std=np.array(STL_FFCV_MEAN),type=np.float32)
                                ]
        self.dataset_side_length = 64
        # self.dataset_side_length = 96
        self.dataset_resize_scale = (0.2,1.0)
        self.dataset_resize_ratio = (0.75,4/3)

class STLTransform(nn.Module):
    """
    Generates pair of transformed images, primarily for SSL.
    """
    def __init__(self):
        super().__init__()        
        self.transform = transforms.Compose([
            # transforms.ConvertImageDtype(torch.float32),
            # ReScale(1/255.),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=0.4, contrast=0.4,
                    saturation=0.4, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomResizedCrop(
                64,
                scale=(0.2,1.0),
                ratio=(0.75,(4/3)),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=STL_MEAN,
                                 std=STL_STD)
        ])


    def forward(self, x):
        y1 = self.transform(x)
        y2 = self.transform(x)
        return (y1, y2)
    

class ImagenetTransformFFCV():
    """
    Defines a list of FFCV transforms for SSL on Imagenet
    """
    def __init__(self):
        self.transform_list = [
                                RandomHorizontalFlip(flip_prob=0.5),
                                RandomColorJitter(jitter_prob=0.8,
                                            brightness=0.4, 
                                            contrast=0.4,
                                            saturation=0.4, 
                                            hue=0.1),
                                RandomGrayscale(p=0.2),
                                RandomSolarization(solarization_prob=0.2, 
                                                   threshold=128),
                                NormalizeImage(mean=np.array(IMAGENET_FFCV_MEAN),
                                        std=np.array(IMAGENET_FFCV_MEAN),
                                        type=np.float16),
                                # transforms.GaussianBlur(kernel_size=(5, 9), 
                                #                         sigma=(0.1, 2))
                                ]
        self.dataset_side_length = 128 #224
        self.dataset_resize_scale = (0.08, 1.0) #(0.2,1.0)
        self.dataset_resize_ratio = (0.75,4/3)


class CifarClassifierTransform(nn.Module):
    """
    Generates transformed images, primarily for image classification.
    """
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            # transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=CIFAR_FFCV_MEAN,
                                 std=CIFAR_FFCV_STD)
        ])

    def forward(self, x):
        return self.transform(x)


class STLClassifierTransform(nn.Module):
    """
    Generates transformed images, primarily for image classification.
    """
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            # transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=STL_FFCV_MEAN,
                                 std=STL_FFCV_STD)
        ])

    def forward(self, x):
        return self.transform(x)
    

class ImagenetClassifierTransform():
    """
    Generates transformed images, primarily for image classification.
    """
    def __init__(self):
        self.transform_list = [
            # transforms.ConvertImageDtype(torch.float32),
            NormalizeImage(mean=IMAGENET_FFCV_MEAN, 
                           std=IMAGENET_FFCV_STD, type=np.float16)
        ]


# transforms for pytorch dataloaders
class SSLPT_CIFAR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            # NOTE : ToTensor normalized uint8 to float32 in range [0.0, 1.0]
            #        This is handled for FFCV manually by adding a scaler.
            # transforms.ToTensor(),
            CifarTransform()
        ])
        self.TensorTransform = transforms.ToTensor()
    
    def forward(self, x):
        x = self.TensorTransform(x)
        # return x
        return self.transform(x)

class SSLPT_STL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            # NOTE : ToTensor normalized uint8 to float32 in range [0.0, 1.0]
            #        This is handled for FFCV manually by adding a scaler.
            transforms.ToTensor(),
            STLTransform()
        ])
    
    def forward(self, x):
        return self.transform(x)
