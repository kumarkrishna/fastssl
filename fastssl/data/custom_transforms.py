import random
from PIL import Image, ImageOps, ImageFilter
import torch
from torch import nn
import torchvision.transforms as transforms

# mean, std for normalized dataset
CIFAR_MEAN = [0.485, 0.456, 0.406]
CIFAR_STD = [0.229, 0.224, 0.225]
CIFAR_FFCV_MEAN = [125.307, 122.961, 113.8575]
CIFAR_FFCV_STD = [51.5865, 50.847, 51.255]
STL_MEAN = [0.4467, 0.4398, 0.4066]
STL_STD = [0.2242, 0.2215, 0.2239]
STL_FFCV_MEAN = [113.9112, 112.1515, 103.6948]
STL_FFCV_STD = [57.1603, 56.4828, 57.0975]

class ReScale(nn.Module):
    def __init__(self, scale):
        super(ReScale, self).__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

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
                                 std=CIFAR_STD)
        ])


    def forward(self, x):
        y1 = self.transform(x)
        y2 = self.transform(x)
        return (y1, y2)

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


# transforms for pytorch dataloaders
class SSLPT_CIFAR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            # NOTE : ToTensor normalized uint8 to float32 in range [0.0, 1.0]
            #        This is handled for FFCV manually by adding a scaler.
            transforms.ToTensor(),
            CifarTransform()
        ])
    
    def forward(self, x):
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

class GaussianBlur(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            gblur = transforms.GaussianBlur(5, sigma=sigma)
            return gblur(img)
        else:
            return img


class Solarization(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

## transform for datasampler
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
            #Solarization(p=0.0),
            transforms.ConvertImageDtype(torch.float16),
            #transforms.ConvertImageDtype(torch.float32),
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
            #Solarization(p=0.2),
            transforms.ConvertImageDtype(torch.float16),
            #transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2