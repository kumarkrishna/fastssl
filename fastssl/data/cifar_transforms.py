import torch
from torch import nn, optim
import torchvision.transforms as transforms


CIFAR_MEAN = [0.485, 0.456, 0.406]
CIFAR_STD = [0.229, 0.224, 0.225]

## transform for datasampler 
class CifarTransform(nn.Module):
    def __init__(self):
        super().__init__()
        # import pdb; pdb.set_trace()
        # self.to_pil = transforms.ToPILImage()
        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float16),
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
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


class CifarClassifierTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float16),
            transforms.Normalize(mean=CIFAR_MEAN,
                                 std=CIFAR_STD)
        ])

    def forward(self, x):
        return self.transform(x)