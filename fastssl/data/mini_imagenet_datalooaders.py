import torch
import torchvision
import numpy as np
from imagenet_transforms import ImageNetClassifierTransform, Transform

DEFAULT_CROP_RATIO = 224 / 256
ImageNet_MEAN = [0.485, 0.456, 0.406]  # official ImageNet mean
ImageNet_STD = [0.229, 0.224, 0.225]  # official ImageNet std
ImageNet_FFCV_MEAN = [0.4820, 0.4538, 0.3998]
ImageNet_FFCV_STD = [0.2208, 0.2165, 0.2157]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255



def get_mini_imagenet_dataloaders_simclr(data_dir=None, batch_size=None, num_workers=None):

    loaders = {}

    dataset = torchvision.datasets.ImageFolder(data_dir, Transform())
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42))

    loaders['train'] = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, shuffle=True, drop_last=True
    )
    return loaders


def get_mini_imagenet_dataloaders_eval(data_dir=None, batch_size=None, eval_perc=0.1, num_workers=None):

    loaders = {}

    dataset = torchvision.datasets.ImageFolder(data_dir, ImageNetClassifierTransform())
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [1-eval_perc, eval_perc], generator=torch.Generator().manual_seed(42))

    loaders['train'] = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, shuffle=True, drop_last=True
    )
    loaders['test'] = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, shuffle=True, drop_last=True
    )

    return loaders
