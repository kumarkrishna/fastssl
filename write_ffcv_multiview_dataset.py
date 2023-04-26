"""
Usage:
python write_ffcv_multiview_dataset.py
"""

import os
import numpy as np
from tqdm import tqdm
import torchvision
import torch
from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from typing import List
from ffcv.fields import IntField, RGBImageField
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
    RandomHorizontalFlip,
    Cutout,
    RandomTranslate,
    Convert,
    ToDevice,
    ToTensor,
    ToTorchImage,
)
from ffcv.transforms.common import Squeeze


# BASE_DATADIR = "/network/datasets"
# BASE_FFCVDIR = "/network/projects/_groups/linclab_users/ffcv/ffcv_datasets/"

BASE_DATADIR = "/home/arnab39/scratch/krishna/dothework/data"
BASE_FFCVDIR = "/home/arnab39/scratch/krishna/dothework/ffcv"

write_dataset = True
num_views = 20
DOWNLOAD = True

dataset = "cifar10"
dataset_folder = f"{BASE_DATADIR}/{dataset}.var/{dataset}_torchvision/"
ffcv_folder = f"{BASE_FFCVDIR}/{dataset}"


class MultiViewDataset(Dataset):
    def __init__(
        self, dataset, dataset_folder, train=True, download=False, transform=None
    ):
        if dataset == "cifar10":
            self.dataset = torchvision.datasets.CIFAR10(
                root=dataset_folder, train=train, download=download
            )
        elif dataset == "cifar100":
            self.dataset = torchvision.datasets.CIFAR100(
                root=dataset_folder, train=train, download=download
            )

        elif dataset == "stl10":
            self.dataset = torchvision.datasets.STL10(
                root=dataset_folder,
                split="unlabeled",
                download=download,
                transform=None,
            )
        else:
            raise NotImplementedError
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.transform is not None:
            sample = []
            for view_idx in range(num_views):
                sample.append(self.transform(self.dataset[idx][0]))
            return sample
        return [self.dataset[idx][0]] * num_views


if write_dataset:
    ## WRITE TO BETON FILES
    multiview_data = MultiViewDataset(
        dataset=dataset, dataset_folder=dataset_folder, download=DOWNLOAD
    )
    os.makedirs(ffcv_folder, exist_ok=True)

    beton_fpath = os.path.join(
        ffcv_folder, "multiview_{}_train.beton".format(num_views)
    )
    fields = {}
    for view_idx in range(num_views):
        fields["image{}".format(view_idx)] = RGBImageField()

    multiview_writer = DatasetWriter(
        beton_fpath,
        fields,
    )
    multiview_writer.from_indexed_dataset(multiview_data)


## TODO (krishna): Add tests to make sure that the dataset is written correctly
