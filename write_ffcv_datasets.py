import os
import numpy as np
from tqdm import tqdm
import torchvision
import torch
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


write_dataset = False

dataset = "stl10"

if dataset == "cifar100":
    dataset_folder = "/network/datasets/cifar100.var/cifar100_torchvision/"
    ffcv_folder = "/network/projects/_groups/linclab_users/ffcv/ffcv_datasets/cifar100"
elif dataset == "stl10":
    dataset_folder = "/network/datasets/stl10.var/stl10_torchvision/"
    # ffcv_folder = "/network/projects/_groups/linclab_users/ffcv/ffcv_datasets/stl10"
    ffcv_folder = "/network/scratch/g/ghosharn/ffcv/ffcv_datasets/stl10"

elif dataset == "cifar10":
    dataset_folder = "/network/datasets/cifar10.var/cifar10_torchvision/"
    # ffcv_folder = "/network/projects/_groups/linclab_users/ffcv/ffcv_datasets/cifar10"
    ffcv_folder = "/network/scratch/g/ghosharn/ffcv/ffcv_datasets/cifar10"


if dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(
        root=dataset_folder, train=True, download=False, transform=None
    )
    testset = torchvision.datasets.CIFAR100(
        root=dataset_folder, train=False, download=False, transform=None
    )

elif dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(
        root=dataset_folder, train=True, download=False, transform=None
    )
    testset = torchvision.datasets.CIFAR10(
        root=dataset_folder, train=False, download=False, transform=None
    )

elif dataset == "stl10":
    unlabeledset = torchvision.datasets.STL10(
        root=dataset_folder, split="unlabeled", download=False, transform=None
    )
    trainset = torchvision.datasets.STL10(
        root=dataset_folder, split="train", download=False, transform=None
    )
    testset = torchvision.datasets.STL10(
        root=dataset_folder, split="test", download=False, transform=None
    )

train_beton_fpath = os.path.join(ffcv_folder, "train.beton")
test_beton_fpath = os.path.join(ffcv_folder, "test.beton")

## WRITE TO BETON FILES
if write_dataset:
    datasets = {"train": trainset, "test": testset}
    for name, ds in datasets.items():
        breakpoint()
        path = train_beton_fpath if name == "train" else test_beton_fpath
        writer = DatasetWriter(path, {"image": RGBImageField(), "label": IntField()})
        writer.from_indexed_dataset(ds)
    if dataset=='stl10':
        datasets = {"unlabeled": unlabeledset}
        unlabeled_beton_fpath = os.path.join(ffcv_folder, "unlabeled.beton")
        for name, ds in datasets.items():
            breakpoint()
            path = unlabeled_beton_fpath
            writer = DatasetWriter(path, {"image": RGBImageField(), "label": IntField()})
            writer.from_indexed_dataset(ds)


## VERIFY the WRITTEN DATASET
BATCH_SIZE = 5000
loaders = {}
for name in ["train", "test"]:
    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
    ]  # ToDevice('cuda:0'),
    image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

    image_pipeline.extend(
        [
            ToTensor(),
            # ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            # torchvision.transforms.ConvertImageDtype(torch.float32),
            Convert(torch.float32),
        ]
    )

    loaders[name] = Loader(
        os.path.join(ffcv_folder, "{}.beton".format(name)),
        batch_size=BATCH_SIZE,
        num_workers=1,
        order=OrderOption.SEQUENTIAL,
        # drop_last=(name=='train'),
        drop_last=False,
        pipelines={"image": image_pipeline, "label": label_pipeline},
    )

transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

if dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(
        root=dataset_folder, train=True, download=False, transform=transform_test
    )
    testset = torchvision.datasets.CIFAR100(
        root=dataset_folder, train=False, download=False, transform=transform_test
    )

elif dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(
        root=dataset_folder, train=True, download=False, transform=transform_test
    )
    testset = torchvision.datasets.CIFAR10(
        root=dataset_folder, train=False, download=False, transform=transform_test
    )

elif dataset == "stl10":
    trainset = torchvision.datasets.STL10(
        root=dataset_folder, split="train", download=False, transform=transform_test
    )
    testset = torchvision.datasets.STL10(
        root=dataset_folder, split="test", download=False, transform=transform_test
    )

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
)

X_ffcv, y_ffcv = next(iter(loaders["train"]))
X_tv, y_tv = next(iter(trainloader))
print("FFCV stats:", X_ffcv.shape, X_ffcv.mean(), X_ffcv.min(), X_ffcv.max())
print("torch stats:", X_tv.shape, X_tv.mean(), X_tv.min(), X_tv.max())
print(torch.allclose(X_ffcv / 255.0, X_tv))

breakpoint()

# calculate mean and std of dataset
print("ffcv dataset stats...")
mean = 0.0
std = 0.0
nb_samples = 0.0
for img, _ in tqdm(loaders["train"]):
    batch_samples = img.size(0)
    data = img.view(batch_samples, img.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples
mean /= nb_samples
std /= nb_samples
print("Train Dataset mean", mean)
print("Train Dataset std", std)
mean = 0.0
std = 0.0
nb_samples = 0.0
for img, _ in tqdm(loaders["test"]):
    batch_samples = img.size(0)
    data = img.view(batch_samples, img.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples
mean /= nb_samples
std /= nb_samples
print("Test Dataset mean", mean)
print("Test Dataset std", std)

print("tv dataset stats...")
mean = 0.0
std = 0.0
nb_samples = 0.0
for img, _ in tqdm(trainloader):
    batch_samples = img.size(0)
    data = img.view(batch_samples, img.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples
mean /= nb_samples
std /= nb_samples
print("Train Dataset mean", mean)
print("Train Dataset std", std)
print("Train Dataset mean*255", mean*255)
print("Train Dataset std*255", std*255)
mean = 0.0
std = 0.0
nb_samples = 0.0
for img, _ in tqdm(testloader):
    batch_samples = img.size(0)
    data = img.view(batch_samples, img.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples
mean /= nb_samples
std /= nb_samples
print("Test Dataset mean", mean)
print("Test Dataset std", std)
print("Test Dataset mean*255", mean*255)
print("Test Dataset std*255", std*255)

### ===============================================================================
# STL10 stats
# ffcv dataset stats...
# Train Dataset mean tensor([113.9112, 112.1515, 103.6948])
# Train Dataset std tensor([57.1603, 56.4828, 57.0975])
# Test Dataset mean tensor([114.5820, 112.7223, 104.1996])
# Test Dataset std tensor([57.3148, 56.5328, 57.2032])
# tv dataset stats...
# Train Dataset mean tensor([0.4467, 0.4398, 0.4066])
# Train Dataset std tensor([0.2242, 0.2215, 0.2239])
# Test Dataset mean tensor([0.4472, 0.4396, 0.4050])
# Test Dataset std tensor([0.2249, 0.2217, 0.2237])
### ===============================================================================
