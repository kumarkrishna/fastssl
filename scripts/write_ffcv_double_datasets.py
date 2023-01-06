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
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze

write_dataset = True

dataset = 'imagenet'
if dataset=='cifar100':
    dataset_folder = '/network/datasets/cifar100.var/cifar100_torchvision/'
    ffcv_folder = '/network/projects/_groups/linclab_users/ffcv/ffcv_datasets/cifar100'
elif dataset=='stl10':
    dataset_folder = '/network/datasets/stl10.var/stl10_torchvision/'
    ffcv_folder = '/network/projects/_groups/linclab_users/ffcv/ffcv_datasets/stl10'
elif dataset=='cifar10':
    dataset_folder = '/network/datasets/cifar10.var/cifar10_torchvision/'
    ffcv_folder = '/network/projects/_groups/linclab_users/ffcv/ffcv_datasets/cifar10'
elif dataset=='imagenet':
    dataset_folder = '/network/datasets/imagenet.var/imagenet_torchvision/'
    ffcv_folder = '/network/scratch/l/lindongy/ffcv_datasets/imagenet'


class DoubleImageDataset(Dataset):
    def __init__(self,dataset,dataset_folder,train=True,download=False,transform=None):
        if dataset=='cifar10':
            self.dataset = torchvision.datasets.CIFAR10(root=dataset_folder,
                                                        train=train,
                                                        download=download)
        elif dataset=='cifar100':
            self.dataset = torchvision.datasets.CIFAR100(root=dataset_folder,
                                                         train=train,
                                                         download=download)

        elif dataset=='stl10':
            self.dataset = torchvision.datasets.STL10(root=dataset_folder,
                                                      split='unlabeled',
                                                      download=download,
                                                      transform=None)
        elif dataset=='imagenet':
            self.dataset = torchvision.datasets.ImageNet(root=dataset_folder,
                                                         split='train')
        else:
            raise NotImplementedError

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        if self.transform is not None:
            return self.transform(self.dataset[idx][0]),self.transform(self.dataset[idx][0])
        return self.dataset[idx][0],self.dataset[idx][0]

if write_dataset:
    ## WRITE TO BETON FILES
    double_data = DoubleImageDataset(dataset=dataset,dataset_folder=dataset_folder)

    beton_fpath = os.path.join(ffcv_folder,'doubleImage_train.beton')
    double_writer = DatasetWriter(beton_fpath,
                                  {
                                      'image1': RGBImageField(
                                          write_mode='smart',
                                          max_resolution=500,
                                          compress_probability=0.5,
                                          jpeg_quality=90
                                          ),
                                      'image2': RGBImageField(
                                          write_mode='smart',
                                          max_resolution=500,
                                          compress_probability=0.5,
                                          jpeg_quality=90
                                          ),
                                  })
    double_writer.from_indexed_dataset(double_data, chunksize=100)


## VERIFY the WRITTEN DATASET
BATCH_SIZE = 5000

image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
image_pipeline.extend([ToTensor(),
                       # ToDevice('cuda:0', non_blocking=True),
                       ToTorchImage(),
                       # torchvision.transforms.ConvertImageDtype(torch.float32),
                       Convert(torch.float32),
                       ])

beton_fpath = os.path.join(ffcv_folder,'doubleImage_train.beton')
loader = Loader(beton_fpath,
                batch_size=BATCH_SIZE,
                num_workers=4,
                order=OrderOption.SEQUENTIAL,
                # os_cache=True,
                drop_last=False,
                pipelines={
                    'image1': image_pipeline,
                    'image2': image_pipeline
                })

# X_ffcv1, X_ffcv2 = next(iter(loader))

transform_tv = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
double_data = DoubleImageDataset(dataset=dataset,dataset_folder=dataset_folder,transform=transform_tv)
tvloader = torch.utils.data.DataLoader(
    double_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# X_tv1, X_tv2 = next(iter(tvloader))
# (Pdb) torch.allclose(X_tv1,X_ffcv1/255.)
# True

mean = 0.0
std = 0.0
nb_samples = 0.
for img,img_ in tqdm(loader):
    assert torch.allclose(img,img_), "Something wrong with the double images saved!"
    batch_samples = img.size(0)
    data = img.view(batch_samples,img.size(1),-1)
    mean+= data.mean(2).sum(0)
    std+= data.std(2).sum(0)
    nb_samples += batch_samples
mean /= nb_samples
std /= nb_samples
print("Dataset mean",mean)
print("Dataset std",std)
print("Dataset mean/255",mean/255.)

### CIFAR stats
# Dataset mean tensor([125.3069, 122.9504, 113.8654])
# Dataset std tensor([51.5867, 50.8503, 51.2452])
# Dataset mean/255 tensor([0.4914, 0.4822, 0.4465])


tv_mean = 0.0
tv_std = 0.0
nb_samples = 0.
for img,img_ in tqdm(tvloader):
    assert torch.allclose(img,img_), "Something wrong with the double images saved!"
    batch_samples = img.size(0)
    data = img.view(batch_samples,img.size(1),-1)
    tv_mean+= data.mean(2).sum(0)
    tv_std+= data.std(2).sum(0)
    nb_samples += batch_samples
tv_mean /= nb_samples
tv_std /= nb_samples
print("Torchvision Dataset mean",tv_mean)
print("Dataset std",tv_std)
assert torch.allclose(tv_mean,mean/255.), "Means are different"
assert torch.allclose(tv_std,std/255.), "Stdev are different"

### CIFAR stats
# Torchvision Dataset mean tensor([0.4914, 0.4822, 0.4465])
# Dataset std tensor([0.2023, 0.1994, 0.2010])
# NO ASSERT FAILED

# import matplotlib.pyplot as plt
# img0 = inp[0][0].detach().cpu().numpy().transpose([1,2,0])
# plt.close('all')
# plt.imsave('tmp5.png',img0/255)
# img1 = inp[1][0].detach().cpu().numpy().transpose([1,2,0])
# plt.close('all')
# plt.imsave('tmp6.png',img1/255)
