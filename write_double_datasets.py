from torch.utils.data import Subset, Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from torchvision.datasets import CIFAR10, ImageFolder, ImageNet

from argparse import ArgumentParser
from fastargs import Section, Param
from fastargs.validation import And, OneOf
from fastargs.decorators import param, section
from fastargs import get_current_config

Section('cfg', 'arguments to give the writer').params(
    dataset=Param(And(str, OneOf(['cifar', 'imagenet'])), 'Which dataset to write', default='imagenet'),
    data_dir=Param(str, 'Where to find the PyTorch dataset', required=True),
    write_path=Param(str, 'Where to write the new dataset', required=True),
    write_mode=Param(str, 'Mode: raw, smart or jpg', required=False, default='smart'),
    max_resolution=Param(int, 'Max image side length', required=True),
    num_workers=Param(int, 'Number of workers to use', default=16),
    chunk_size=Param(int, 'Chunk size for writing', default=100),
    jpeg_quality=Param(float, 'Quality of jpeg images', default=90),
    subset=Param(int, 'How many images to use (-1 for all)', default=-1),
    compress_probability=Param(float, 'compress probability', default=None)
)

class DoubleImageDataset(Dataset):
    def __init__(self,dataset,dataset_folder,train=True,download=False,transform=None):
        if dataset=='cifar':
            self.dataset = CIFAR10(root=dataset_folder,train=train,download=download)
        elif dataset=='imagenet':
            self.dataset = ImageFolder(root=dataset_folder)
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


@section('cfg')
@param('dataset')
@param('data_dir')
@param('write_path')
@param('max_resolution')
@param('num_workers')
@param('chunk_size')
@param('subset')
@param('jpeg_quality')
@param('write_mode')
@param('compress_probability')
def main(dataset, data_dir, write_path, max_resolution, num_workers,
         chunk_size, subset, jpeg_quality, write_mode, compress_probability):
    if dataset == 'cifar':
        # my_dataset = CIFAR10(root=data_dir, train=True, download=True)
        my_dataset = DoubleImageDataset(dataset=dataset, dataset_folder=data_dir)
    elif dataset == 'imagenet':
        # my_dataset = ImageFolder(root=data_dir)
        my_dataset = DoubleImageDataset(dataset=dataset, dataset_folder=data_dir)
    else:
        raise ValueError('Unrecognized dataset', dataset)

    if subset > 0: my_dataset = Subset(my_dataset, range(subset))
    writer = DatasetWriter(write_path, {
        'image1': RGBImageField(write_mode=write_mode,
                                max_resolution=max_resolution,
                                compress_probability=compress_probability,
                                jpeg_quality=jpeg_quality),
        'image2': RGBImageField(write_mode=write_mode,
                                max_resolution=max_resolution,
                                compress_probability=compress_probability,
                                jpeg_quality=jpeg_quality),
    }, num_workers=num_workers)

    writer.from_indexed_dataset(my_dataset, chunksize=chunk_size)


if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()
