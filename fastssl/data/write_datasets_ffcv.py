from argparse import ArgumentParser
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
import torchvision


def create_dataset_STL10(args):
    """
    Create FFCV compatible dataset for STL10.
    """


def create_dataset_CIFAR(args):
    datadir = args.datadir 
    train_dataset, val_dataset = args.train_dataset, args.val_dataset
    datasets = {
        'train': torchvision.datasets.CIFAR10(datadir, train=True, download=True),
        'test': torchvision.datasets.CIFAR10(datadir, train=False, download=True)
        }

    for (name, ds) in datasets.items():
        path = train_dataset if name == 'train' else val_dataset
        writer = DatasetWriter(path, {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)


if __name__ == "__main__":
    # gather arguments 
    args = get_arguments()

    # create dataset
    create_dataset_CIFAR(args)
