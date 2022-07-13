import torch.nn as nn
from torchvision.models.resnet import resnet50


class BackBone(nn.Module):
    def __init__(self,
                 name='resnet50feat',
                 dataset='cifar10', 
                 projector_dim=128):
        super(BackBone, self).__init__()
        self.name = name
        self.build_backbone(dataset, projector_dim)

    def build_backbone(self, dataset='cifar10', projector_dim=128):
        """
        Build backbone model.
        """
        self._resnet50mod(dataset)
        if self.name == 'resnet50proj':
            self.build_projector(projector_dim=projector_dim)

    def _resnet50mod(self, dataset):
        backbone = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # check validity for adding layer to module
            if self.is_valid_layer(module, dataset):
                backbone.append(module)
        self.feats = nn.Sequential(*backbone)
    
    def build_projector(self, projector_dim):
        projector = [
            nn.Linear(2048, projector_dim, bias=False),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_dim, projector_dim, bias=True),
        ]
        self.proj = nn.Sequential(*projector)

    def is_valid_layer(self, module, dataset):
        """
        Check if a layer is valid for the dataset.
        """
        if 'cifar' in dataset:
            return self._check_valid_layer_cifar10(module)
        elif 'stl' in dataset:
            return self._check_valid_layer_stl10(module)
        elif 'imagenet' in dataset:
            return self._check_valid_layer_imagenet(module)
        else:
            raise NotImplementedError
    
    def _check_valid_layer_cifar10(self, module):
        if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
           return True
        return False

    def _check_valid_layer_stl10(self, module):
        if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
           return True
        return False

    def _check_valid_layer_imagenet(self, module):
        if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
           return True
        return False
    
    def forward(self, x):
        return self.feats(x)
