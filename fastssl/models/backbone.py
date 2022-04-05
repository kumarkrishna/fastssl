import torch.nn as nn
from torchvision.models.resnet import resnet50


class BackBone(nn.Module):
    def __init__(self,
                 name='resnet50feat',
                 dataset='cifar10', 
                 bottleneck=None,
                 feat_dim=2048,
                 projector_dim=128):
        super(BackBone, self).__init__()
        self.name = name
        self.build_backbone(
            dataset, bottleneck, feat_dim, projector_dim)

    def build_backbone(self,
                       dataset='cifar10',
                       bottleneck=None,
                       feat_dim=2048,
                       projector_dim=128):
        """
        Build backbone model.
        """
        # build the basic backbone
        self._resnet50modify(dataset)

        # build the projector
        if self.name == 'resnet50proj':
            self.build_projector(
                bottleneck=bottleneck,
                feat_dim=feat_dim,
                projector_dim=projector_dim)

    def _resnet50modify(self, dataset):
        backbone = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # check validity for adding layer to module
            if self.is_valid_layer(module, dataset):
                backbone.append(module)
        self.feats = nn.Sequential(*backbone)
    
    def build_projector(self, feat_dim, projector_dim, bottleneck=None):
        if bottleneck is None:
            bottleneck = projector_dim

        projector = [
            nn.Linear(feat_dim, bottleneck, bias=False),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, projector_dim, bias=True),
        ]

        # build the projector
        self.proj = nn.Sequential(*projector)

    def is_valid_layer(self, module, dataset):
        """
        Check if a layer is valid for the dataset.
        """
        if dataset == 'cifar10':
            return self._check_valid_layer_cifar10(module)
        else:
            raise NotImplementedError
    
    def _check_valid_layer_cifar10(self, module):
        if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
           return True
        return False
    
    def forward(self, x):
        return self.feats(x)
    
    def features(self, x):
        return self.feats(x)

    def project(self, x):
        feats = self.feats(x)
        return self.proj(feats)
