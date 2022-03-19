import glob
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m, "x should be a square matrix but got shape ({} x {})".format(m,n)
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def BarlowTwinLoss(model, inp, _lambda=None):
    """
    Peform model forward pass and compute the BarlowTwin loss.

    Args:
        model: a torch.nn.Module
        inp: a torch.Tensor
        _lambda: a float
    Returns:
        loss: scalar tensor
    """

    # generate samples from tuple 
    (x1, x2), _ = inp
    x1, x2 = x1.cuda(non_blocking=True), x2.cuda(non_blocking=True)
    bsz = x1.shape[0]

    
    # forward pass
    z1 = model(x1)
    z2 = model(x2)

    z1_norm = (z1 - z1.mean(0)) / z1.std(0) # NxD
    z2_norm = (z2 - z2.mean(0)) / z2.std(0) # NxD
    
    c = torch.mm(z1_norm.T, z2_norm) / bsz # DxD

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()

    loss = on_diag + _lambda * off_diag
    return loss


class ResNet50Modified(nn.Module):
    """
    Modified ResNet50 architecture to work with CIFAR10
    """
    def __init__(self, projector_dim=128, dataset='cifar10'):
        super(ResNet50Modified, self).__init__()
        self.projector_dim = projector_dim
        self.dataset = dataset 

        ## add encoder 
        self.build_encoder()

        ## add projection head
        self.build_projector()

    def build_encoder(self):
        """
        Build the encoder part of the network
        """
        ## modify conv1, remove (linear, maxpool2d)
        backbone = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            
            ## check validity for adding layer to module
            if self.is_valid_layer(module, self.dataset):
                backbone.append(module)

        ## add encoder 
        self.backbone = nn.Sequential(*backbone)


    def build_projector(self):
        """
        Build the projector part of the network
        """
        # projector_dim = list(map(int, self.projector_arch.split('-')))
        
        ## for now assume that we have two projector layers
        self.projector = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.projector_dim, bias=True),
        )

    def is_valid_layer(self, module, dataset):
        """
        Check if the layer is valid for adding to the network
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.MaxPool2d):
            return False
        if dataset == 'cifar10':
            if isinstance(module, nn.BatchNorm2d):
                return False
        elif dataset == 'tiny_imagenet' or dataset == 'stl10':
            if isinstance(module, nn.BatchNorm1d):
                return False
        return True

    def is_valid_layer(self, module, dataset):
        if dataset == 'cifar10':
            return self._check_valid_layer_cifar10(module)

    def _check_valid_layer_cifar10(self, module):
        return not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d)

    def forward(self, x):
        """
        Args:
            x: input image
        Returns:
            features: feature vector
            projections: normalized projection vector
        """
        x = self.backbone(x)
        features = torch.flatten(x, start_dim=1)
        projections = self.projector(features)
        return F.normalize(projections, dim=-1)

    def feats(self, x):
        """
        Args:
            x: input image
        Returns:
            features: feature vector
        """
        x = self.backbone(x)
        features = torch.flatten(x, start_dim=1)
        return F.normalize(features, dim=-1)


class LinearClassifier(nn.Module):
    """
    Linear classifier with a backbone
    """
    def __init__(self,
                 num_classes=10, dataset='cifar10',
                 bkey="resnet50M",
                 pretrained_path=None, 
                 ckpt_epoch=None, feat_dim=2048):
        super(LinearClassifier, self).__init__()
        # set arguments
        self.bkey = bkey
        self.dataset = dataset

        # define model : backbone(resnet50modified) 
        self.build_backbone()

        # define linear classifier
        self.fc = nn.Linear(feat_dim, num_classes, bias=True)

        # load pretrained weights
        # TODO : support loading pretrained classifier
        self.load_from_ckpt(pretrained_path, ckpt_epoch=ckpt_epoch, classifer=False)
        
        for param in self.backbone.parameters():
            param.requires_grad = False

    def build_backbone(self):
        if self.bkey == 'resnet50M':
            self.backbone = ResNet50Modified(dataset=self.dataset).backbone

    def load_from_ckpt(self, pretrained_path, ckpt_epoch=None, classifer=True):
        ckpt_path = self.get_ckpt_path(pretrained_path, ckpt_epoch)
        if ckpt_path is not None:
            ## map location cpu
            self.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
            print("Loaded pretrained weights from {}".format(ckpt_path))
        

    def get_ckpt_path(self, pretrained_path, ckpt_epoch=None):
        if pretrained_path is None:
            return None
        fnames = glob.glob(os.path.join(pretrained_path, '*.pth'))
        if len(fnames) == 0:
            return None
        # fnames.sort()
        fnames.sort(key=lambda x:int(x.split('_')[-1].split('.')[0]))
        if ckpt_epoch is None:
            # load the last checkpoint
            ckpt_path = fnames[-1]
        else:
            # load the specified checkpoint_idx
            ckpt_path = fnames[ckpt_epoch]
        return ckpt_path


    def forward(self, x):
        feats = self.backbone(x)
        feats = torch.flatten(feats, start_dim=1)
        preds = self.fc(feats)
        return preds