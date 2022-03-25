import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

from fastssl.models.ssl import SSL


def infoNCE(nn, p, temperature=0.1):
    """
    Refer to https://github.com/facebookresearch/directclr/blob/main/main.py
    for the original implementation.
    """
    nn = F.normalize(nn, dim=1)
    p = F.normalize(p, dim=1)
    
    logits = nn @ p.T
    logits /= temperature
    
    n = p.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    loss = F.cross_entropy(logits, labels)
    return loss


def DirectCLRLoss(model, inp, lastidx=None):
    """
    Refer to https://arxiv.org/abs/2110.09348

    Peform model forward pass and compute the DirectCLR loss.
    For X be BxD data matrix, augmentations T, T' and encoder E. 
    
    generat embeddings : 
        z = E(T(X))      # BxD
        z' = E(T'(X))    # BxD
    
    define masked representation : 
        z_mask = z[:lastidx]
        z'_mask = z'[:lastidx]

    loss = InfoNCE(z_mask, z'_mask)
    
    Args:
        model: a torch.nn.Module
        inp: a torch.Tensor
        lastidx: a float
    Returns:
        loss: scalar tensor
    """

    # generate samples from tuple 
    (x1, x2), _ = inp
    x1, x2 = x1.cuda(non_blocking=True), x2.cuda(non_blocking=True)
    bsz = x1.shape[0]
    
    # forward pass
    z1 = model(x1) # BxD
    z2 = model(x2) # BxD

    z1_mask = z1[:lastidx] # BxD
    z2_mask = z2[:lastidx] # BxD

    # compute InfoNCE loss jointly on both z1 and z2
    loss = infoNCE(z1_mask, z2_mask) / 2 + infoNCE(z2_mask, z1_mask) / 2

    return loss


class DirectCLR(SSL):
    """
    Modified ResNet50 architecture to work with CIFAR10
    """
    def __init__(self, bkey='resnet50proj', feat_dim=128, dataset='cifar10'):
        super(DirectCLR, self).__init__()
        self.feat_dim = feat_dim
        self.dataset = dataset 
        self.bkey = bkey

        self.backbone = BackBone(
            name=self.bkey, dataset=self.dataset, feat_dim=self.feat_dim)
    
    def forward(self, x):
        return self.unnormalized_feats(x)

    def unnormalized_feats(self, x):
        x = self.backbone(x)
        feats = torch.flatten(x, start_dim=1)
        return feats
