import torch
import torch.nn as nn
import torch.nn.functional as F

from fastssl.models.ssl import SSL
from fastssl.models.backbone import BackBone
from fastssl.data.cifar_transforms import CifarTransform
from fastssl.data.imagenet_dataloaders import TransformGPU

ssl_transform = CifarTransform()


def SpectralRegLoss(model, inp, hypersphere_radius=0.3, spectral_loss_weight=None, version=None):
    """
    Peform model forward pass and compute the BarlowTwin loss.

    Args:
        model: a torch.nn.Module
        inp: a torch.Tensor
        spectral_loss_weight: a float
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

    if version == '1':       
        z2_norm = z2_norm.detach()

        variance = torch.mm(z1_norm.T, z1_norm) / bsz # DxD
        eigenvals = torch.linalg.eigvals(variance.to(torch.float32)).real
        spectral_loss = torch.maximum(
            hypersphere_radius - eigenvals, 
            torch.zeros_like(eigenvals).to(eigenvals.device)
        ).mean()
        
        invariance_loss = torch.mean((z1_norm - z2_norm) ** 2)

    elif version == '2':
        variance_z1 = torch.mm(z1_norm.T, z1_norm) / bsz # DxD
        eigenvals_z1 = torch.linalg.eigvals(variance_z1.to(torch.float32)).real

        variance_z2 = torch.mm(z2_norm.T, z2_norm) / bsz # DxD
        eigenvals_z2 = torch.linalg.eigvals(variance_z2.to(torch.float32)).real

        spectral_loss = torch.maximum(
            hypersphere_radius - eigenvals_z1, 
            torch.zeros_like(eigenvals_z1).to(eigenvals_z1.device)
        ).mean() + torch.maximum(
            hypersphere_radius - eigenvals_z2, 
            torch.zeros_like(eigenvals_z2).to(eigenvals_z2.device)
        ).mean()
        
        invariance_loss = torch.mean((z1_norm - z2_norm) ** 2)

    elif version == '3':
        z2_norm = z2_norm.detach()

        variance = torch.mm(z1_norm.T, z1_norm) / bsz # DxD
        eigenvals = torch.linalg.eigvals(variance.to(torch.float32)).real
        spectral_loss = torch.maximum(
            hypersphere_radius - eigenvals, 
            torch.zeros_like(eigenvals).to(eigenvals.device)
        ).mean()
        
        invariance_loss = ((1 - (z1_norm * z2_norm).sum(dim=-1))**2).mean()

    elif version == '4':
        z2_norm = z2_norm.detach()

        variance = torch.mm(z1_norm.T, z1_norm) / bsz # DxD
        eigenvals = torch.linalg.eigvals(variance.to(torch.float32)).real
        spectral_loss = torch.maximum(
            hypersphere_radius - eigenvals, 
            torch.zeros_like(eigenvals).to(eigenvals.device)
        ).mean()
        
        invariance_loss = (2 - 2 * (F.normalize(z1_norm, dim=-1, p=2) * F.normalize(z2_norm, dim=-1, p=2)).sum(dim=-1)).mean()

    elif version == '5':
        variance_z1 = torch.mm(z1_norm.T, z1_norm) / bsz # DxD
        eigenvals_z1 = torch.linalg.eigvals(variance_z1.to(torch.float32)).real

        variance_z2 = torch.mm(z2_norm.T, z2_norm) / bsz # DxD
        eigenvals_z2 = torch.linalg.eigvals(variance_z2.to(torch.float32)).real

        spectral_loss = torch.maximum(
            hypersphere_radius - eigenvals_z1, 
            torch.zeros_like(eigenvals_z1).to(eigenvals_z1.device)
        ).mean() + torch.maximum(
            hypersphere_radius - eigenvals_z2, 
            torch.zeros_like(eigenvals_z2).to(eigenvals_z2.device)
        ).mean()
        
        invariance_loss = (2 - 2 * (F.normalize(z1_norm, dim=-1, p=2) * F.normalize(z2_norm, dim=-1, p=2)).sum(dim=-1)).mean()
        

    loss = invariance_loss + spectral_loss_weight * spectral_loss
    return loss


class SpectralReg(SSL):
    """
    Modified ResNet50 architecture to work with CIFAR10
    """
    def __init__(self, bkey='resnet50proj', projector_dim=128, dataset='cifar10', hidden_dim=None):
        super(SpectralReg, self).__init__()
        self.projector_dim = projector_dim
        self.hidden_dim = hidden_dim
        self.dataset = dataset 
        self.bkey = bkey
        print(f'Network defined with projector dim {projector_dim} and hidden dim {hidden_dim}')
        self.backbone = BackBone(
            name=self.bkey, dataset=self.dataset,
            projector_dim=projector_dim, hidden_dim=hidden_dim
        )
    
    def forward(self, x):
        projections = self._unnormalized_project(x)
        return F.normalize(projections, dim=-1)

    def _unnormalized_project(self, x):
        feats = self._unnormalized_feats(x)
        projections = self.backbone.proj(feats)
        return projections

    def _unnormalized_feats(self, x):
        x = self.backbone(x)
        feats = torch.flatten(x, start_dim=1)
        return feats