import torch
import torch.nn as nn
import torch.nn.functional as F

from fastssl.models.ssl import SSL
from fastssl.models.backbone import BackBone
from fastssl.fastssl.data.misc_transforms import CifarTransform
from fastssl.data.imagenet_dataloaders import TransformGPU
from tqdm import tqdm

ssl_transform = CifarTransform()


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m, "x should be a square matrix but got shape ({} x {})".format(m, n)
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def VICRegLoss(model, inp, _lambda=None, _mu=None):
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
    inp = list(inp)
    _ = inp.pop(1)
    num_augs = len(inp)
    for x in inp:
        x = x.cuda(non_blocking=True)

    bsz = inp[0].shape[0]

    # forward pass
    z_list = model(torch.vstack(inp)).chunk(num_augs)

    all_z = torch.stack(z_list)
    z_center = all_z - all_z.mean(1, keepdims=True) # A x NxD

    repr_loss = 0.0
    std_loss = 0.0
    cov_loss = 0.0
    # compute sum across all patches of each image for each embedding dim
    z_sum = torch.sum(all_z, dim=0)
    z_center_sum = torch.sum(z_center, dim=0)
    
    # mulitpatch v1
    if num_augs == 2:
        # avoiding redundant computation of loss
        for i in range(num_augs-1):
            # take embedding of one patch
            z1 = z_list[i]
            z1_center = z_center[i]
            # take mean embedding of all other patches
            z2 = (z_sum - z_list[i]) / (num_augs - 1)

            z2_center = (z_center_sum - z_center[i]) / (num_augs - 1)
            
            # compute VICReg loss for each such pairing
            # take sum across all patches
            repr_loss += F.mse_loss(z1, z2)

            std_z1 = z1.std(0) #torch.sqrt(z1_center.var(dim=0) + 0.0001)
            std_z2 = z2.std(0) #torch.sqrt(z2_center.var(dim=0) + 0.0001)
            # take sum across all patches
            std_loss += torch.mean(F.relu(1 - std_z1)) / 2 + \
                torch.mean(F.relu(1 - std_z2)) / 2
            
            # cov_z1 = (z1.T @ z1) / (bsz - 1)
            # cov_z2 = (z2.T @ z2) / (bsz - 1)
            cov_z1 = torch.einsum("ni,nj->ij", z1_center, z1_center) / (bsz - 1)
            cov_z2 = torch.einsum("ni,nj->ij", z2_center, z2_center) / (bsz - 1)
            # take sum across all patches
            cov_loss += off_diagonal(cov_z1).pow_(2).sum().div(z1.shape[1]) + \
                off_diagonal(cov_z2).pow_(2).sum().div(z2.shape[1])
        # return average across all patches as the final loss
        # tqdm.write(f'{repr_loss.data.item():.4f}, {std_loss.data.item():.4f}, {cov_loss.data.item():.4f}')
        loss = (_lambda * repr_loss + _mu * std_loss + cov_loss)# / num_augs

    else:
        for i in range(num_augs):
            # take embedding of one patch
            z1 = z_list[i]
            z1_center = z_center[i]
            # take mean embedding of all other patches
            z2 = (z_sum - z_list[i]) / (num_augs - 1)

            z2_center = (z_center_sum - z_center[i]) / (num_augs - 1)
            
            # compute VICReg loss for each such pairing
            # take sum across all patches
            repr_loss += F.mse_loss(z1, z2)

            std_z1 = z1.std(0) #torch.sqrt(z1_center.var(dim=0) + 0.0001)
            std_z2 = z2.std(0) #torch.sqrt(z2_center.var(dim=0) + 0.0001)
            # take sum across all patches
            std_loss += torch.mean(F.relu(1 - std_z1)) / 2 + \
                torch.mean(F.relu(1 - std_z2)) / 2
            
            # cov_z1 = (z1.T @ z1) / (bsz - 1)
            # cov_z2 = (z2.T @ z2) / (bsz - 1)
            cov_z1 = torch.einsum("ni,nj->ij", z1_center, z1_center) / (bsz - 1)
            cov_z2 = torch.einsum("ni,nj->ij", z2_center, z2_center) / (bsz - 1)
            # take sum across all patches
            cov_loss += off_diagonal(cov_z1).pow_(2).sum().div(z1.shape[1]) + \
                off_diagonal(cov_z2).pow_(2).sum().div(z2.shape[1])
        # return average across all patches as the final loss
        # tqdm.write(f'{repr_loss.data.item():.4f}, {std_loss.data.item():.4f}, {cov_loss.data.item():.4f}')
        loss = (_lambda * repr_loss + _mu * std_loss + cov_loss)/ num_augs
    
    return loss


class VICReg(SSL):
    """
    Modified ResNet50 architecture to work with CIFAR10
    """

    def __init__(
        self, bkey="resnet50proj", projector_dim=128, dataset="cifar10", hidden_dim=None
    ):
        super(VICReg, self).__init__()
        self.projector_dim = projector_dim
        if hidden_dim is None:
            self.hidden_dim = projector_dim
        else:
            assert (
                hidden_dim == projector_dim
            ), "Implementation only for hidden_dim ({}) = projector_dim ({})".format(
                hidden_dim, projector_dim
            )
            self.hidden_dim = hidden_dim
        self.dataset = dataset
        self.bkey = bkey
        print(
            f"Network defined with projector dim {projector_dim} and hidden dim {hidden_dim}"
        )
        self.backbone = BackBone(
            name=self.bkey,
            dataset=self.dataset,
            projector_dim=projector_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, x):
        projections = self._unnormalized_project(x)
        return projections

    def _unnormalized_project(self, x):
        feats = self._unnormalized_feats(x)
        projections = self.backbone.proj(feats)
        return projections

    def _unnormalized_feats(self, x):
        x = self.backbone(x)
        feats = torch.flatten(x, start_dim=1)
        return feats
