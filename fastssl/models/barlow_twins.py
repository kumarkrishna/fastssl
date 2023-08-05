import torch
import torch.nn as nn
import torch.nn.functional as F

from fastssl.models.ssl import SSL
from fastssl.models.backbone import BackBone
from fastssl.data.cifar_transforms import CifarTransform
from fastssl.data.imagenet_dataloaders import TransformGPU

ssl_transform = CifarTransform()


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m, "x should be a square matrix but got shape ({} x {})".format(m, n)
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
    inp = list(inp)
    _ = inp.pop(1)
    num_augs = len(inp)
    for x in inp:
        x = x.cuda(non_blocking=True)
    # (x1, x2), _ = inp
    # x1, x2 = x1.cuda(non_blocking=True), x2.cuda(non_blocking=True)
    # x1, x2 = TransformGPU(x1, x2)

    bsz = inp[0].shape[0]
    # bsz = x1.shape[0]
    # x,_ = inp
    # x1,x2 = ssl_transform(x.cuda(non_blocking=True))
    # bsz = x1.shape[0]

    # forward pass
    z_list = [model(x) for x in inp]
    # z1 = model(x1)
    # z2 = model(x2)

    z_norm_list = [(z - z.mean(0)) / z.std(0) for z in z_list]
    # z1_norm = (z1 - z1.mean(0)) / z1.std(0) # NxD
    # z2_norm = (z2 - z2.mean(0)) / z2.std(0) # NxD

    on_diag = 0.0
    off_diag = 0.0
    # compute sum across all patches of each image for each embedding dim
    z_norm_sum = torch.sum(torch.stack(z_norm_list), dim=0)
    # for i in range(num_augs):
    #     # take embedding of one patch
    #     z1_norm = z_norm_list[i]
    #     # take mean embedding of all other patches
    #     z2_norm = (z_norm_sum - z_norm_list[i]) / (num_augs - 1)
    #     # compute BarlowTwins loss for each such pairing
    #     c = torch.mm(z1_norm.T, z2_norm) / bsz  # DxD

    #     # take sum across all patches
    #     on_diag += torch.diagonal(c).add_(-1).pow_(2).sum()
    #     off_diag += off_diagonal(c).pow_(2).sum()
    # # return average across all patches as the final loss
    # loss = (on_diag + _lambda * off_diag) / num_augs

    i = 0
    # take embedding of one patch
    z1_norm = z_norm_list[i]
    # take mean embedding of all other patches
    z2_norm = (z_norm_sum - z_norm_list[i]) / (num_augs - 1)
    # compute BarlowTwins loss for each such pairing
    c = torch.mm(z1_norm.T, z2_norm) / bsz  # DxD

    # take sum across all patches
    on_diag += torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag += off_diagonal(c).pow_(2).sum()
    # return average across all patches as the final loss
    loss = (on_diag + _lambda * off_diag)
    
    return loss


class BarlowTwins(SSL):
    """
    Modified ResNet50 architecture to work with CIFAR10
    """

    def __init__(
        self, bkey="resnet50proj", projector_dim=128, dataset="cifar10", hidden_dim=None
    ):
        super(BarlowTwins, self).__init__()
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
        return F.normalize(projections, dim=-1)

    def _unnormalized_project(self, x):
        feats = self._unnormalized_feats(x)
        projections = self.backbone.proj(feats)
        return projections

    def _unnormalized_feats(self, x):
        x = self.backbone(x)
        feats = torch.flatten(x, start_dim=1)
        return feats
