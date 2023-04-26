import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastssl.models.ssl import SSL
from fastssl.models.backbone import BackBone


def BYOLLoss(model, model_target, inp):
    """
    Peform model forward pass and compute the BYOL loss.

    Args:
        model: a torch.nn.Module
        inp: a torch.Tensor
    Returns:
        loss: scalar tensor
    """

    # generate samples from tuple

    (x1, x2), _ = inp
    x1, x2 = x1.cuda(non_blocking=True), x2.cuda(non_blocking=True)
    bsz = x1.shape[0]

    # forward pass
    z1 = model(x1, use_predictor=True)  # NXD
    z2 = model_target(x2).detach()  # NXD

    z1_norm = F.normalize(z1, dim=-1, p=2)
    z2_norm = F.normalize(z2, dim=-1, p=2)

    loss = (2 - 2 * (z1_norm * z2_norm).sum(dim=-1)).mean()

    return loss, {
        "loss": loss.detach(),
    }


def update_state_dict(model, state_dict, tau=1):
    """Update the state dict of ``model`` using the input ``state_dict``, which
    must match format.  ``tau==1`` applies hard update, copying the values, ``0<tau<1``
    applies soft update: ``tau * new + (1 - tau) * old``.
    """
    if tau == 1:
        model.load_state_dict(state_dict)
    elif tau > 0:
        update_sd = {
            k: tau * state_dict[k] + (1 - tau) * v
            for k, v in model.state_dict().items()
        }
        model.load_state_dict(update_sd)


class BYOL(SSL):
    """
    Modified ResNet50 architecture to work with CIFAR10
    """

    def __init__(
        self, bkey="resnet50proj", projector_dim=128, dataset="cifar10", hidden_dim=512
    ):
        super(BYOL, self).__init__()
        self.dataset = dataset
        self.bkey = bkey

        self.backbone = BackBone(
            name=self.bkey,
            dataset=self.dataset,
            projector_dim=projector_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, x, use_predictor=False):
        projections = self._unnormalized_project(x, use_predictor)
        return F.normalize(projections, dim=-1)

    def _unnormalized_project(self, x, use_predictor=False):
        feats = self._unnormalized_feats(x)
        projections = self.backbone.proj(feats)
        if use_predictor:
            projections = self.backbone.pred(projections)
        return projections

    def _unnormalized_feats(self, x):
        x = self.backbone(x)
        feats = torch.flatten(x, start_dim=1)
        return feats
