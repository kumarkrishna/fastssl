import torch
import torch.nn as nn
import torch.nn.functional as F

from fastssl.models.ssl import SSL
from fastssl.models.backbone import BackBone
from fastssl.data.cifar_transforms import CifarTransform
from fastssl.data.imagenet_dataloaders import TransformGPU

ssl_transform = CifarTransform()


def get_collapse_avoiding_loss(z, criterion, num_patches):
    z_list = z.chunk(num_patches, dim=0)
    loss = 0
    for i in range(num_patches):
        loss += criterion(z_list[i])
    loss = loss / num_patches
    return loss


class ClusterCosineSimilarity(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        pass

    def forward(self, z_proj_clusters, num_patches=2):
        """
        Compute cluster centers and similarity loss.
        Args:
            z_proj_clusters: list of projected features from different views (NP x B x D)
            num_patches: (int) number of patches (NP)
        """
        z_proj_centers = torch.stack(z_proj_clusters, dim=0).mean(dim=0, keepdim=True)
        z_sim = F.cosine_similarity(z_proj_centers, z_proj_clusters, dim=-1).mean()
        # we want to maximize similarity, therefore negative sign
        return -z_sim


class TotalCodingRate(nn.Module):
    # credits: https://github.com/tsb0601/EMP-SSL/blob/main/main.py
    def __init__(self, eps=0.2):
        super(TotalCodingRate, self).__init__()
        self.eps = eps

    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape  # [d, B]
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.0

    def forward(self, X):
        return -self.compute_discrimn_loss(X.T)


def EMPLoss(model, inp, _lambda=None, patch_sim_scale=200, num_patches=2):
    """
    Implement Extreme Multi-Patch training.
    """
    # build losses
    simloss = ClusterCosineSimilarity()
    tcrloss = TotalCodingRate()

    inp_multiview = torch.cat(inp, dim=0)
    inp_multiview = inp_multiview.cuda(non_blocking=True)

    z_proj_multiview = model(inp_multiview)
    z_proj_clusters = z_proj_multiview.chunk(num_patches, dim=0)

    # compute similarity loss
    similarity_loss = simloss(z_proj_clusters, num_patches=num_patches)

    # compute collapse-avoiding loss
    collapse_avoiding_loss = get_collapse_avoiding_loss(
        z_proj_multiview, tcrloss, num_patches
    )

    loss = patch_sim_scale * similarity_loss + collapse_avoiding_loss
    return loss, {
        "loss": loss.detach(),
        "similarity_loss": similarity_loss.detach(),
        "collapse_avoiding_loss": collapse_avoiding_loss.detach(),
    }


class EMP(SSL):
    """
    Modified ResNet50 architecture to work with CIFAR10
    """

    def __init__(
        self, bkey="resnet50proj", projector_dim=128, dataset="cifar10", hidden_dim=None
    ):
        super(EMP, self).__init__()
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
