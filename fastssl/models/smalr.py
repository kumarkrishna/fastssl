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


class SmaLRLoss(nn.Module):
    def __init__(self, ncloss_type="sp", closs_type="mf", hypersphere_radius=1.0, temperature=1.0, num_patches=2, lambd=5e-3, **extra_kwargs):
        super().__init__()
        self.ncloss_type = ncloss_type
        self.closs_type = closs_type
        self.lambd = lambd
        self.temperature = temperature
        self.num_patches = num_patches
        self.hypersphere_radius = hypersphere_radius

        # build the loss functions
        closs_dict = {
            "mf": ClusteringMeanfieldLoss(num_patches=self.num_patches),
            "pw": ClusteringPairwiseLoss(num_patches=self.num_patches),
        }
        ncloss_dict = {
            "sp": SpectralLoss(
                num_patches=self.num_patches, hypersphere_radius=self.hypersphere_radius)
        }
        self.closs_fn = closs_dict[closs_type]
        self.ncloss_fn = ncloss_dict[ncloss_type]
    
    def __call__(self, model, x):
        """
        Peform model forward pass and compute the BarlowTwin loss.

        Args:
            model: a torch.nn.Module
            inp: a torch.Tensor
            _lambda: a float
        Returns:
            loss: scalar tensor
        """
        z = model(x)
        ncloss = self.ncloss_fn(z)
        closs = self.closs_fn(z)
        loss = ncloss + self.lambd * closs
        return {"loss": loss, "ncloss": ncloss, "closs": closs}

class SmaLR(SSL):
    """
    Modified ResNet50 architecture to work with CIFAR10
    """

    def __init__(
        self, bkey="resnet50proj", projector_dim=128, dataset="cifar10", hidden_dim=None, **extra_kwargs
    ):
        super(SmaLR, self).__init__()
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

    def forward(self, xs, is_test=False):
        xs = list(xs)
        _ = xs.pop(1)
        num_augs = len(xs)
        xs = [x.cuda(non_blocking=True) for x in xs]
        xs = torch.cat(xs, dim=0)

        feature = self.backbone.forward_backbone(xs)
        feature = torch.flatten(feature, start_dim=1)
        embedding = self.backbone.forward_projector(feature)

        if is_test:
            return embedding, feature
        else:
            return embedding
        


class ClusteringMeanfieldLoss(nn.Module):
    def __init__(self, num_patches=2):
        super().__init__()
        self.num_patches = num_patches

    def forward(self, z_proj):
        z_list = z_proj.chunk(self.num_patches, dim=0)
        num_patch = len(z_list)
        z_list = torch.stack(list(z_list), dim=0)
        z_avg = z_list.mean(dim=0)
        z_avg_norm = (z_avg - z_avg.mean(0)) / z_avg.std(0)

        loss = 0
        for i in range(num_patch):
            z = z_list[i]
            z_norm = (z - z.mean(0)) / z.std(0)
            loss += torch.mean((z_norm - z_avg_norm) ** 2)

        loss = loss / num_patch

        return loss


class ClusteringPairwiseLoss(nn.Module):
    def __init__(self, num_patches=2):
        super().__init__()
        self.num_patches = num_patches

    def forward(self, z_proj):
        """
        z_list: list of z's from different patches
        """
        loss = 0
        z_list = z_proj.chunk(self.num_patches, dim=0)
        z_stack = torch.stack(list(z_list), dim=0)
        z_norm = (z_stack - z_stack.mean(1, keepdim=True)) / z_stack.std(
            1, keepdim=True
        )

        for i in range(self.num_patches):
            for j in range(self.num_patches):
                if i != j:
                    loss += torch.mean((z_norm[i] - z_norm[j]) ** 2)
        loss = loss / (self.num_patches * (self.num_patches - 1))
        return loss


class SpectralLoss(nn.Module):
    def __init__(self, hypersphere_radius=1.0, num_patches=2):
        super().__init__()
        self.hypersphere_radius = hypersphere_radius

    def forward(self, z):
        bsz = z.shape[0]
        z_norm = (z - z.mean(0)) / z.std(0)  # NxD
        variance_z = torch.mm(z_norm.T, z_norm) / bsz  # DxD
        eigenvals_z = torch.linalg.eigvals(variance_z.to(torch.float32)).real
        return torch.maximum(
            self.hypersphere_radius - eigenvals_z,
            torch.zeros_like(eigenvals_z).to(eigenvals_z.device),
        ).mean()
