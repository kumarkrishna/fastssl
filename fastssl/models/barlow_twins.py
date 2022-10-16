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
    # x1, x2 = TransformGPU(x1, x2)

    bsz = x1.shape[0]
    # x,_ = inp
    # x1,x2 = ssl_transform(x.cuda(non_blocking=True))
    # bsz = x1.shape[0]
    
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


class BarlowTwins(SSL):
    """
    Modified ResNet50 architecture to work with CIFAR10
    """
    def __init__(self, bkey='resnet50proj', projector_dim=128, dataset='cifar10', hidden_dim=None):
        super(BarlowTwins, self).__init__()
        self.projector_dim = projector_dim
        if hidden_dim is None:
            self.hidden_dim = projector_dim
        else:
            assert hidden_dim==projector_dim, "Implementation only for hidden_dim ({}) = projector_dim ({})".format(hidden_dim,projector_dim)
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
    


class LinearClassifier(nn.Module):
    """
    Linear classifier with a backbone
    """
    def __init__(self,
                 num_classes=10,
                 dataset='cifar10',
                 bkey="resnet50feat",
                 ckpt_path=None, 
                 ckpt_epoch=None,
                 feat_dim=2048):
        super(LinearClassifier, self).__init__()
        # set arguments
        self.bkey = bkey
        self.dataset = dataset
        self.feat_dim = feat_dim

        # define model : backbone(resnet50modified) 
        self.backbone = BackBone(self.bkey, self.dataset, self.feat_dim)

        # load pretrained weights
        self.load_backbone(ckpt_path, requires_grad=False)

        # define linear classifier
        self.fc = nn.Linear(feat_dim, num_classes, bias=True)


        # TODO : support loading pretrained classifier
        # TODO : support finetuning projector
    

    def load_backbone(self, ckpt_path, requires_grad=False):
        # ckpt_path = self.get_ckpt_path(ckpt_path, ckpt_epoch)
        # import pdb; pdb.set_trace()
        if ckpt_path is not None:
            ## map location cpu
            self.load_state_dict(torch.load(ckpt_path, map_location="cpu")['model'], strict=False)
            print("Loaded pretrained weights from {}".format(ckpt_path))
        
        for param in self.backbone.parameters():
            param.requires_grad = requires_grad
        

    def forward(self, x):
        if 'proj' in self.bkey:
            # WE WANT TO FORWARD PROPAGATE THROUGH self.backbone() FIRST??
            feats = self.backbone.proj(x)
        elif 'feat' in self.bkey:
            feats = self.backbone(x)
        feats = torch.flatten(feats, start_dim=1)
        preds = self.fc(feats)
        return preds