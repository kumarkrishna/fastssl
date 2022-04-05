import torch
import torch.nn as nn

from fastssl.models.backbone import BackBone


class NonContrastiveSSL(nn.Module):
    """
    Non-constrastive SSL model.
    """
    def __init__(self, bkey='resnet50proj', feat_dim=128, dataset='cifar10'):
        super(NonContrastiveSSL, self).__init__()
        self.feat_dim = feat_dim
        self.dataset = dataset 
        self.bkey = bkey

        self.backbone = BackBone(
            name=self.bkey, dataset=self.dataset, feat_dim=self.feat_dim)

    def unnormalized_feats(self, x):
        x = self.backbone(x)
        feats = torch.flatten(x, start_dim=1)
        return feats
    
    def forward(self, x):
        raise NotImplementedError
    
    def load_from_ckpt(self, ckpt_path):
        self.load_state_dict(
            torch.load(ckpt_path, map_locatgit add ion="cpu"), strict=False)
        print("Loaded pretrained weights from {}".format(ckpt_path))


class ConstrastiveSSL(NonContrastiveSSL):
    """
    Contrastive SSL model.
    """
    def __init__(self, bkey='resnet50proj', feat_dim=128, dataset='cifar10'):
        super(ConstrastiveSSL, self).__init__(
            bkey=bkey, feat_dim=feat_dim, dataset=dataset)

    def forward(self, x):
        raise NotImplementedError