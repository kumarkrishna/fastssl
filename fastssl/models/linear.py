import torch
import torch.nn as nn
import torch.nn.functional as F

from fastssl.models.ssl import SSL
from fastssl.models.backbone import BackBone

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
                 feat_dim=2048,
                 proj_hidden_dim=128):
        super(LinearClassifier, self).__init__()
        # set arguments
        self.bkey = bkey
        self.dataset = dataset
        self.feat_dim = feat_dim
        self.proj_hidden_dim = proj_hidden_dim

        # define model : backbone(resnet50modified) 
        self.backbone = BackBone(name = self.bkey, 
                                dataset = self.dataset, 
                                projector_dim = self.feat_dim,
                                hidden_dim = proj_hidden_dim)

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
            x = self.backbone(x)
            x = torch.flatten(x, start_dim=1)
            feats = self.backbone.proj(x)
        elif 'feat' in self.bkey:
            feats = self.backbone(x)
        feats = torch.flatten(feats, start_dim=1)
        preds = self.fc(feats)
        return preds