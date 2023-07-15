import torch
import torch.nn as nn
import torch.nn.functional as F

from fastssl.models.ssl import SSL
from fastssl.models.backbone import BackBone


class LinearClassifier(nn.Module):
    """
    Linear classifier with a backbone
    """

    def __init__(
        self,
        num_classes=10,
        dataset="cifar10",
        bkey="resnet50feat",
        ckpt_path=None,
        ckpt_epoch=None,
        feat_dim=2048,
        proj_hidden_dim=128,
        base_width=64,
    ):
        super(LinearClassifier, self).__init__()
        # set arguments
        self.bkey = bkey
        self.dataset = dataset
        self.feat_dim = feat_dim
        self.proj_hidden_dim = proj_hidden_dim

        if len(self.bkey) > 0:
            # define model : backbone(resnet50modified)
            self.backbone = BackBone(
                name=self.bkey,
                base_width=base_width,
                dataset=self.dataset,
                projector_dim=self.feat_dim,
                hidden_dim=proj_hidden_dim,
            )

            # load pretrained weights
            self.load_backbone(ckpt_path, requires_grad=False)
        else:
            # not using any backbone
            self.backbone = nn.Identity()

        # define linear classifier
        self.fc = nn.Linear(feat_dim, num_classes, bias=True)

        # TODO : support loading pretrained classifier
        # TODO : support finetuning projector

    def load_backbone(self, ckpt_path, requires_grad=False):
        # ckpt_path = self.get_ckpt_path(ckpt_path, ckpt_epoch)
        # import pdb; pdb.set_trace()
        if ckpt_path is not None:
            ## map location cpu
            self.load_state_dict(
                torch.load(ckpt_path, map_location="cpu")["model"], strict=False
            )
            print("Loaded pretrained weights from {}".format(ckpt_path))

        for param in self.backbone.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        if isinstance(x, list):
            # x is a list of augs -- need to average feats over augs
            if self.backbone is not None:
                feats_augs = [self.backbone(x_i) for x_i in x]
            else:
                feats_augs = [x_i for x_i in x]
            if "proj" in self.bkey:
                feats_augs = [torch.flatten(x_i, start_dim=1) for x_i in feats_augs]
                feats_augs = [self.backbone.proj(x_i) for x_i in feats_augs]
            # mean of features across different augmentations of each image
            feats = torch.mean(torch.stack(feats_augs), dim=0).cuda()
            feats = torch.flatten(feats, start_dim=1)
        else:
            # x is just one input
            if self.backbone is not None:
                x = self.backbone(x)
            if "proj" in self.bkey:
                x = torch.flatten(x, start_dim=1)
                x = self.backbone.proj(x)
            feats = torch.flatten(x, start_dim=1)
        preds = self.fc(feats)
        return preds
