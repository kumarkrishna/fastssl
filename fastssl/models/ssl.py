import torch
import torch.nn as nn
from fastssl.models.backbone import Backbone


class NonContrastiveSSL(nn.Module):
    """
    Non-constrastive SSL model.
    """
    def __init__(self):
        super(NonContrastiveSSL, self).__init__()
    
    def forward(self, x):
        raise NotImplementedError
    
    def load_from_ckpt(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        print("Loaded pretrained weights from {}".format(ckpt_path))
    
    def get_epoch(path):
        return map(int, path.split('_')[-1].split('.')[0])
