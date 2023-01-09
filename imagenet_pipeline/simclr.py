import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import BackBone

# class NonContrastiveSSL(nn.Module):
class SSL(nn.Module):
    """
    Non-constrastive SSL model.
    """
    def __init__(self):
        # super(NonContrastiveSSL, self).__init__()
        super(SSL, self).__init__()
    
    def forward(self, x):
        raise NotImplementedError
    
    def load_from_ckpt(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        print("Loaded pretrained weights from {}".format(ckpt_path))
    
    def get_epoch(path):
        return map(int, path.split('_')[-1].split('.')[0])



def SimCLRLoss(model, inp, _temperature=0.05):
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
    z1 = model(x1)   #NXD
    z2 = model(x2)   #NXD

    z1_norm = F.normalize(z1, dim=-1) 
    z2_norm = F.normalize(z2, dim=-1)
    
    all_z_norm = torch.cat([z1_norm, z2_norm], dim=0)
    
    similarity_scores = (all_z_norm @ all_z_norm.T) / _temperature
    eps = 1e-9
    
    ones = torch.ones(bsz)
    mask = (torch.diag(ones, bsz) + torch.diag(ones, -bsz)).cuda(non_blocking=True)
    
    # subtract max value for stability
    logits = similarity_scores - similarity_scores.max(dim=-1, keepdim=True)[0].detach()
    # remove the diagonal entries of all 1./_temperature because they are cosine
    # similarity of one image to itself.
    exp_logits = torch.exp(logits) * (1 - torch.eye(2 * bsz)).cuda(non_blocking=True)
    
    log_likelihood = - logits + torch.log(exp_logits.sum(dim=-1, keepdim=True) + eps)
    
    loss = (log_likelihood * mask).sum()/ mask.sum()

    return loss

class SimCLR(SSL):
    """
    Modified ResNet50 architecture to work with CIFAR10
    """
    def __init__(self, bkey='resnet50proj', projector_dim=128, dataset='cifar10', hidden_dim=512):
        super(SimCLR, self).__init__()
        self.dataset = dataset 
        self.bkey = bkey

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
