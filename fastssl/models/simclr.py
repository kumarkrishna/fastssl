import glob
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastssl.models.ssl import SSL
from fastssl.models.backbone import BackBone

def SimCLRLoss(model, inp, temperature=0.05):
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
    
    similarity_scores = (all_z_norm @ all_z_norm.T) / temperature
    eps = 1e-9
    
    ones = torch.ones(bsz)
    mask = (torch.diag(ones, bsz) + torch.diag(ones, -bsz)).cuda(non_blocking=True)
    
    logits = similarity_scores - similarity_scores.max(dim=-1, keepdim=True).detach()
    exp_logits = torch.exp(logits) * (1 - torch.ones(2 * bsz)).cuda(non_blocking=True)
    
    log_likelihood = - logits + torch.log(exp_logits.sum(dim=-1, keepdim=True) + eps)
    
    loss = (log_likelihood * mask).sum()/ mask.sum()

    return loss
