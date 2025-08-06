import torch
from torch import Tensor

def dice_loss(logits, targets, eps=1e-6):

    p = torch.sigmoid(logits)
    p_flat = p.view(-1)
    t_flat = targets.view(-1)

    intersection = (p_flat * t_flat).sum()
    union = p_flat.sum() + t_flat.sum()

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice