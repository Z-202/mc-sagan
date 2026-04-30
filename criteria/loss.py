import torch

EPS = 1e-6

def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    b = pred.size(0)
    pred = pred.reshape(b, -1)
    target = target.reshape(b, -1)
    intersection = (pred * target).sum(dim=1)
    dice = (2 * intersection + EPS) / (pred.sum(dim=1) + target.sum(dim=1) + EPS)
    return 1 - dice.mean()

def dice_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1 - dice_loss(pred, target)
