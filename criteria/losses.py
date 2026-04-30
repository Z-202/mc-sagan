import os
from typing import Optional, Sequence, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

def to01(x: torch.Tensor) -> torch.Tensor:
    x_min = float(x.amin().detach())
    x_max = float(x.amax().detach())
    if x_min >= -1e-3 and x_max <= 1.0 + 1e-3:
        return x.clamp(0.0, 1.0)
    return (x.clamp(-1.0, 1.0) + 1.0) * 0.5

def _dilate_mask(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    if k <= 1:
        return mask
    pad = k // 2
    return F.max_pool3d(mask, kernel_size=k, stride=1, padding=pad)

def _masked_stats_from_ref(ref_det: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    mask = mask.float()
    denom = mask.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)
    mean = (ref_det * mask).sum(dim=(2, 3, 4), keepdim=True) / denom
    var  = ((ref_det - mean) ** 2 * mask).sum(dim=(2, 3, 4), keepdim=True) / denom
    std  = torch.sqrt(var + eps).clamp_min(eps)
    return mean, std

def _apply_stats(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x - mean) / (std + eps)

class _MedicalNet3DFeatures(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        for p in self.parameters():
            p.requires_grad_(False)

        self.expected_in_channels = int(getattr(backbone.conv1, "in_channels", 1))
        self.eval()

    def train(self, mode: bool = True):

        super().train(False)
        return self

    def _match_in_channels(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        if c == self.expected_in_channels:
            return x
        if self.expected_in_channels == 3 and c == 1:
            return x.repeat(1, 3, 1, 1, 1)
        if self.expected_in_channels == 1 and c == 3:
            return x.mean(dim=1, keepdim=True)
        if c < self.expected_in_channels:
            reps = (self.expected_in_channels + c - 1) // c
            return x.repeat(1, reps, 1, 1, 1)[:, : self.expected_in_channels]
        return x[:, : self.expected_in_channels]

    def forward(self, x: torch.Tensor):
        x = self._match_in_channels(x)
        feats = {}
        x = self.stem(x)
        x = self.layer1(x); feats["layer1"] = x
        x = self.layer2(x); feats["layer2"] = x
        x = self.layer3(x); feats["layer3"] = x
        x = self.layer4(x); feats["layer4"] = x
        return feats

class PerceptualLoss(nn.Module):
    def __init__(
        self,
        use_3d: bool = True,
        hub_repo: Optional[str] = None,
        hub_model: str = "medicalnet_resnet50",
        layers: Sequence[str] = ("layer1", "layer2", "layer3", "layer4"),
        layer_weights: Sequence[float] = (1.0, 0.5, 0.25, 0.1),
        mask_thresh01: float = 0.02,
        mask_dilate_k: int = 5,
        eps: float = 1e-6,
        feat_l2norm: bool = False,
        criterion: Literal["l1", "mse"] = "l1",
        feature_mask: bool = True,
    ):
        super().__init__()
        self.layers = tuple(layers)
        self.layer_weights = tuple(layer_weights)
        assert len(self.layers) == len(self.layer_weights)

        self.mask_thresh01 = float(mask_thresh01)
        self.mask_dilate_k = int(mask_dilate_k)
        self.eps = float(eps)
        self.feat_l2norm = bool(feat_l2norm)
        self.feature_mask = bool(feature_mask)
        self.criterion = criterion

        if use_3d:
            repo = hub_repo or os.environ.get("MEDNET_HUB_REPO", "warvito/MedicalNet-models")
            torch.hub._validate_not_a_forked_repo = lambda *a, **k: True
            backbone = torch.hub.load(repo, model=hub_model, verbose=False)
            self.backbone = _MedicalNet3DFeatures(backbone)
        else:
            raise ValueError("This implementation is for 3D MedicalNet only (use_3d=True).")

        self.eval()

    def train(self, mode: bool = True):

        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def _build_mask01(self, ref: torch.Tensor) -> torch.Tensor:
        ref01 = to01(ref)
        mask = (ref01 > self.mask_thresh01).float()
        if self.mask_dilate_k > 1:
            mask = _dilate_mask(mask, k=self.mask_dilate_k)
        return mask

    def _norm_feats(self, f: torch.Tensor) -> torch.Tensor:
        if not self.feat_l2norm:
            return f
        return f / (f.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps)

    def _feat_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.criterion == "mse":
            return (a - b).pow(2)
        return (a - b).abs()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask_from: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert pred.shape == target.shape and pred.dim() == 5, "Expect (B,C,D,H,W) and matching shapes"
        pred = pred.float()
        target = target.float()

        ref = mask_from if mask_from is not None else target
        mask = self._build_mask01(ref).detach()

        mean, std = _masked_stats_from_ref(target.detach(), mask, eps=self.eps)

        pred_n = _apply_stats(pred, mean, std, eps=self.eps)
        targ_n = _apply_stats(target, mean, std, eps=self.eps)

        feats_p = self.backbone(pred_n)

        with torch.no_grad():
            feats_t = self.backbone(targ_n.detach())

        loss = pred.new_tensor(0.0)
        for name, w in zip(self.layers, self.layer_weights):
            fp = self._norm_feats(feats_p[name])
            ft = self._norm_feats(feats_t[name])

            diff = self._feat_distance(fp, ft)

            if self.feature_mask:

                m = F.interpolate(mask, size=diff.shape[2:], mode="nearest")
                diff = diff * m

                denom = (m.sum() * diff.shape[1]).clamp_min(1.0)
                layer_loss = diff.sum() / denom
            else:
                layer_loss = diff.mean()

            loss = loss + float(w) * layer_loss

        return loss
