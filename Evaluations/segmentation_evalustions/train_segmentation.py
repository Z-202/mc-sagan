import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bratsloader import BRATSVolumes
from models.unet import Unet
from project_config import DATA, TRAIN_UNET, ensure_project_dirs


def seed_all(seed: int = 10):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()

@torch.no_grad()
def dice_score(pred_bin: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred_bin.contiguous().view(pred_bin.size(0), -1)
    target = target_bin.contiguous().view(target_bin.size(0), -1)
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return dice.sum()

def _filter_valid_batch(batch: dict):
    miss = batch.get("missing", None)
    if miss is None:
        return batch, None
    if isinstance(miss, (list, tuple)):
        valid_idx = [i for i, m in enumerate(miss) if str(m).lower() == "none"]
        if len(valid_idx) == len(miss):
            return batch, None
        if len(valid_idx) == 0:
            return None, None
        idx = torch.as_tensor(valid_idx, dtype=torch.long)
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v) and v.dim() >= 1 and v.size(0) == len(miss):
                out[k] = v.index_select(0, idx)
            else:
                out[k] = v
        return out, idx
    if str(miss).lower() != "none":
        return None, None
    return batch, None

def main():
    ensure_project_dirs()
    seed_all(TRAIN_UNET["seed"])
    print("******************* train_unet *******************")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = DATA["train_new"]
    model_save = TRAIN_UNET["model_path"]
    os.makedirs(os.path.dirname(model_save), exist_ok=True)
    batch_size = TRAIN_UNET["batch_size"]
    num_workers = TRAIN_UNET["num_workers"]
    lr = TRAIN_UNET["lr"]
    epochs = TRAIN_UNET["epochs"]
    train_data = BRATSVolumes(data_root, mode="train")
    train_loader = data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    unet = Unet(in_dim=4, conv_dim=32, out_dim=1).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)
    print("EPOCH:", epochs)
    print("Dataset cases:", len(train_data))
    print("Steps/epoch:", len(train_loader))
    print("device:", device)
    unet.train()
    for epoch in range(epochs):
        batch_score = 0.0
        num_batch = 0
        t0 = time.time()
        for batch in train_loader:
            batch, _ = _filter_valid_batch(batch)
            if batch is None:
                continue
            t1n = batch["t1n"].to(device, non_blocking=True)
            t1c = batch["t1c"].to(device, non_blocking=True)
            t2w = batch["t2w"].to(device, non_blocking=True)
            t2f = batch["t2f"].to(device, non_blocking=True)
            img = torch.cat([t1n, t1c, t2w, t2f], dim=1)
            label = batch["seg"].to(device, non_blocking=True)
            seg_pred = unet(img.float())
            loss = dice_loss(seg_pred, label.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            seg_cpu = (seg_pred.detach().cpu() >= 0.5).float()
            label_cpu = label.detach().cpu().float()
            batch_score += dice_score(seg_cpu, label_cpu).item()
            num_batch += img.size(0)
        img_min, img_max = img.min().item(), img.max().item()
        lab_min, lab_max = label.min().item(), label.max().item()
        seg_min, seg_max = seg_pred.min().item(), seg_pred.max().item()
        print(f"img   min/max: {img_min:.6f} / {img_max:.6f} | label min/max: {lab_min:.6f} / {lab_max:.6f} | seg   min/max: {seg_min:.6f} / {seg_max:.6f}")
        train_score = batch_score / max(1, num_batch)
        print(f"EPOCH {epoch:03d} : train_score = {train_score:.4f}  | time {(time.time() - t0):.1f}s")
    torch.save(unet.state_dict(), model_save)
    print("Saved:", model_save)

if __name__ == "__main__":
    main()
