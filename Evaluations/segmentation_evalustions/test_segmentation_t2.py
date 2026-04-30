import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.utils.data as data

from datasets.bratsloader import BRATSVolumes
from models.unet_t2 import Unet
from project_config import DATA, TEST_UNET_T2

torch.backends.cudnn.benchmark = True
torch.manual_seed(TEST_UNET_T2["seed"])
torch.cuda.manual_seed_all(TEST_UNET_T2["seed"])

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = DATA["gan_test"]
    model_path = TEST_UNET_T2["model_path"]
    batch_size = TEST_UNET_T2["batch_size"]
    num_workers = TEST_UNET_T2["num_workers"]
    test_data = BRATSVolumes(data_root, mode="test")
    test_loader = data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    unet = Unet(in_dim=1, conv_dim=32, out_dim=1).to(device)
    unet.load_state_dict(torch.load(model_path, map_location=device))
    unet.eval()
    total_dice_sum = 0.0
    total_cases = 0
    with torch.no_grad():
        for batch in test_loader:
            batch, _ = _filter_valid_batch(batch)
            if batch is None:
                continue
            img = batch["t2w"].to(device, non_blocking=True)
            label = batch["seg"].to(device, non_blocking=True)
            seg = unet(img.float())
            seg_bin = (seg >= 0.5).float()
            dice_sum = dice_score(seg_bin, label.float()).item()
            total_dice_sum += dice_sum
            total_cases += img.size(0)
        img_min, img_max = img.min().item(), img.max().item()
        lab_min, lab_max = label.min().item(), label.max().item()
        seg_min, seg_max = seg.min().item(), seg.max().item()
    if total_cases == 0:
        print("No valid samples found.")
        return
    print("test_score = %.4f" % (total_dice_sum / total_cases))

if __name__ == "__main__":
    main()
