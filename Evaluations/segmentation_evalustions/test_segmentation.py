"""
python test_segmentation.py --mode real
python test_segmentation.py --mode generated
"""

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import torch.utils.data as data

from datasets.bratsloader import BRATSVolumes
from models.unet import Unet
from project_config import DATA, TEST_GAN, TEST_UNET, TRAIN_GAN


GENERATED_DOMAINS = {"t2f": 0, "t1c": 1, "t1n": 2}
GENERATED_TARGETS = ("t2f", "t1c", "t1n")


def _setup_runtime() -> None:
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(TEST_UNET["seed"])
    torch.cuda.manual_seed_all(TEST_UNET["seed"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("real", "generated"),
        default=TEST_UNET.get("mode", "real"),
        help="real: evaluate UNet on real 4-channel test data | "
             "generated: generate (t1n, t1c, t2f) from t2w, then evaluate Dice",
    )
    return parser.parse_args()


@torch.no_grad()
def dice_score(pred_bin: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred_bin.contiguous().view(pred_bin.size(0), -1)
    target = target_bin.contiguous().view(target_bin.size(0), -1)
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return dice.sum()


def _filter_valid_batch(batch: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[torch.Tensor]]:
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
        for key, value in batch.items():
            if torch.is_tensor(value) and value.dim() >= 1 and value.size(0) == len(miss):
                out[key] = value.index_select(0, idx)
            else:
                out[key] = value
        return out, idx

    if str(miss).lower() != "none":
        return None, None

    return batch, None


def _build_test_loader(data_root: str, batch_size: int, num_workers: int) -> data.DataLoader:
    test_data = BRATSVolumes(data_root, mode="test")
    return data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


def _load_unet(device: torch.device) -> Unet:
    model_path = TEST_UNET["model_path"]
    unet = Unet(in_dim=4, conv_dim=32, out_dim=1).to(device)
    unet.load_state_dict(torch.load(model_path, map_location=device))
    unet.eval()
    return unet


def _load_real_inputs(batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    t1n = batch["t1n"].to(device, non_blocking=True)
    t1c = batch["t1c"].to(device, non_blocking=True)
    t2w = batch["t2w"].to(device, non_blocking=True)
    t2f = batch["t2f"].to(device, non_blocking=True)
    img = torch.cat([t1n, t1c, t2w, t2f], dim=1)
    label = batch["seg"].to(device, non_blocking=True).float()
    return img, label


def _compute_tumour_voxels(label: torch.Tensor, count_positive_labels_only: bool) -> torch.Tensor:
    if count_positive_labels_only:
        return (label > 0).float().sum(dim=(1, 2, 3, 4))
    return label.sum(dim=(1, 2, 3, 4))


def _to01(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) * 0.5


def _pad_right_to_multiple_3d(
    x: torch.Tensor,
    multiple: int = 16,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int, int], Tuple[int, int, int]]:
    assert x.dim() == 5, f"Expected 5D (B,C,D,H,W), got {x.shape}"
    _, _, d, h, w = x.shape

    def _pad_len(n: int) -> int:
        r = n % multiple
        return 0 if r == 0 else (multiple - r)

    pd = _pad_len(d)
    ph = _pad_len(h)
    pw = _pad_len(w)
    pad = (0, pw, 0, ph, 0, pd)

    if (pd + ph + pw) == 0:
        return x, pad, (d, h, w)

    x_pad = F.pad(x, pad, mode="constant", value=0.0)
    return x_pad, pad, (d, h, w)


def _unpad_3d(x: torch.Tensor, orig_dhw: Tuple[int, int, int]) -> torch.Tensor:
    d, h, w = orig_dhw
    return x[:, :, :d, :h, :w]


def _load_generator_checkpoint(generator: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt_path = str(ckpt_path)
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"GAN checkpoint not found: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location=device)

    if isinstance(obj, dict) and ("G" in obj) and isinstance(obj["G"], dict):
        generator.load_state_dict(obj["G"], strict=True)
        print(f"Loaded GAN (latest-style dict) : {ckpt_path}")
        return

    if isinstance(obj, dict):
        generator.load_state_dict(obj, strict=True)
        print(f"Loaded GAN (raw state_dict)    : {ckpt_path}")
        return

    raise RuntimeError(f"Unrecognized checkpoint format at: {ckpt_path}")


def _ensure_lbl_one_is_bx3(lbl_one: torch.Tensor) -> torch.Tensor:
    if lbl_one.dim() == 2:
        return lbl_one
    return lbl_one.view(lbl_one.size(0), lbl_one.size(1))


def _resolve_generated_config() -> Dict[str, Any]:
    return {
        "checkpoint_path": TEST_UNET.get("generated_gan_checkpoint", TEST_GAN["checkpoint_path"]),
        "in_channels": TEST_UNET.get(
            "generated_generator_in_channels",
            TEST_GAN.get("generator_in_channels", TRAIN_GAN.get("generator_in_channels", 4)),
        ),
        "out_channels": TEST_UNET.get(
            "generated_generator_out_channels",
            TEST_GAN.get("generator_out_channels", TRAIN_GAN.get("generator_out_channels", 1)),
        ),
        "base_channels": TEST_UNET.get(
            "generated_generator_base_channels",
            TEST_GAN.get("generator_base_channels", TRAIN_GAN.get("generator_base_channels", 64)),
        ),
        "use_amp": TEST_UNET.get("generated_use_amp", False),
        "brain_mask_threshold": TEST_UNET.get("generated_brain_mask_threshold", 0.05),
    }


def _build_generator(device: torch.device) -> torch.nn.Module:
    try:
        from models.genrator import Generator
    except ImportError:
        from genrator import Generator

    cfg = _resolve_generated_config()
    generator = Generator(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        base_channels=cfg["base_channels"],
    ).to(device)
    generator.eval()
    _load_generator_checkpoint(generator, cfg["checkpoint_path"], device)
    return generator


def _get_label2onehot():
    try:
        from helpers.utils import label2onehot
    except ImportError as exc:
        raise ImportError("Could not import label2onehot from helpers.utils") from exc
    return label2onehot


def _get_autocast_context(device: torch.device):
    cfg = _resolve_generated_config()
    use_amp = cfg["use_amp"]

    if (device.type == "cuda") and use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"AMP enabled: dtype={amp_dtype} (bf16_supported={torch.cuda.is_bf16_supported()})")
        return lambda: torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype)

    return nullcontext


def _prepare_generated_target(seg: torch.Tensor) -> torch.Tensor:
    if seg.dim() == 4:
        seg = seg.unsqueeze(1)
    return (seg > 0).float()


def _get_subject_id(batch: Dict[str, Any], index: int) -> str:
    try:
        return Path(batch["filedict"]["t2w"][0]).parent.name
    except Exception:
        return f"case_{index:04d}"


def _generate_modalities(
    generator: torch.nn.Module,
    t2w: torch.Tensor,
    label2onehot,
    device: torch.device,
    autocast_ctx,
    brain_mask_threshold: float,
) -> Dict[str, torch.Tensor]:
    brain_m = (_to01(t2w) > brain_mask_threshold)
    batch_size = t2w.size(0)
    generated = {}

    for target_name in GENERATED_TARGETS:
        idx = torch.full((batch_size,), GENERATED_DOMAINS[target_name], device=device, dtype=torch.long)
        lbl_one = label2onehot(idx, 3).to(device)
        lbl_one = _ensure_lbl_one_is_bx3(lbl_one)

        with autocast_ctx():
            fake = generator(t2w, lbl_one.float())

        fake = fake.float()
        fake = torch.where(brain_m, fake, torch.full_like(fake, -1.0))
        generated[target_name] = fake

    return generated


def _maybe_apply_sigmoid(pred: torch.Tensor) -> torch.Tensor:
    try:
        pred_min = float(pred.min().item())
        pred_max = float(pred.max().item())
        if (pred_min < 0.0) or (pred_max > 1.0):
            return torch.sigmoid(pred)
        return pred
    except Exception:
        return torch.sigmoid(pred)


def _run_on_real_data(unet: torch.nn.Module, test_loader: data.DataLoader, device: torch.device) -> None:
    tumour_min_voxels = TEST_UNET["tumour_min_voxels"]
    count_positive_labels_only = TEST_UNET["count_positive_labels_only"]

    total_dice_sum = 0.0
    total_cases = 0
    kept_samples = 0
    dropped_samples = 0
    skipped_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            batch, _ = _filter_valid_batch(batch)
            if batch is None:
                continue

            img, label = _load_real_inputs(batch, device)
            tumour_vox = _compute_tumour_voxels(label, count_positive_labels_only)
            keep = tumour_vox > tumour_min_voxels

            if keep.sum().item() == 0:
                skipped_batches += 1
                dropped_samples += img.size(0)
                continue

            img_kept = img[keep]
            label_kept = label[keep]

            kept_samples += img_kept.size(0)
            dropped_samples += img.size(0) - img_kept.size(0)

            pred = unet(img_kept.float())
            pred_bin = (pred >= 0.5).float()

            total_dice_sum += dice_score(pred_bin, label_kept).item()
            total_cases += img_kept.size(0)

    if total_cases == 0:
        print("No valid tumour samples found.")
        return

    print(f"kept={kept_samples} dropped={dropped_samples} skipped_batches={skipped_batches}")
    print("test_score = %.4f" % (total_dice_sum / total_cases))


def _run_on_generated_data(unet: torch.nn.Module, test_loader: data.DataLoader, device: torch.device) -> None:
    cfg = _resolve_generated_config()
    generator = _build_generator(device)
    label2onehot = _get_label2onehot()
    autocast_ctx = _get_autocast_context(device)

    total_dice_sum = 0.0
    total_cases = 0
    dataset_size = len(test_loader.dataset)

    with torch.no_grad():
        for index, batch in enumerate(test_loader, 1):
            batch, _ = _filter_valid_batch(batch)
            if batch is None:
                continue

            t2w = batch["t2w"].to(device, non_blocking=True)
            seg = batch["seg"].to(device, non_blocking=True)
            seg_bin_gt = _prepare_generated_target(seg)
            subject_id = _get_subject_id(batch, index)

            generated = _generate_modalities(
                generator=generator,
                t2w=t2w,
                label2onehot=label2onehot,
                device=device,
                autocast_ctx=autocast_ctx,
                brain_mask_threshold=cfg["brain_mask_threshold"],
            )

            img = torch.cat([generated["t1n"], generated["t1c"], t2w, generated["t2f"]], dim=1)

            img_pad, pad, orig_dhw = _pad_right_to_multiple_3d(img, multiple=16)
            seg_pad = F.pad(seg_bin_gt, pad, mode="constant", value=0.0) if sum(pad) > 0 else seg_bin_gt

            pred = unet(img_pad.float())
            pred = _maybe_apply_sigmoid(pred)

            pred = _unpad_3d(pred, orig_dhw)
            lab = _unpad_3d(seg_pad, orig_dhw)

            pred_bin = (pred >= 0.5).float()
            lab_bin = (lab >= 0.5).float()

            dice_sum = dice_score(pred_bin, lab_bin).item()
            total_dice_sum += dice_sum
            total_cases += img.size(0)

            print(f"[{index:4d}/{dataset_size}] {subject_id} | dice={dice_sum / img.size(0):.4f}")

    if total_cases == 0:
        print("No valid samples found.")
        return

    print("test_score = %.4f" % (total_dice_sum / total_cases))
    print("total_dice_sum  = ", total_dice_sum)
    print("total_cases     = ", total_cases)


def main() -> None:
    _setup_runtime()
    args = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = _build_test_loader(
        data_root=DATA["gan_test"],
        batch_size=TEST_UNET["batch_size"],
        num_workers=TEST_UNET["num_workers"],
    )
    unet = _load_unet(device)

    if args.mode == "generated":
        _run_on_generated_data(unet, test_loader, device)
    else:
        _run_on_real_data(unet, test_loader, device)


if __name__ == "__main__":
    main()
