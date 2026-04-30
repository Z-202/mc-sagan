import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from torch.cuda.amp import autocast, GradScaler

from models.segmentor import UNet3D
from helpers.utils import label2onehot, seed_torch
from criteria.loss import dice_loss, dice_score
from datasets.bratsloader import BRATSVolumes
from project_config import DATA, PATHS, TRAIN_PRE_UNET, ensure_project_dirs

def tumour_aware_crop_pair(img, seg, out_d, out_h, out_w, tumour_prob=0.75, jitter=8):
    assert img.dim() == 5 and seg.dim() == 5
    batch_size, _, depth, height, width = img.shape
    device = img.device
    out_img = torch.empty((batch_size, 1, out_d, out_h, out_w), device=device, dtype=img.dtype)
    out_seg = torch.empty((batch_size, 1, out_d, out_h, out_w), device=device, dtype=seg.dtype)
    for batch_index in range(batch_size):
        use_tumour = (torch.rand((), device=device) < tumour_prob) and (seg[batch_index].sum() > 0)
        if use_tumour:
            coords = torch.nonzero(seg[batch_index, 0] > 0, as_tuple=False)
            ridx = torch.randint(0, coords.shape[0], (1,), device=device).item()
            center_d = int(coords[ridx, 0].item())
            center_h = int(coords[ridx, 1].item())
            center_w = int(coords[ridx, 2].item())
            start_d = center_d - out_d // 2
            start_h = center_h - out_h // 2
            start_w = center_w - out_w // 2
            if jitter > 0:
                start_d += int(torch.randint(-jitter, jitter + 1, (1,), device=device).item())
                start_h += int(torch.randint(-jitter, jitter + 1, (1,), device=device).item())
                start_w += int(torch.randint(-jitter, jitter + 1, (1,), device=device).item())
        else:
            start_d = (depth - out_d) // 2
            start_h = (height - out_h) // 2
            start_w = (width - out_w) // 2
            if jitter > 0:
                start_d += int(torch.randint(-jitter, jitter + 1, (1,), device=device).item())
                start_h += int(torch.randint(-jitter, jitter + 1, (1,), device=device).item())
                start_w += int(torch.randint(-jitter, jitter + 1, (1,), device=device).item())
        start_d = max(0, min(start_d, depth - out_d))
        start_h = max(0, min(start_h, height - out_h))
        start_w = max(0, min(start_w, width - out_w))
        out_img[batch_index] = img[batch_index:batch_index + 1, :, start_d:start_d + out_d, start_h:start_h + out_h, start_w:start_w + out_w]
        out_seg[batch_index] = seg[batch_index:batch_index + 1, :, start_d:start_d + out_d, start_h:start_h + out_h, start_w:start_w + out_w]
    return out_img, out_seg

def main():
    ensure_project_dirs()
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = TRAIN_PRE_UNET["cuda_alloc_conf"]
    seed_torch(TRAIN_PRE_UNET["seed"])
    torch.backends.cudnn.benchmark = TRAIN_PRE_UNET["cudnn_benchmark"]
    print("\n******************* train_pretrained_unet (3-D) *******************")
    data_dir = DATA["gan_train"]
    save_dir = PATHS["weight_dir"]
    os.makedirs(save_dir, exist_ok=True)
    ckpt_latest = PATHS["pre_unet_latest"]
    ckpt_best = PATHS["pre_unet_best"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = TRAIN_PRE_UNET["batch_size"]
    num_workers = TRAIN_PRE_UNET["num_workers"]
    epochs = TRAIN_PRE_UNET["epochs"]
    lr = TRAIN_PRE_UNET["lr"]
    print_every = TRAIN_PRE_UNET["print_every"]
    crop_d = TRAIN_PRE_UNET["crop_d"]
    crop_h = TRAIN_PRE_UNET["crop_h"]
    crop_w = TRAIN_PRE_UNET["crop_w"]
    tumour_crop_prob = TRAIN_PRE_UNET["tumour_crop_prob"]
    w_dice = TRAIN_PRE_UNET["w_dice"]
    w_bce = TRAIN_PRE_UNET["w_bce"]
    base_ch = TRAIN_PRE_UNET["base_ch"]
    train_set = BRATSVolumes(data_dir, mode="train")
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    steps_per_epoch = len(train_loader)
    print(f"Dataset: {len(train_set)} cases → {steps_per_epoch} steps/epoch | device={device}\n")
    unet = UNet3D(in_dim=1, c_dim=3, base_ch=base_ch, out_dim=1).to(device)
    optimizer = optim.Adam(unet.parameters(), lr=lr)
    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    best_mean_dice = -1.0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        unet.train()
        epoch_dice_sum = 0.0
        epoch_n = 0
        for step, batch in enumerate(train_loader, start=1):
            t2f = batch["t2f"].to(device, non_blocking=True)
            t1c = batch["t1c"].to(device, non_blocking=True)
            t1n = batch["t1n"].to(device, non_blocking=True)
            seg = batch["seg"].to(device, non_blocking=True)
            current_batch = seg.size(0)
            info_c_idx = torch.randint(0, 3, (current_batch,), device=device)
            info_c = label2onehot(info_c_idx, 3)
            mods = torch.stack([t2f, t1c, t1n], dim=1)
            img = mods[torch.arange(current_batch, device=device), info_c_idx]
            img_c, seg_c = tumour_aware_crop_pair(
                img,
                seg,
                crop_d,
                crop_h,
                crop_w,
                tumour_prob=tumour_crop_prob,
                jitter=8,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp, dtype=amp_dtype):
                logits = unet(img_c, info_c, return_logits=True)
                prob = torch.sigmoid(logits)
                loss_d = dice_loss(prob.float(), seg_c.float())
                loss_b = F.binary_cross_entropy_with_logits(logits.float(), seg_c.float())
                loss = w_dice * loss_d + w_bce * loss_b
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                pred_bin = (prob > 0.5).float()
                batch_dice = dice_score(pred_bin, seg_c).item()
            epoch_dice_sum += batch_dice * current_batch
            epoch_n += current_batch
            if (step % print_every) == 0 or step == steps_per_epoch:
                dt = time.time() - t0
                print(
                    f"Epoch [{epoch:03d}/{epochs}] Step [{step:04d}/{steps_per_epoch}] "
                    f"Loss {loss.item():.4f} (Dice {loss_d.item():.4f}, BCE {loss_b.item():.4f}) "
                    f"Dice {batch_dice:.4f} t/iter {dt / print_every:.3f}s"
                )
                t0 = time.time()
        mean_dice = epoch_dice_sum / max(1, epoch_n)
        print(f"Epoch {epoch:03d} ➟ mean Dice = {mean_dice:.4f}")
        torch.save(unet.state_dict(), ckpt_latest)
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            torch.save(unet.state_dict(), ckpt_best)
            print(f"✅ New best checkpoint saved: {ckpt_best} (mean Dice={best_mean_dice:.4f})")
    print(f"\n✅ Training finished — latest: {ckpt_latest} | best: {ckpt_best}\n")

if __name__ == "__main__":
    main()
