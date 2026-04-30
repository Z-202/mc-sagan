from __future__ import annotations

import csv
import math
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from torch.autograd import grad
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

from criteria.loss import dice_loss
from criteria.losses import PerceptualLoss
from datasets.bratsloader import BRATSVolumes
from helpers.utils import label2onehot
from models.segmentor import UNet3D
from models.genrator import Generator
from models.critic import PatchGAN_Critic
from project_config import DATA, PATHS, TRAIN_GAN, ensure_project_dirs

def seed_all(seed: int = 10):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

ensure_project_dirs()
seed_all(TRAIN_GAN["seed"])
torch.backends.cudnn.benchmark = TRAIN_GAN["cudnn_benchmark"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = DATA["gan_train"]
data_root_val = DATA["gan_val"]
save_dir = PATHS["gan_run_dir"]
BS = TRAIN_GAN["batch_size"]
NUM_WORKERS = TRAIN_GAN["num_workers"]
EPOCHS = TRAIN_GAN["epochs"]
PRINT_EVERY = TRAIN_GAN["print_every"]
N_CRITIC = TRAIN_GAN["n_critic"]
TARGET_KEY = TRAIN_GAN["target_key"]
SRC_KEY = TRAIN_GAN["src_key"]
LR_G = TRAIN_GAN["lr_g"]
LR_D = TRAIN_GAN["lr_d"]
BETAS = TRAIN_GAN["betas"]
LMB = TRAIN_GAN["lmb"]
RESUME = PATHS["gan_resume"] if TRAIN_GAN["resume"] else ""
USE_AMP = TRAIN_GAN["use_amp"]

def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@torch.no_grad()
def psnr_3d(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a01 = to01(a).float().clamp(0,1)
    b01 = to01(b).float().clamp(0,1)
    mse = F.mse_loss(a01, b01).item()
    if mse <= eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)

class CSVLogger:
    def __init__(self, csv_path: Path, fieldnames: list[str]):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        self._fh = open(self.csv_path, "a", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
        if not file_exists:
            self._w.writeheader(); self._fh.flush()
    def log(self, row: dict):
        out = {k: row.get(k, "") for k in self.fieldnames}
        self._w.writerow(out); self._fh.flush()
    def close(self):
        try: self._fh.close()
        except Exception: pass

LOG_DIR = Path(save_dir) / "logs"
FIG_DIR = Path(save_dir) / "figs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

STEP_CSV = LOG_DIR / "train_steps.csv"
STEP_FIELDS = [
    "time", "global_step", "epoch", "step",
    "lr_G", "lr_D",

    "D_tot", "D_real", "D_fake", "D_cls", "GP",

    "G_tot", "G_adv", "G_cls", "G_rec", "G_seg", "G_perc", "G_ssim",

    "MS_SSIM", "PSNR"
]
step_logger = CSVLogger(STEP_CSV, STEP_FIELDS)

EPOCH_CSV = LOG_DIR / "train_epochs.csv"
EPOCH_FIELDS = [
    "time", "epoch", "steps",
    "D_tot", "D_real", "D_fake", "D_cls", "GP",
    "G_tot", "G_adv", "G_cls", "G_rec", "G_seg", "G_perc", "G_ssim"
]
epoch_logger = CSVLogger(EPOCH_CSV, EPOCH_FIELDS)

VAL_CSV = LOG_DIR / "val_epochs.csv"
VAL_FIELDS = [
    "time",
    "epoch",
    "num_subjects",
    "num_pairs",
    "MAE",
    "MSE",
    "SSIM",
    "PSNR",
]
val_logger = CSVLogger(VAL_CSV, VAL_FIELDS)

train_ds = BRATSVolumes(data_root, mode="train")
assert SRC_KEY in train_ds[0] and TARGET_KEY in train_ds[0], \
    f"Dataset sample must contain keys '{SRC_KEY}' and '{TARGET_KEY}'"

train_ld = DataLoader(
    train_ds, batch_size=BS, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    persistent_workers=(NUM_WORKERS > 0),
)
steps_per_epoch = len(train_ld)
print(f"Dataset: {len(train_ds)} cases  →  {steps_per_epoch} steps/epoch  |  device={device}")

G = Generator(
    in_channels=TRAIN_GAN["generator_in_channels"],
    out_channels=TRAIN_GAN["generator_out_channels"],
    base_channels=TRAIN_GAN["generator_base_channels"],
).to(device)
D = PatchGAN_Critic(
    in_channels=TRAIN_GAN["critic_in_channels"],
    base_channels=TRAIN_GAN["critic_base_channels"],
    num_classes=TRAIN_GAN["critic_num_classes"],
).to(device)
S = UNet3D(
    in_dim=TRAIN_GAN["seg_in_dim"],
    c_dim=TRAIN_GAN["seg_c_dim"],
    base_ch=TRAIN_GAN["seg_base_ch"],
    out_dim=TRAIN_GAN["seg_out_dim"],
).to(device)
perc_loss_module = PerceptualLoss(
    use_3d=True,
    hub_repo=TRAIN_GAN["perc_hub_repo"],
    hub_model=TRAIN_GAN["perc_hub_model"],
    feat_l2norm=TRAIN_GAN["perc_feat_l2norm"],
    criterion=TRAIN_GAN["perc_criterion"],
    feature_mask=TRAIN_GAN["perc_feature_mask"],
    mask_thresh01=TRAIN_GAN["perc_mask_thresh01"],
    mask_dilate_k=TRAIN_GAN["perc_mask_dilate_k"],
).to(device)

perc_loss_module.eval()

pre_ckpt = TRAIN_GAN["pretrained_seg_ckpt"]
if os.path.isfile(pre_ckpt):
    S.load_state_dict(torch.load(pre_ckpt, map_location=device))
    print("Loaded pretrained UNet3D for segmentation → frozen (feature only).")

for p in S.parameters():
    p.requires_grad_(False)
S.eval()

opt_g = torch.optim.Adam(G.parameters(), lr=LR_G, betas=BETAS)
opt_d = torch.optim.Adam(D.parameters(), lr=LR_D, betas=BETAS)

def maybe_update_lr(epoch: int):

    if epoch in TRAIN_GAN["lr_milestones"]:
        for opt in (opt_g, opt_d):
            for pg in opt.param_groups:
                pg["lr"] *= 0.5

@torch.no_grad()
def save_mid_slice_triplet(inp: torch.Tensor, gen: torch.Tensor, tgt: torch.Tensor, fn: str):
    def to_img(x):
        x = (x.clamp(-1, 1) + 1) * 0.5
        x = x[0, 0]
        z = x.shape[0] // 2
        return x[z].detach().cpu().numpy()
    imgs  = [to_img(inp), to_img(gen), to_img(tgt)]
    titles = ["Input (T2)", "Generated", "Target"]
    h, w = imgs[0].shape
    dpi  = 100
    fig, axs = plt.subplots(1, 3, figsize=(3*w/dpi, h/dpi), dpi=dpi)
    for ax, im, t in zip(axs, imgs, titles):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="none")
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(fn, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def gradient_penalty(critic_fn, real, fake, cond=None, gp_lambda=10.0):

    device = real.device
    alpha = torch.rand(real.size(0), 1, 1, 1, 1, device=device)
    x_hat = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    score_hat, _ = critic_fn(x_hat, cond)

    score_hat_sum = score_hat.view(score_hat.size(0), -1).sum(dim=1)

    grads = torch.autograd.grad(
        outputs=score_hat_sum,
        inputs=x_hat,
        grad_outputs=torch.ones_like(score_hat_sum),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grads = grads.view(grads.size(0), -1)
    gp = ((grads.norm(2, dim=1) - 1.0)**2).mean() * gp_lambda
    return gp

def to01(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) * 0.5
def largest_odd_leq(n: int) -> int:
    if n < 3:
        return 3
    return n if (n % 2 == 1) else (n - 1)


def safe_ssim_skimage(
    x: np.ndarray,
    y: np.ndarray,
    data_range: float = 1.0,
    win_size: int = 11,
    gaussian_weights: bool = False,
    sigma: float = 1.5,
    use_sample_covariance: bool = True,
) -> float:
    if x.shape != y.shape:
        raise ValueError(f"SSIM shape mismatch: {x.shape} vs {y.shape}")

    if float(np.std(x)) < 1e-12 and float(np.std(y)) < 1e-12:
        return 1.0 if float(np.max(np.abs(x - y))) < 1e-12 else 0.0

    min_dim = min(x.shape)
    w = largest_odd_leq(min(win_size, min_dim))

    if w > min_dim:
        w = largest_odd_leq(min_dim)

    return float(
        structural_similarity(
            x,
            y,
            data_range=data_range,
            win_size=w,
            gaussian_weights=gaussian_weights,
            sigma=sigma,
            use_sample_covariance=use_sample_covariance,
        )
    )


def maybe_fix_axis_order(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape == b.shape:
        return a, b

    if sorted(a.shape) != sorted(b.shape):
        raise ValueError(f"Shape mismatch not fixable by axis permutation: {a.shape} vs {b.shape}")

    perms = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]

    for p in perms:
        ap = np.transpose(a, p)
        if ap.shape == b.shape:
            return ap, b

    raise ValueError(f"Could not permute axes to match shapes: {a.shape} vs {b.shape}")


def tensor_to_eval_numpy01(x: torch.Tensor) -> np.ndarray:
    arr = to01(x.detach()).float().cpu().numpy()

    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected single-channel volume, got shape {arr.shape}")
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume after channel removal, got shape {arr.shape}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return arr


def eval_metrics_3d_from_tensors(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    ssim_win: int = 11,
    ssim_sigma: float = 1.5,
) -> dict[str, float]:
    pred_np = tensor_to_eval_numpy01(pred)
    target_np = tensor_to_eval_numpy01(target)

    pred_np, target_np = maybe_fix_axis_order(pred_np, target_np)

    mae_val = float(np.mean(np.abs(target_np - pred_np)))

    mse_val = float(mean_squared_error(target_np, pred_np))

    psnr_val = float(
        peak_signal_noise_ratio(
            target_np,
            pred_np,
            data_range=data_range,
        )
    )

    ssim_val = safe_ssim_skimage(
        target_np,
        pred_np,
        data_range=data_range,
        win_size=ssim_win,
        gaussian_weights=False,
        sigma=ssim_sigma,
        use_sample_covariance=True,
    )

    return {
        "MAE": mae_val,
        "MSE": mse_val,
        "SSIM": ssim_val,
        "PSNR": psnr_val,
    }


def mean_finite(values: list[float]) -> float:
    vals = np.array(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else float("nan")
@torch.no_grad()
def validate_epoch(G_eval, data_root_val, device, epoch: int):
    try:
        val_ds = BRATSVolumes(data_root_val, mode="val")
    except Exception as e:
        print(f"[VAL] skipped: could not create validation dataset: {e}")
        return

    val_ld = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=TRAIN_GAN["val_num_workers"],
        pin_memory=True,
    )

    val_targets = [
        ("t1n", 2),
        ("t1c", 1),
        ("t2f", 0),
    ]

    was_training = G_eval.training
    G_eval.eval()

    mae_values: list[float] = []
    mse_values: list[float] = []
    ssim_values: list[float] = []
    psnr_values: list[float] = []

    num_subjects = 0
    num_pairs = 0

    saved_preview_modalities = set()

    try:
        for batch in val_ld:
            if "t2w" not in batch:
                raise KeyError("Validation batch is missing required source key: 't2w'")

            t2 = batch["t2w"].to(device, non_blocking=True)
            bsz = t2.size(0)
            num_subjects += int(bsz)

            for target_key, label_index in val_targets:
                if target_key not in batch:
                    raise KeyError(f"Validation batch is missing required target key: '{target_key}'")

                real = batch[target_key].to(device, non_blocking=True)

                lbl_idx = torch.full(
                    size=(bsz,),
                    fill_value=label_index,
                    device=device,
                    dtype=torch.long,
                )

                lbl_one = label2onehot(lbl_idx, 3).to(device=device, dtype=torch.float32)

                fake = G_eval(t2, lbl_one)

                for i in range(bsz):
                    metrics = eval_metrics_3d_from_tensors(
                        pred=fake[i],
                        target=real[i],
                        data_range=1.0,
                        ssim_win=11,
                        ssim_sigma=1.5,
                    )

                    mae_values.append(metrics["MAE"])
                    mse_values.append(metrics["MSE"])
                    ssim_values.append(metrics["SSIM"])
                    psnr_values.append(metrics["PSNR"])
                    num_pairs += 1

                if target_key not in saved_preview_modalities:
                    preview_path = Path(save_dir) / "images" / f"epoch_VAL_{epoch:03d}_{target_key}.png"
                    save_mid_slice_triplet(
                        t2[:1],
                        fake[:1],
                        real[:1],
                        str(preview_path),
                    )
                    saved_preview_modalities.add(target_key)

        if num_pairs == 0:
            print("[VAL] no validation pairs were evaluated.")
            return

        mae_mean = mean_finite(mae_values)
        mse_mean = mean_finite(mse_values)
        ssim_mean = mean_finite(ssim_values)
        psnr_mean = mean_finite(psnr_values)

        print(
            f"[VAL] subjects {num_subjects} | pairs {num_pairs} | "
            f"MAE {mae_mean:.6f} | "
            f"MSE {mse_mean:.6f} | "
            f"SSIM {ssim_mean:.6f} | "
            f"PSNR {psnr_mean:.6f}"
        )

        val_logger.log({
            "time": _now(),
            "epoch": int(epoch),
            "num_subjects": int(num_subjects),
            "num_pairs": int(num_pairs),
            "MAE": float(mae_mean),
            "MSE": float(mse_mean),
            "SSIM": float(ssim_mean),
            "PSNR": float(psnr_mean),
        })

    finally:
        G_eval.train(was_training)
start_epoch  = 1
global_step  = 0
@torch.no_grad()

def _move_optimizer_state_to_device(optimizer, device):

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def is_valid_torch_file(path):
    try:
        _ = torch.load(path, map_location="cpu")
        return True
    except Exception:
        return False

if RESUME and os.path.isfile(RESUME):
    assert is_valid_torch_file(RESUME), f"Corrupted ckpt: {RESUME}"
    ckpt = torch.load(RESUME, map_location="cpu")

    G.load_state_dict(ckpt["G"], strict=True)
    D.load_state_dict(ckpt["D"], strict=True)

    opt_g.load_state_dict(ckpt["opt_g"])
    opt_d.load_state_dict(ckpt["opt_d"])
    _move_optimizer_state_to_device(opt_g, device)
    _move_optimizer_state_to_device(opt_d, device)

    last_epoch   = int(ckpt.get("epoch", 0))
    start_epoch  = max(1, last_epoch + 1)
    global_step  = int(ckpt.get("global_step", 0))

    print(f"Resumed from {RESUME} at epoch {start_epoch} (last finished: {last_epoch}, global_step={global_step})")

for epoch in range(start_epoch, EPOCHS + 1):
    G.train(); D.train()
    maybe_update_lr(epoch)

    running = dict(
        d_real=0.0, d_fake=0.0, d_cls=0.0, gp=0.0, d_total=0.0,
        g_adv=0.0, g_cls=0.0, g_rec=0.0, g_seg=0.0, g_perc=0.0, g_ssim=0.0, g_total=0.0
    )

    steps_this_epoch = 0
    for step, batch in enumerate(train_ld, 1):
        steps_this_epoch += 1
        global_step += 1
        t2   = batch['t2w'].to(device)
        flair= batch['t2f'].to(device)
        t1c  = batch['t1c'].to(device)
        t1   = batch['t1n'].to(device)
        seg  = batch['seg' ].to(device)

        lbl_idx = torch.randint(3, (t2.size(0),), device=device)
        lbl_one = label2onehot(lbl_idx, 3).to(device)

        real = torch.zeros_like(t2)
        real[lbl_idx==0] = flair[lbl_idx==0]
        real[lbl_idx==1] = t1c [lbl_idx==1]
        real[lbl_idx==2] = t1  [lbl_idx==2]
        for p in D.parameters(): 
            p.requires_grad_(True)
        d_loop_acc = dict(d_real=0.0, d_fake=0.0, gp=0.0, d_total=0.0, d_cls=0.0)

        for _ in range(N_CRITIC):
            D.train(); G.eval()
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake = G(t2, lbl_one)

            out_r_src, out_r_cls = D(real, t2)
            out_f_src, out_f_cls = D(fake, t2)

            d_loss_real = out_r_src.mean()
            d_loss_fake = out_f_src.mean()

            d_loss_cls  = F.cross_entropy(out_r_cls, lbl_idx)

            with autocast(False):
                gp = gradient_penalty(D, real.float(), fake.float(), t2.float(), gp_lambda=LMB['GP'])

            d_total = (d_loss_fake - d_loss_real) + LMB['CLS'] * d_loss_cls + gp
            d_total.backward()
            opt_d.step()
            d_loop_acc["d_real"]  += d_loss_real.item()
            d_loop_acc["d_fake"]  += d_loss_fake.item()
            d_loop_acc["d_cls"]   += d_loss_cls.item()
            d_loop_acc["gp"]      += gp.item()
            d_loop_acc["d_total"] += d_total.item()
            
        for k in d_loop_acc:
            d_loop_acc[k] /= float(N_CRITIC)

        psnr_val = 0.0
        for p in D.parameters():
            p.requires_grad_(False)
        D.eval(); G.train()
        opt_g.zero_grad(set_to_none=True)

        fake = G(t2, lbl_one.float())
        fake = fake.float()

        out_f_src, out_f_cls = D(fake, t2)
        out_f_src = out_f_src.float()
        out_f_cls = out_f_cls.float()
        g_loss_fake = (-out_f_src).mean()
        g_loss_cls  = F.cross_entropy(out_f_cls, lbl_idx)
        seg_w = seg.float()
        if seg_w.dim() == 4:
            seg_w = seg_w.unsqueeze(1)

        alpha = 4.0  
        w = 1.0 + alpha * seg_w
        g_loss_rec = (w * (fake.float() - real.float()).abs()).mean()
        def s_forward_logits(a, b):
            return S(a, b, return_logits=True)

        with autocast(False):
            logits_seg = checkpoint(
                s_forward_logits,
                fake.float(),
                lbl_one.float(),
                use_reentrant=False,
                preserve_rng_state=False
            )

            seg_bin = seg.float()
            if seg_bin.dim() == 4:
                seg_bin = seg_bin.unsqueeze(1)

            bce  = F.binary_cross_entropy_with_logits(logits_seg, seg_bin)

            prob = torch.sigmoid(logits_seg)
            g_dice = dice_loss(prob, seg_bin)

            g_loss_seg = 0.5 * bce + 0.5 * g_dice


        g_loss_ssim = (1.0 - ms_ssim(to01(fake).float(), to01(real).float(),
                             data_range=1.0, win_size=7, size_average=True)).float()
        g_loss_perc = perc_loss_module(fake.float(), real.float(), mask_from=t2.float()).float()
        g_total = (g_loss_fake
                   + LMB['CLS'] * g_loss_cls
                   + LMB['REC'] * g_loss_rec
                   + LMB['SEG'] * g_loss_seg
                   + LMB['PERC']* g_loss_perc
                   + LMB['SSIM']* g_loss_ssim)

        try:
            psnr_val += psnr_3d(fake, real)
        except Exception:
            pass
            
        g_total.backward()
        opt_g.step()

        running["d_real"]  += d_loop_acc["d_real"]
        running["d_fake"]  += d_loop_acc["d_fake"]
        running["gp"]      += d_loop_acc["gp"]
        running["d_total"] += d_loop_acc["d_total"]
        running["d_cls"]   += d_loop_acc["d_cls"]

        running["g_adv"]   += g_loss_fake.item()
        running["g_cls"]   += g_loss_cls.item()
        running["g_rec"]   += g_loss_rec.item()
        running["g_seg"]   += g_loss_seg.item()
        running["g_perc"]  += g_loss_perc.item()
        running["g_ssim"]  += g_loss_ssim.item()
        running["g_total"] += g_total.item()


        with torch.no_grad():
            ms_ssim_val = 1.0 - float(g_loss_ssim)
            psnr_val = float(psnr_val) if psnr_val != 0.0 else ""

        step_logger.log({
            "time": _now(),
            "global_step": int(global_step),
            "epoch": int(epoch),
            "step": int(step),
            "lr_G": float(opt_g.param_groups[0]["lr"]),
            "lr_D": float(opt_d.param_groups[0]["lr"]),
            "D_tot": float(d_loop_acc["d_total"]),
            "D_real": float(d_loop_acc["d_real"]),
            "D_fake": float(d_loop_acc["d_fake"]),
            "D_cls": float(d_loop_acc["d_cls"]),
            "GP": float(d_loop_acc["gp"]),
            "G_tot": float(g_total),
            "G_adv": float(g_loss_fake),
            "G_cls": float(g_loss_cls),
            "G_rec": float(g_loss_rec),
            "G_seg": float(g_loss_seg),
            "G_perc": float(g_loss_perc),
            "G_ssim": float(g_loss_ssim),
            "MS_SSIM": float(ms_ssim_val),
            "PSNR": float(psnr_val) if psnr_val != "" else "",
        })

        if (step % PRINT_EVERY) == 0 or step == steps_per_epoch:
            print(
                f"[{epoch:03d}/{EPOCHS}] "
                f"step {step:03d}/{steps_per_epoch} | "
                f"D_tot {float(d_loop_acc['d_total']):6.3f}  "
                f"(real {float(d_loop_acc['d_real']):5.3f}  "
                f"fake {float(d_loop_acc['d_fake']):5.3f}  "
                f"cls {float(d_loop_acc['d_cls']):5.3f}  "
                f"gp {float(d_loop_acc['gp']):5.3f}) | "
                f"G_tot {float(g_total):6.3f}  "
                f"(adv {float(g_loss_fake):5.3f}  "
                f"cls {float(g_loss_cls):5.3f}  "
                f"rec {float(g_loss_rec):5.3f}  "
                f"seg {float(g_loss_seg):5.3f}  "
                f"perc {float(g_loss_perc):5.3f}  "
                f"ssim {float(g_loss_ssim):5.3f}) | "
                f"lr {opt_g.param_groups[0]['lr']:.1e}  "
            )

    for k in running: running[k] /= steps_this_epoch
    print(
        f"Epoch {epoch:03d} | "
        f"D: tot {running['d_total']:.4f}, real {running['d_real']:.4f}, fake {running['d_fake']:.4f}, cls {running['d_cls']:.4f}, gp {running['gp']:.4f} | "
        f"G: tot {running['g_total']:.4f}, adv {running['g_adv']:.4f}, cls {running['g_cls']:.4f}, L1 {running['g_rec']:.4f}, seg {running['g_seg']:.4f}, perc {running['g_perc']:.4f}, SSIM {running['g_ssim']:.4f}"
    )

    epoch_logger.log({
        "time": _now(),
        "epoch": int(epoch),
        "steps": int(steps_this_epoch),
        "D_tot": float(running["d_total"]),
        "D_real": float(running["d_real"]),
        "D_fake": float(running["d_fake"]),
        "D_cls": float(running["d_cls"]),
        "GP": float(running["gp"]),
        "G_tot": float(running["g_total"]),
        "G_adv": float(running["g_adv"]),
        "G_cls": float(running["g_cls"]),
        "G_rec": float(running["g_rec"]),
        "G_seg": float(running["g_seg"]),
        "G_perc": float(running["g_perc"]),
        "G_ssim": float(running["g_ssim"]),
    })

    if epoch % 1 == 0:
        G.eval()
        with torch.no_grad(): samp = G(t2[:1], lbl_one[:1])
        G.train()
        save_mid_slice_triplet(t2[:1], samp, real[:1], f"{save_dir}/images/epoch_{epoch:03d}.png")

    if (epoch % 1) == 0:
        validate_epoch(G, data_root_val, device, epoch)

    G.train()
    ckpt = dict(
        epoch=epoch,
        global_step=global_step,
        G=G.state_dict(),
        D=D.state_dict(),
        opt_g=opt_g.state_dict(),
        opt_d=opt_d.state_dict(),
    )
    torch.save(ckpt, f"{save_dir}/ckpt/latest.pt")
    if (epoch % 1) == 0:
        torch.save(G.state_dict(), f"{save_dir}/ckpt/epoch_{epoch:03d}.pt")

try:
    step_logger.close()
    epoch_logger.close()
    val_logger.close()
except Exception:
    pass

print("Training completed ✔")
