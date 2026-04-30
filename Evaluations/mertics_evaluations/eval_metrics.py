import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
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


def load_nifti_as_float(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine


def maybe_fix_axis_order(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a.shape == b.shape:
        return a, b
    if sorted(a.shape) != sorted(b.shape):
        raise ValueError(f"Shape mismatch not fixable by axis permutation: {a.shape} vs {b.shape}")

    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    for p in perms:
        ap = np.transpose(a, p)
        if ap.shape == b.shape:
            return ap, b
    raise ValueError(f"Could not permute axes to match shapes: {a.shape} vs {b.shape}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Path to generated_samples directory")
    ap.add_argument("--modalities", nargs="+", default=["flair", "t1", "t1c"])
    ap.add_argument("--sample_name", type=str, default="sample.nii.gz")
    ap.add_argument("--target_name", type=str, default="target.nii.gz")
    ap.add_argument("--data_range", type=float, default=1.0, help="Since your data is in [0,1], keep 1.0")
    ap.add_argument("--ssim_win", type=int, default=11, help="SSIM window size (odd). Common: 11")
    ap.add_argument("--ssim_sigma", type=float, default=1.5, help="SSIM gaussian sigma. Common: 1.5")
    ap.add_argument("--mask_thresh", type=float, default=0.0, help="Unused; kept for CLI compatibility")
    ap.add_argument("--crop_margin", type=int, default=2, help="Unused; kept for CLI compatibility")
    ap.add_argument("--clip_01", action="store_true", help="Optionally clip arrays to [0,1] defensively")
    ap.add_argument("--device", type=str, default="cpu", help="Unused; kept for CLI compatibility")
    ap.add_argument("--out_csv", type=str, default="eval_3d_metrics_per_subject.csv")
    ap.add_argument("--out_summary_csv", type=str, default="eval_3d_metrics_summary.csv")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"ROOT not found: {root}")

    subjects = sorted([p for p in root.iterdir() if p.is_dir()])
    if not subjects:
        raise RuntimeError(f"No subject folders found under: {root}")

    rows: List[Dict[str, object]] = []

    for subj_dir in subjects:
        subject_id = subj_dir.name
        for mod in args.modalities:
            mod_dir = subj_dir / mod
            sample_path = mod_dir / args.sample_name
            target_path = mod_dir / args.target_name
            if not sample_path.exists() or not target_path.exists():
                continue

            pred, pred_aff = load_nifti_as_float(sample_path)
            targ, targ_aff = load_nifti_as_float(target_path)
            pred, targ = maybe_fix_axis_order(pred, targ)

            pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            targ = np.nan_to_num(targ, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            if args.clip_01:
                pred = np.clip(pred, 0.0, 1.0)
                targ = np.clip(targ, 0.0, 1.0)
            affine_diff = float(np.max(np.abs(pred_aff - targ_aff)))
            if affine_diff > 1e-3:
                pass
            mse_sk = float(mean_squared_error(targ, pred))
            psnr_sk = float(peak_signal_noise_ratio(targ, pred, data_range=args.data_range))
            ssim_uni = safe_ssim_skimage(
                targ,
                pred,
                data_range=args.data_range,
                win_size=args.ssim_win,
                gaussian_weights=False,
                sigma=args.ssim_sigma,
                use_sample_covariance=True,
            )

            row: Dict[str, object] = {
                "subject_id": subject_id,
                "modality": mod,
                "shape": "x".join(map(str, pred.shape)),
                "affine_max_abs_diff": affine_diff,
                "mse_skimage": mse_sk,
                "psnr_skimage": psnr_sk,
                "ssim_skimage": ssim_uni,
            }
            rows.append(row)

    if not rows:
        raise RuntimeError("No (sample,target) pairs found. Check your folder structure and filenames.")
    out_csv = Path(args.out_csv)
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def to_float(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    metrics_to_summarize = [
        "mse_skimage",
        "psnr_skimage",
        "ssim_skimage",
    ]

    summary_rows: List[Dict[str, object]] = []
    mods = sorted(set(r["modality"] for r in rows))
    for mod in mods:
        rmod = [r for r in rows if r["modality"] == mod]
        print("\n" + "=" * 90)
        print(f"MODALITY: {mod}  |  N subjects: {len(set(rr['subject_id'] for rr in rmod))}  |  N pairs: {len(rmod)}")
        print("=" * 90)

        srow: Dict[str, object] = {"modality": mod, "n_pairs": len(rmod)}
        for m in metrics_to_summarize:
            vals = np.array([to_float(rr[m]) for rr in rmod], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            mean = float(np.mean(vals)) if vals.size else float("nan")
            std = float(np.std(vals)) if vals.size else float("nan")
            srow[m + "_mean"] = mean
            srow[m + "_std"] = std
            print(f"{m:30s}  mean={mean:.6f}   std={std:.6f}   (n={vals.size})")

        summary_rows.append(srow)

    out_sum = Path(args.out_summary_csv)
    with out_sum.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print("\nSaved:")
    print(f"  Per-subject metrics: {out_csv.resolve()}")
    print(f"  Summary metrics:     {out_sum.resolve()}")


if __name__ == "__main__":
    main()
