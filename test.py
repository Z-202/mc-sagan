from pathlib import Path
import time
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.bratsloader import BRATSVolumes
from models.genrator import Generator
from helpers.utils import seed_torch
from project_config import DATA, TEST_GAN, ensure_project_dirs

def to01(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) * 0.5

def save_nifti_like(data_np: np.ndarray, out_path, ref_img: nib.Nifti1Image) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdr = ref_img.header.copy()
    try:
        hdr.set_data_dtype(np.float32)
    except Exception:
        pass
    img = nib.Nifti1Image(data_np.astype(np.float32), ref_img.affine, hdr)
    nib.save(img, str(out_path))


def _load_ref_img(sample: dict, key_fallback: str) -> nib.Nifti1Image:
    filedict = sample.get("filedict", {})
    ref_key = key_fallback if key_fallback in filedict else "t2w"
    ref_path = filedict[ref_key][0]
    return nib.load(ref_path)

def main():
    ensure_project_dirs()
    seed_torch(TEST_GAN["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = BRATSVolumes(DATA["gan_test"], mode="test")
    ld = DataLoader(ds, batch_size=TEST_GAN["batch_size"], shuffle=False, num_workers=TEST_GAN["num_workers"])
    print(f"Found {len(ds)} test cases")
    generator = Generator(
        in_channels=TEST_GAN["generator_in_channels"],
        out_channels=TEST_GAN["generator_out_channels"],
        base_channels=TEST_GAN["generator_base_channels"],
    ).to(dev).eval()
    ckpt_path = TEST_GAN["checkpoint_path"]
    if Path(ckpt_path).is_file():
        print("Loading checkpoint:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=dev)
        state = ckpt["G"] if isinstance(ckpt, dict) and "G" in ckpt else ckpt
        generator.load_state_dict(state, strict=True)
    else:
        print(f"Checkpoint not found: {ckpt_path}")
    output_dir = Path(TEST_GAN["output_dir"])
    domains = TEST_GAN["domains"]
    targets = TEST_GAN["targets"]
    real_key = TEST_GAN["real_key"]
    threshold = TEST_GAN["brain_mask_threshold"]
    t0 = time.time()
    for index, sample in enumerate(ld, 1):
        t2 = sample["t2w"].to(dev)
        t2_01 = to01(t2).clamp(0.0, 1.0)
        brain_m = t2_01 > threshold
        subj_id = Path(sample["filedict"]["t2w"][0]).parent.name
        for tgt in targets:
            lbl = torch.zeros(1, 3, 1, 1, 1, device=dev, dtype=torch.float32)
            lbl[:, domains[tgt]] = 1.0
            with torch.no_grad():
                fake = generator(t2, lbl)
            fake = fake.squeeze(1)
            real = sample[real_key[tgt]].squeeze(1).to(dev)
            fake = to01(fake).clamp(0.0, 1.0)
            real = to01(real).clamp(0.0, 1.0)
            mask = brain_m.squeeze(1)
            fake[mask == 0] = 0.0
            real[mask == 0] = 0.0
            ref_img = _load_ref_img(sample, real_key[tgt])
            out_dir = output_dir / subj_id / tgt
            save_nifti_like(fake.detach().cpu().numpy()[0], out_dir / "sample.nii.gz", ref_img)
            save_nifti_like(real.detach().cpu().numpy()[0], out_dir / "target.nii.gz", ref_img)
        if index % 10 == 0 or index == len(ds):
            print(f"[{index:4d}/{len(ds)}] {subj_id} ✓ {(time.time() - t0) / index:.2f}s/subject")
    print(f"\n✅  All volumes written to {output_dir}")

if __name__ == "__main__":
    main()
