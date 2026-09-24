"""Inference on one validation patient: GT vs prediction vs Grad-CAM for each region."""

from __future__ import annotations

import sys
import pathlib
import argparse

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src import config as C
from src.models import get_model
from src.dataset import load_sample, list_volumes, files_for_volume
from src.explainability import compute_gradcam, overlay_cam

# Usage (after training):
#   python src/predict.py                # first validation patient
#   python src/predict.py --volume 42    # a specific patient

# 1) Load the Trained Model ---

def load_model(device: torch.device):
    ckpt_path = C.CHECKPOINT_DIR / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint: {ckpt_path}\nTrain first: python src/train.py --epochs 3")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = get_model(ckpt.get("model_name", "unet")).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt

# 2) Pick a Patient + Slice ---

# same split as dataset.get_dataloaders (same seed) -> the patient was never seen in training
def validation_volumes():
    _, val_vols = train_test_split(list_volumes(), test_size=C.VAL_SPLIT, random_state=C.SEED)
    return sorted(val_vols)

# the slice with the most tumor pixels is the most informative one to show
def most_tumor_slice(vol_id: int) -> str:
    best_f, best_sum = None, -1
    for f in files_for_volume(vol_id):
        with h5py.File(f, "r") as h:
            s = int(np.asarray(h["mask"]).sum())
        if s > best_sum:
            best_sum, best_f = s, f
    return best_f

# 3) Figure ---

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--volume", type=int, default=None, help="patient id (default: first val patient)")
    args = p.parse_args()

    device = C.get_device()
    model, ckpt = load_model(device)
    print(f"Model: {ckpt['model_name']} | epoch {ckpt['epoch']} | val Dice {ckpt['val_dice_mean']:.3f}")

    vol = args.volume if args.volume is not None else validation_volumes()[0]
    path = most_tumor_slice(vol)
    print("Slice:", pathlib.Path(path).name)

    image, gt = load_sample(path)          # image [4,H,W], gt [3,H,W] (WT/TC/ET)
    gt = gt.numpy()
    flair = image[0].numpy()               # FLAIR as the background (shows the whole tumor best)

    with torch.no_grad():
        probs = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0].cpu().numpy()
    pred = (probs > 0.5).astype(np.float32)

    # rows: modalities, then one row per region (GT | prediction | Grad-CAM)
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    for i, m in enumerate(C.MODALITIES):
        axes[0, i].imshow(image[i].numpy(), cmap="gray")
        axes[0, i].set_title(m.upper())

    for r, region in enumerate(C.REGIONS):
        row = r + 1
        cam, _ = compute_gradcam(model, image, region_idx=r, device=device)
        inter = (pred[r] * gt[r]).sum()
        dice = (2 * inter + 1e-6) / (pred[r].sum() + gt[r].sum() + 1e-6)

        axes[row, 0].imshow(flair, cmap="gray")
        axes[row, 0].set_title(f"{region}: FLAIR")
        axes[row, 1].imshow(flair, cmap="gray")
        axes[row, 1].imshow(np.ma.masked_where(gt[r] == 0, gt[r]), cmap="autumn", alpha=0.6)
        axes[row, 1].set_title(f"{region}: ground truth")
        axes[row, 2].imshow(flair, cmap="gray")
        axes[row, 2].imshow(np.ma.masked_where(pred[r] == 0, pred[r]), cmap="winter", alpha=0.6)
        axes[row, 2].set_title(f"{region}: prediction (Dice {dice:.2f})")
        axes[row, 3].imshow(overlay_cam(flair, cam))
        axes[row, 3].set_title(f"{region}: Grad-CAM")

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"volume_{vol} (validation patient) - {pathlib.Path(path).name}", fontsize=14)
    plt.tight_layout()
    out = C.FIGURES_DIR / "demo_prediction.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("Saved:", out)

if __name__ == "__main__":
    main()
