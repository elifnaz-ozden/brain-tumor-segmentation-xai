"""Training + validation loop: per-region Dice/IoU, best checkpoint, loss/Dice curves."""

from __future__ import annotations

import sys
import json
import pathlib
import argparse

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src import config as C
from src.dataset import get_dataloaders
from src.models import get_model
from src.losses import get_loss

# Usage:
#   python src/train.py --epochs 3                      # baseline (unet + dicefocal)
#   python src/train.py --model attention_unet          # needs: pip install monai
#   python src/train.py --epochs 1 --limit-batches 5    # quick smoke test

# 1) Metrics ---

# The loss is what the model optimizes; these metrics are the "real" thresholded performance.
# Instead of averaging Dice per batch, intersection / sums are accumulated over the WHOLE
# validation set and Dice/IoU is computed once at the end. Per-batch averaging is unstable
# because many slices have tiny or empty masks for TC/ET.

@torch.no_grad()
def update_metric_sums(logits, targets, sums, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    dims = (0, 2, 3)   # sum over batch + H + W, keep the channel axis -> [3]
    sums["inter"] += (preds * targets).sum(dim=dims).cpu()
    sums["pred"] += preds.sum(dim=dims).cpu()
    sums["target"] += targets.sum(dim=dims).cpu()

def compute_dice_iou(sums, eps=1e-6):
    inter, p, t = sums["inter"], sums["pred"], sums["target"]
    dice = (2 * inter + eps) / (p + t + eps)       # [3] -> WT, TC, ET
    iou = (inter + eps) / (p + t - inter + eps)    # [3]
    return dice, iou

# 2) One Training Epoch ---

def train_one_epoch(model, loader, loss_fn, optimizer, device, limit_batches=0):
    model.train()   # BatchNorm uses batch statistics in train mode
    running, n = 0.0, 0
    for b, (x, y) in enumerate(tqdm(loader, desc="  train", leave=False)):
        if limit_batches and b >= limit_batches:
            break
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

        running += loss.item() * x.size(0)
        n += x.size(0)
    return running / max(n, 1)

# 3) Validation ---

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, limit_batches=0):
    model.eval()    # BatchNorm uses its running statistics in eval mode
    running, n = 0.0, 0
    sums = {k: torch.zeros(C.OUT_CHANNELS) for k in ("inter", "pred", "target")}
    for b, (x, y) in enumerate(tqdm(loader, desc="  val  ", leave=False)):
        if limit_batches and b >= limit_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        running += loss_fn(logits, y).item() * x.size(0)
        n += x.size(0)
        update_metric_sums(logits, y, sums)
    dice, iou = compute_dice_iou(sums)
    return running / max(n, 1), dice, iou

# 4) Main Loop ---

def main():
    args = parse_args()

    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)

    device = C.get_device()
    print(f"Device: {device} | Model: {args.model} | Loss: {args.loss}")

    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        filter_tumor=not args.all_slices,
        img_size=args.img_size,
    )

    model = get_model(args.model).to(device)
    loss_fn = get_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # cosine schedule: slowly lowers the learning rate -> fine-tuning toward the end
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": [], "val_dice_mean": []}
    best_dice, best = 0.0, {}

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, loss_fn, optimizer,
                                  device, args.limit_batches)
        val_loss, dice, iou = evaluate(model, val_loader, loss_fn,
                                       device, args.limit_batches)
        scheduler.step()

        mean_dice = float(dice.mean())
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_dice_mean"].append(mean_dice)

        dice_str = " ".join(f"{r}={d:.3f}" for r, d in zip(C.REGIONS, dice.tolist()))
        iou_str = " ".join(f"{r}={j:.3f}" for r, j in zip(C.REGIONS, iou.tolist()))
        tqdm.write(f"[{epoch:03d}/{args.epochs}] train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
                   f"Dice(mean={mean_dice:.3f} | {dice_str})  IoU({iou_str})")

        # keep only the best model (by mean validation Dice)
        if mean_dice > best_dice:
            best_dice = mean_dice
            best = {"epoch": epoch,
                    "dice": dict(zip(C.REGIONS, dice.tolist())),
                    "iou": dict(zip(C.REGIONS, iou.tolist()))}
            ckpt = C.CHECKPOINT_DIR / "best_model.pth"
            torch.save({"model_state": model.state_dict(),
                        "model_name": args.model,
                        "epoch": epoch,
                        "val_dice_mean": best_dice,
                        "regions": C.REGIONS}, ckpt)
            tqdm.write(f"      best model saved (Dice={best_dice:.3f}) -> {ckpt.name}")

    print(f"\nTraining finished. Best mean Dice: {best_dice:.3f}")

    # metrics are saved as json so the README numbers can be traced back to a run
    with open(C.OUTPUT_DIR / "metrics.json", "w") as fh:
        json.dump({"args": vars(args), "best": best, "history": history}, fh, indent=2)
    plot_curves(history)

# 5) Curves ---

def plot_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], marker="o", label="train")
    ax1.plot(epochs, history["val_loss"], marker="o", label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend()

    ax2.plot(epochs, history["val_dice_mean"], marker="o", color="green")
    ax2.set_title("Validation Dice (mean of WT/TC/ET)"); ax2.set_xlabel("epoch")
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    out = C.FIGURES_DIR / "training_curves.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("Curves saved:", out)

def parse_args():
    p = argparse.ArgumentParser(description="BraTS 2D segmentation training")
    p.add_argument("--model", default="unet", help="unet | monai_unet | attention_unet")
    p.add_argument("--loss", default="dicefocal", help="dice | focal | dicefocal | monai_dicefocal")
    p.add_argument("--epochs", type=int, default=C.NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=C.LEARNING_RATE)
    p.add_argument("--limit-batches", type=int, default=0,
                   help="0 = all batches; N > 0 = only the first N (quick test)")
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader workers; use 0 if it causes problems on Mac")
    p.add_argument("--img-size", type=int, default=C.IMG_SIZE)
    p.add_argument("--all-slices", action="store_true",
                   help="disable the tumor-slice filter (skips the cache scan)")
    return p.parse_args()

# on macOS DataLoader workers use 'spawn', so the entry point must be guarded
if __name__ == "__main__":
    main()
