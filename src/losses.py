"""Segmentation losses: Dice, Focal and their combination (multi-label, sigmoid per channel)."""

from __future__ import annotations

import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Why a special loss? The tumor is a small part of the image (class imbalance).
# A plain pixel-wise loss can score well by predicting "background everywhere"
# and still miss the tumor.

# Dice  : optimizes the OVERLAP directly, works well on small objects.
# Focal : down-weights easy (background) pixels, focuses on the hard ones.
# Both  : in practice the most robust combination.

# All losses take logits [B,3,H,W] and targets [B,3,H,W] in {0,1}.
# Each channel (WT/TC/ET) gets its own sigmoid, not a softmax, since the regions overlap.

# 1) Dice Loss ---

class DiceLoss(nn.Module):

    # Dice = 2|P n G| / (|P| + |G|)   (0 = no overlap, 1 = perfect)
    # "soft" Dice uses probabilities instead of thresholded masks, so it is differentiable.
    # smooth: avoids division by zero and keeps empty masks stable.
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (2, 3)                                             # sum over H, W
        intersection = (probs * targets).sum(dim=dims)            # [B, C]
        cardinality = probs.sum(dim=dims) + targets.sum(dim=dims) # [B, C]
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()

# 2) Focal Loss (Lin et al., 2017) ---

class FocalLoss(nn.Module):

    # focal = (1 - p_t)^gamma * BCE
    # gamma: if a pixel is already easy (p_t close to 1), its loss is pushed toward 0.
    #        gamma = 0 gives plain BCE.
    # alpha: positive / negative class balance (0.25 is the usual value).
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # per pixel
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)   # probability of the TRUE class
        focal = (1 - p_t) ** self.gamma * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal = alpha_t * focal
        return focal.mean()

# 3) Dice + Focal ---

class DiceFocalLoss(nn.Module):

    def __init__(self, lambda_dice: float = 1.0, lambda_focal: float = 1.0,
                 smooth: float = 1.0, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.dice = DiceLoss(smooth=smooth)
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.lambda_dice * self.dice(logits, targets)
                + self.lambda_focal * self.focal(logits, targets))

# 4) Loss Factory ---

def get_loss(name: str = "dicefocal") -> nn.Module:
    name = name.lower()
    if name == "dice":
        return DiceLoss()
    if name == "focal":
        return FocalLoss()
    if name == "dicefocal":
        return DiceFocalLoss()
    if name == "monai_dicefocal":
        from monai.losses import DiceFocalLoss as MonaiDiceFocal
        return MonaiDiceFocal(sigmoid=True)   # sigmoid=True -> multi-label
    raise ValueError(f"Unknown loss: {name} (options: dice, focal, dicefocal, monai_dicefocal)")

# 5) Quick test  (python src/losses.py, no data needed) ---

if __name__ == "__main__":
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 128, 128)
    targets = (torch.rand(2, 3, 128, 128) > 0.7).float()

    for name in ["dice", "focal", "dicefocal"]:
        print(f"{name:10s} loss: {get_loss(name)(logits, targets).item():.4f}")

    # sanity check: a perfect prediction should give Dice loss close to 0
    perfect = get_loss("dice")(targets * 20 - 10, targets)   # large +/- logits -> probs ~0/1
    print(f"\nDice loss on a perfect prediction (expected ~0): {perfect.item():.4f}")
