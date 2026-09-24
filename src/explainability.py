"""Grad-CAM explainability: where does the model look when it segments a region?"""

from __future__ import annotations

import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn

from src import config as C

# How Grad-CAM works:
# 1. Take the gradient of the chosen output score w.r.t. a conv layer's feature maps.
# 2. Average the gradients per feature map -> "how important is this map for the decision?"
# 3. Weighted sum of the feature maps, ReLU, upsample to image size -> heatmap (0..1).

# For segmentation there is no single class score like in classification. The target is
# the sum of the chosen region channel (WT/TC/ET) over the pixels the model predicted
# (pytorch-grad-cam's SemanticSegmentationTarget does exactly this).

# 1) Target Layer ---

# The LAST conv is the 1x1 output conv (it is basically the prediction itself),
# so the second to last conv is used: it carries richer, more interpretable features.
# In our UNet this is the second conv of the last DoubleConv.

def get_target_layer(model: nn.Module) -> nn.Module:
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("No Conv2d layer found in the model.")
    return convs[-2] if len(convs) >= 2 else convs[-1]

# 2) Grad-CAM ---

# image: [4, H, W] single slice | region_idx: 0=WT, 1=TC, 2=ET
# returns cam [H, W] in 0..1 and pred [H, W] in {0,1} (the model's mask for that region)

def compute_gradcam(model: nn.Module, image: torch.Tensor, region_idx: int,
                    device: torch.device, target_layer: nn.Module | None = None):
    # lazy import: the rest of the project works without the grad-cam package
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget

    if target_layer is None:
        target_layer = get_target_layer(model)

    model.eval().to(device)
    input_tensor = image.unsqueeze(0).to(device)   # [1, 4, H, W]

    with torch.no_grad():
        pred = (torch.sigmoid(model(input_tensor)) > 0.5).float()[0, region_idx].cpu().numpy()

    # if the model predicts nothing, the gradient would be 0 everywhere;
    # fall back to the whole image so the CAM is still meaningful
    mask = pred if pred.sum() > 0 else np.ones_like(pred, dtype=np.float32)
    targets = [SemanticSegmentationTarget(region_idx, mask)]

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]   # [H, W]

    return grayscale_cam, pred

# 3) Overlay (for plots / the app) ---

def overlay_cam(background_2d: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    import matplotlib.cm as cm

    bg = background_2d.astype(np.float32)
    bg = (bg - bg.min()) / (np.ptp(bg) + 1e-8)    # np.ptp: arr.ptp() was removed in NumPy 2.0
    bg_rgb = np.stack([bg, bg, bg], axis=-1)
    heat = cm.jet(cam)[..., :3]                   # CAM -> jet colors (RGB)
    return (1 - alpha) * bg_rgb + alpha * heat

# 4) Quick test  (python src/explainability.py, no data needed) ---
# Untrained model + random input: only checks that the pipeline runs (the map is meaningless).

if __name__ == "__main__":
    from src.models import get_model

    device = C.get_device()
    model = get_model("unet")
    dummy = torch.randn(C.IN_CHANNELS, C.IMG_SIZE, C.IMG_SIZE)

    try:
        cam, pred = compute_gradcam(model, dummy, region_idx=0, device=device)
        print("Grad-CAM ran OK")
        print("CAM :", cam.shape, "| min/max:", round(float(cam.min()), 3), round(float(cam.max()), 3))
        print("pred:", pred.shape)
    except ImportError:
        print("pytorch-grad-cam is not installed:  pip install grad-cam")
