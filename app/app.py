"""Streamlit demo: pick a patient + slice, see the segmentation and Grad-CAM side by side."""

# Run from the project root:
#   streamlit run app/app.py
# Needs: checkpoints/best_model.pth (python src/train.py) and data/brats_h5 (symlink)

import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import h5py
import torch
import streamlit as st

from src import config as C
from src.models import get_model
from src.dataset import load_sample, list_volumes, files_for_volume, slice_id
from src.explainability import compute_gradcam, overlay_cam

st.set_page_config(page_title="Brain Tumor Segmentation + XAI", page_icon="🧠", layout="wide")
st.title("Brain Tumor Segmentation & Grad-CAM Explainability")
st.caption("BraTS 2020 · 2D U-Net · WT / TC / ET · Grad-CAM heatmaps")

# 1) Model + Data (cached, loaded once) ---

@st.cache_resource
def load_model():
    ckpt_path = C.CHECKPOINT_DIR / "best_model.pth"
    if not ckpt_path.exists():
        return None, None, None
    device = C.get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    model = get_model(ckpt.get("model_name", "unet")).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device, ckpt

@st.cache_data
def get_volumes():
    return list_volumes()

@st.cache_data
def most_tumor_slice(vol_id: int) -> str:
    best_f, best_sum = None, -1
    for f in files_for_volume(vol_id):
        with h5py.File(f, "r") as h:
            s = int(np.asarray(h["mask"]).sum())
        if s > best_sum:
            best_sum, best_f = s, f
    return best_f

model, device, ckpt = load_model()
if model is None:
    st.warning("No trained model found (`checkpoints/best_model.pth`). "
               "Train first:\n\n```\npython src/train.py --epochs 3\n```")
    st.stop()

if not C.H5_DATA_DIR.exists():
    st.warning(f"Data not found: `{C.H5_DATA_DIR}`. Create the `data/brats_h5` symlink first.")
    st.stop()

st.sidebar.success(f"Model: {ckpt.get('model_name', 'unet')} · val Dice {ckpt.get('val_dice_mean', 0):.3f}")

# 2) Sidebar: Input Selection ---

st.sidebar.header("Input")
vol = st.sidebar.selectbox("Patient (volume)", get_volumes())
files = files_for_volume(vol)
slice_ids = [slice_id(f) for f in files]
default_idx = files.index(most_tumor_slice(vol))   # start at the most informative slice
pick = st.sidebar.select_slider("Slice", options=list(range(len(files))), value=default_idx,
                                format_func=lambda i: f"slice {slice_ids[i]}")
threshold = st.sidebar.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)

# 3) Inference ---

image, gt_mask = load_sample(files[pick])   # image [4,H,W], gt [3,H,W]
gt_mask = gt_mask.numpy()
flair = image[0].numpy()

with torch.no_grad():
    probs = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0].cpu().numpy()
pred_mask = (probs > threshold).astype(np.float32)

def color_overlay(bg, mask, color, alpha=0.5):
    bg = (bg - bg.min()) / (np.ptp(bg) + 1e-8)
    rgb = np.stack([bg, bg, bg], axis=-1)
    for c in range(3):
        rgb[..., c] = np.where(mask > 0, (1 - alpha) * rgb[..., c] + alpha * color[c], rgb[..., c])
    return rgb

# 4) Display ---

st.subheader(f"volume_{vol} · slice {slice_ids[pick]}")

st.markdown("**Input modalities**")
for col, i, name in zip(st.columns(4), range(4), C.MODALITIES):
    im = image[i].numpy()
    im = (im - im.min()) / (np.ptp(im) + 1e-8)
    col.image(im, caption=name.upper(), use_container_width=True, clamp=True)

for i, r in enumerate(C.REGIONS):
    st.markdown(f"### Region: {r}")
    c1, c2, c3 = st.columns(3)
    c1.image(color_overlay(flair, gt_mask[i], (1, 0.4, 0)),
             caption=f"Ground truth · {r}", use_container_width=True, clamp=True)
    c2.image(color_overlay(flair, pred_mask[i], (0, 0.6, 1)),
             caption=f"Prediction · {r}", use_container_width=True, clamp=True)
    cam, _ = compute_gradcam(model, image, region_idx=i, device=device)
    c3.image(overlay_cam(flair, cam, alpha=0.5),
             caption=f"Grad-CAM · {r}", use_container_width=True, clamp=True)

    inter = (pred_mask[i] * gt_mask[i]).sum()
    dice = (2 * inter + 1e-6) / (pred_mask[i].sum() + gt_mask[i].sum() + 1e-6)
    st.caption(f"{r} Dice on this slice: **{dice:.3f}**")

st.divider()
st.caption("The model was trained on 2D slices. Grad-CAM shows which areas of the last "
           "conv layer drove the prediction for the selected region.")
