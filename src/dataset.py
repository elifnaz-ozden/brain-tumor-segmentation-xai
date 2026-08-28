from __future__ import annotations

import os
import sys
import glob
import random
import pathlib
from typing import List, Tuple, Optional

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader 
from sklearn.model_selection import train_test_split
from tqdm import tqdm 

from src import config as C

# 1) Files List / Volume (Patient) Identity ---

_ALL_FILES: Optional[List[str]] = None

def all_h5_files() -> List[str]:
    global _ALL_FILES
    if _ALL_FILES is None:
        _ALL_FILES = sorted(glob.glob(str(C.H5_DATA_DIR / "*.h5")))
    return _ALL_FILES

def volume_id(path: str) -> int:
    return int(os.path.basename(path).split("_")[1])

def slice_id(path: str) -> int:
    return int(os.path.basename(path).split("_")[3].split(".")[0])

def list_volumes() -> List[int]:
    return sorted({volume_id(f) for f in all_h5_files()})

def files_for_volume(vol_id: int) -> List[str]:
    fs = [f for f in all_h5_files() if volume_id(f) == vol_id]
    return sorted(fs, key=slice_id)

# 2) raw [NCR, ED, ET] -> nested [WT, TC, ET] ---

def mask_to_wttcet(mask_hwc: np.ndarray) -> np.ndarray:
    ncr = mask_hwc[..., 0].astype(bool) # necrotic / non-enhancing
    ed = mask_hwc[..., 1].astype(bool) # edema
    et = mask_hwc[..., 2].astype(bool) # enhancing
    wt = ncr | ed | et # whole tumor
    tc = ncr | et # core (not including edema)
    return np.stack([wt, tc, et], axis = -1).astype(np.float32)

# 3) Tumor including slices' indexes ( + cache ) ---

def build_tumor_fileset(cache_path: Optional[pathlib.Path] = None) -> set:
    if cache_path and cache_path.exists():
        with open(cache_path) as fh:
            names = {line.strip() for line in fh if line.strip()}
        print(f"[cache] Tumor including slices list is loaded: {len(names)} slices")
        return names

    print(">> Tumor including slices are getting scanned (for the first time; a few minutes)...")
    names = set()
    for fp in tqdm(all_h5_files(), desc = "tumor scanning"):
        with h5py.File(fp, "r") as f:
            if np.asarray(f["mask"]).any():
                names.add(os.path.basename(fp))

    if cache_path:
        with open(cache_path, "w") as fh:
            fh.write("\n".join(sorted(names)))
        print(f"[cache] {len(names)} tumor including slices are saved -> {cache_path}")
    return names 

# 4) Dataset ---

class H5SliceDataset(Dataset):

    def __init__(self, files: List[str], img_size: int = C.IMG_SIZE,
                 train: bool = False):
        self.files = files
        self.img_size = img_size
        self.train = train      

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with h5py.File(path, "r") as f:
            image = np.asarray(f["image"], dtype=np.float32)     # HxWx4 (normalize)
            mask = np.asarray(f["mask"], dtype=np.uint8)         # HxWx3 [NCR,ED,ET]

        mask = mask_to_wttcet(mask)                              # HxWx3 [WT,TC,ET]

        # HWC -> CHW tensor
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))  # [4,H,W]
        mask = torch.from_numpy(np.ascontiguousarray(mask.transpose(2, 0, 1)))    # [3,H,W]

        image = F.interpolate(image.unsqueeze(0), size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(self.img_size, self.img_size),
                             mode="nearest").squeeze(0)

        if self.train and random.random() < 0.5:
            image = torch.flip(image, dims=[2]) 
            mask = torch.flip(mask, dims=[2])

        return image.float(), mask.float()

# 5) Single sample loading  (for app / notebook) ---

def load_sample(h5_path: str, img_size: int = C.IMG_SIZE):
    ds = H5SliceDataset([h5_path], img_size=img_size, train=False)
    return ds[0]


# 6) Dataloader ---

def get_dataloaders(
    val_split: float = C.VAL_SPLIT,
    batch_size: int = C.BATCH_SIZE,
    seed: int = C.SEED,
    num_workers: int = C.NUM_WORKERS,
    filter_tumor: bool = True,
    img_size: int = C.IMG_SIZE,
) -> Tuple[DataLoader, DataLoader]:
    files = all_h5_files()
    if not files:
        raise RuntimeError(f"H5 not found: {C.H5_DATA_DIR}\n"
                           f"Is data/brats_h5 symlink correct?")

    vols = sorted({volume_id(f) for f in files})
    train_vols, val_vols = train_test_split(vols, test_size=val_split, random_state=seed)
    train_vols, val_vols = set(train_vols), set(val_vols)

    train_files = [f for f in files if volume_id(f) in train_vols]
    val_files = [f for f in files if volume_id(f) in val_vols]

    if filter_tumor:
        tumor = build_tumor_fileset(C.DATA_CACHE_DIR / "tumor_slices.txt")
        train_files = [f for f in train_files if os.path.basename(f) in tumor]
        val_files = [f for f in val_files if os.path.basename(f) in tumor]

    train_ds = H5SliceDataset(train_files, img_size=img_size, train=True)
    val_ds = H5SliceDataset(val_files, img_size=img_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    print(f"Patient : {len(train_vols)} train / {len(val_vols)} val")
    print(f"Slice : {len(train_files)} train / {len(val_files)} val "
          f"(filter_tumor={filter_tumor})")
    return train_loader, val_loader

# 7) Quick test  (python src / dataset.py) ---

if __name__ == "__main__":
    print("Total .h5:", len(all_h5_files()), "| patient:", len(list_volumes()))

    train_loader, val_loader = get_dataloaders(batch_size=4, num_workers=0,
                                               filter_tumor=False)
    x, y = next(iter(train_loader))
    print("\n--- One batch ---")
    print("image:", tuple(x.shape), x.dtype, "| min/max:",
          round(float(x.min()), 2), round(float(x.max()), 2))
    print("mask :", tuple(y.shape), y.dtype, "| unique :", torch.unique(y).tolist())
    
    import matplotlib.pyplot as plt
    
    j = int(y.sum(dim=(1, 2, 3)).argmax())
    img, msk = x[j].numpy(), y[j].numpy()

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i, m in enumerate(C.MODALITIES):
        axes[0, i].imshow(img[i], cmap="gray")
        axes[0, i].set_title(f"channel {i} ({m})"); axes[0, i].axis("off")
    for i, r in enumerate(C.REGIONS):
        axes[1, i].imshow(img[0], cmap="gray")
        axes[1, i].imshow(np.ma.masked_where(msk[i] == 0, msk[i]), cmap="autumn", alpha=0.5)
        axes[1, i].set_title(f"mask: {r}"); axes[1, i].axis("off")
    axes[1, 3].axis("off")
    fig.suptitle("dataset.py (h5) — image + WT/TC/ET mask")
    plt.tight_layout()
    out = C.FIGURES_DIR / "dataset_sample.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("\nSample image saved:", out)


