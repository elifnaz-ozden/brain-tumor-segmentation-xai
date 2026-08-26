"""Central config: paths, labels, hyperparameters, device selection."""

from pathlib import Path
import torch

# 1) Paths ---

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

H5_DATA_DIR = DATA_DIR / "brats_h5"     # symlink to the pre-sliced .h5 dataset
DATA_CACHE_DIR = DATA_DIR / "cache"     # tumor-slice index cache

for _d in (FIGURES_DIR, CHECKPOINT_DIR, OUTPUT_DIR, DATA_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 2) Dataset (BraTS 2020) Information ---

# Every patient has 4 different MRI modalities and every one of them
# highlights a different aspect of the tumor, therefore all 4 are used.

# flair: shows the liquid around the tumor and the whole tumor the best.
# t1: anatomic reference (healthy tissue detail).
# t1ce: T1 + contrast matter. Emphasizes the enhancing tumor and the core.
# t2: shows the borders of the tumor and the liquid around it.

# Every patient folder also has a "_seg.nii" file: the tumor mask a radiology
# expert has drawn (the ground truth); the model will try to mimic this.

MODALITIES = ["flair", "t1", "t1ce", "t2"]
SEG_SUFFIX = "seg"

# Every .nii has the dimensions: 240 (height) x 240 (width) x 155 (number of slices)
VOLUME_SHAPE = (240, 240, 155)

# _seg.nii file's raw pixel values: 0, 1, 2, 4
LABEL_BACKGROUND = 0  # not tumor (healthy tissue / background)
LABEL_NCR_NET = 1     # necrotic + non-enhancing tumor core
LABEL_ED = 2          # edema (liquid around the tumor)
LABEL_ET = 4          # enhancing tumor

# Regions:

# These regions are nested and overlapping, therefore the model output
# is set up as multi-label (each one has a separate sigmoid) and not softmax.

# WT (Whole Tumor)     = 1 u 2 u 4 (everything)
# TC (Tumor Core)      = 1 u 4
# ET (Enhancing Tumor) = 4

# Model's output channels are also in this exact order: [WT, TC, ET]

REGIONS = ["WT", "TC", "ET"]
REGION_LABELS = {
    "WT": (LABEL_NCR_NET, LABEL_ED, LABEL_ET),  # 1, 2, 4
    "TC": (LABEL_NCR_NET, LABEL_ET),            # 1, 4
    "ET": (LABEL_ET,),                          # 4  
}

# 3) Preprocessing Parameters ---

# The h5 version of the dataset is already 2D sliced and normalized,
# therefore there is no need to open .nii / get slices / normalize.

# What is needed: eliminate tumor-free slices and give the rest to the model.
# Some slices have no tumor at all; only the ones with tumor will be used.
# (dataset.py will extract the tumor index and cache it.)

MIN_TUMOR_PIXELS = 1  # count a slice "positive" if it has at least this many tumor pixels

IMG_SIZE = 128

# 4) Model Parameters ---

IN_CHANNELS = len(MODALITIES)
OUT_CHANNELS = len(REGIONS)

# 5) Training Parameters ---

SEED = 42
VAL_SPLIT = 0.2
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_WORKERS = 4

# 6) Device Selection (GPU / MPS / CPU) ---

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    print("Project root:", PROJECT_ROOT)
    print("h5 data:", H5_DATA_DIR, "(exists? ->", H5_DATA_DIR.exists(), ")")
    print("Device:", get_device())
    print("In / Out Channels:", IN_CHANNELS, "/", OUT_CHANNELS)
    print("Regions:", REGIONS)
