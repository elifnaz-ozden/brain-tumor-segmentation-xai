# Brain Tumor Segmentation with Explainable AI (XAI)

Segmentation of brain tumor sub-regions in multi-modal MRI (BraTS 2020) with a from-scratch 2D U-Net in PyTorch, plus Grad-CAM heatmaps that show where the model looks when it predicts each region.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)
![Grad-CAM](https://img.shields.io/badge/XAI-Grad--CAM-orange)

## Overview

Each patient has four MRI modalities (FLAIR, T1, T1ce, T2). The model predicts three **nested** tumor regions:

| Region | Contains | Raw labels |
|---|---|---|
| **WT** – Whole Tumor | everything abnormal | 1, 2, 4 |
| **TC** – Tumor Core | tumor without the edema | 1, 4 |
| **ET** – Enhancing Tumor | actively enhancing part | 4 |

Because the regions overlap, the output is **multi-label** (one sigmoid per region), not a softmax.

## Results

2D U-Net (7.76M parameters), `DiceFocalLoss`, Adam + cosine LR, **5 epochs** on an Apple M-series GPU (MPS, ~15 min).
Validation set: **74 patients never seen in training** (patient-level split, 295 / 74).

| Region | Dice | IoU |
|---|---|---|
| WT | **0.891** | 0.803 |
| TC | **0.839** | 0.723 |
| ET | **0.775** | 0.632 |
| **Mean** | **0.835** | 0.719 |

> Metrics are computed on 2D slices that contain tumor (4,902 validation slices), accumulated over the whole validation set. This is not the official BraTS 3D volume-level evaluation.

**Prediction + Grad-CAM** on a validation patient (the slice with the most tumor pixels):

![Prediction and Grad-CAM](figures/demo_prediction.png)

**Training curves:** validation Dice was still rising at epoch 5, so longer training should help.

![Training curves](figures/training_curves.png)

## Method

- **Data:** pre-sliced HDF5 version of BraTS 2020 (369 patients, 57,195 slices of 240×240×4, already z-normalized). Only the 24,422 slices that contain tumor are used. Slices are resized to 128×128, and training slices are randomly flipped horizontally.
- **Model:** classic U-Net encoder–decoder with skip connections (`src/models.py`). MONAI's residual U-Net and Attention U-Net are available through the same `get_model()` factory.
- **Loss:** Dice loss (optimizes overlap, robust to the small tumor area) + Focal loss (down-weights easy background pixels).
- **Explainability:** Grad-CAM on the last feature-rich conv layer, targeting the predicted pixels of the chosen region (`src/explainability.py`).

## Project Structure

```
├── src/
│   ├── config.py           # paths, labels, hyperparameters, device selection
│   ├── dataset.py          # h5 loader, raw labels -> WT/TC/ET, patient-level split
│   ├── models.py           # 2D U-Net from scratch + model factory
│   ├── losses.py           # Dice / Focal / DiceFocal
│   ├── train.py            # training loop, per-region Dice/IoU, best checkpoint
│   ├── explainability.py   # Grad-CAM
│   └── predict.py          # GT vs prediction vs Grad-CAM figure
├── app/
│   └── app.py              # Streamlit demo
├── figures/                # images used in this README
└── requirements.txt
```

## Getting Started

```bash
pip install -r requirements.txt

# Data: download the HDF5 version from Kaggle (awsaf49/brats2020-training-data),
# then link the folder with the .h5 files into the project:
ln -sfn /path/to/BraTS2020_training_data/content/data data/brats_h5

python src/config.py                 # check paths + device
python src/dataset.py                # sample figure -> figures/dataset_sample.png
python src/train.py --epochs 5       # train (on Mac: PYTORCH_ENABLE_MPS_FALLBACK=1)
python src/predict.py                # demo figure -> figures/demo_prediction.png
streamlit run app/app.py             # interactive demo
```

The data (~15 GB) and model weights are not included in the repository.

## Dataset & Citation

[BraTS 2020 (HDF5) on Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)

```bibtex
@article{menze2015brats,
  title={The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)},
  author={Menze, Bjoern H and others},
  journal={IEEE Transactions on Medical Imaging}, volume={34}, number={10},
  pages={1993--2024}, year={2015}
}
@article{bakas2017advancing,
  title={Advancing The Cancer Genome Atlas glioma MRI collections with expert
         segmentation labels and radiomic features},
  author={Bakas, Spyridon and others},
  journal={Scientific Data}, volume={4}, pages={170117}, year={2017}
}
```
