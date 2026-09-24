"""Segmentation models: a from-scratch 2D U-Net + MONAI alternatives via get_model()."""

from __future__ import annotations

import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config as C

# U-Net idea (Ronneberger et al., 2015):
# encoder (contracting path) shrinks the image and learns "what is there?",
# decoder (expanding path) grows it back and recovers "where is it?",
# skip connections carry fine details from encoder to decoder -> sharp tumor borders.

# Input : image  [B, 4, H, W]  (4 modalities)
# Output: logits [B, 3, H, W]  (WT/TC/ET raw scores; the sigmoid is applied in the loss, not here)

# 1) Building Block: (Conv -> BatchNorm -> ReLU) x 2 ---

class DoubleConv(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # bias=False because the BatchNorm right after already learns a bias
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),  # 3x3, keeps H/W
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

# 2) 2D U-Net (from scratch) ---

class UNet(nn.Module):

    # features = channel count at each level. The original paper uses (64,128,256,512);
    # a lighter version is used here so that training on a Mac (MPS) is fast.
    def __init__(self, in_channels: int = C.IN_CHANNELS,
                 out_channels: int = C.OUT_CHANNELS,
                 features=(32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()   # encoder blocks
        self.ups = nn.ModuleList()     # decoder: pairs of (upsample, DoubleConv)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)   # halves H and W

        # encoder
        c = in_channels
        for f in features:
            self.downs.append(DoubleConv(c, f))
            c = f

        # bottleneck (the deepest level)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # decoder: upsample 2x, then concatenate with the skip and apply DoubleConv
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))   # f (from below) + f (skip) = 2f inputs

        # 1x1 conv: reduce channels to the number of regions (WT/TC/ET)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []

        for down in self.downs:
            x = down(x)
            skips.append(x)       # save before pooling -> used by the decoder
            x = self.pool(x)

        x = self.bottleneck(x)

        skips = skips[::-1]       # deepest skip first
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)    # ConvTranspose (2x upsample)
            skip = skips[i // 2]

            # if H/W is not divisible by 16 the sizes can be off by 1px; align to the skip
            # (never triggers with 128x128 inputs)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                  align_corners=False)

            x = torch.cat([skip, x], dim=1)   # concatenate on the channel axis
            x = self.ups[i + 1](x)            # DoubleConv

        return self.final_conv(x)             # logits [B, out, H, W]

# 3) Model Factory ---

# "unet"           -> the U-Net above (baseline)
# "monai_unet"     -> MONAI residual U-Net (deeper, more stable)
# "attention_unet" -> MONAI Attention U-Net (attention gates focus on the tumor)
# MONAI is imported only when needed, so the baseline runs without it.
# (UNETR / SwinUNETR are left out: they are mainly for 3D and this pipeline is 2D.)

def get_model(name: str = "unet",
              in_channels: int = C.IN_CHANNELS,
              out_channels: int = C.OUT_CHANNELS) -> nn.Module:
    name = name.lower()

    if name == "unet":
        return UNet(in_channels, out_channels)

    if name == "monai_unet":
        from monai.networks.nets import UNet as MonaiUNet
        return MonaiUNet(spatial_dims=2, in_channels=in_channels, out_channels=out_channels,
                         channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
                         num_res_units=2)

    if name == "attention_unet":
        from monai.networks.nets import AttentionUnet
        return AttentionUnet(spatial_dims=2, in_channels=in_channels, out_channels=out_channels,
                             channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2))

    raise ValueError(f"Unknown model: {name} (options: unet, monai_unet, attention_unet)")

# 4) Quick test  (python src/models.py, no data needed) ---

if __name__ == "__main__":
    model = get_model("unet")
    x = torch.randn(2, C.IN_CHANNELS, C.IMG_SIZE, C.IMG_SIZE)   # fake batch
    y = model(x)
    print("Model :", model.__class__.__name__)
    print("Input :", tuple(x.shape))
    print("Output:", tuple(y.shape), "(expected: (2, 3, 128, 128))")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
