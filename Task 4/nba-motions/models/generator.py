"""DCGAN generator for 64x128 RGB trajectory cards (H x W).

Plain DCGAN in the spirit of the assignment example:
    z(100) -> Linear -> reshape (fg*8, 8, 16) -> 3 x ConvTranspose stride 2
    -> (3, 64, 128) -> Tanh

With feature_g=32 the reshape is (256, 8, 16) and the model is ~4M params.
BatchNorm + ReLU between the transposed convolutions; Tanh on the output so
values live in [-1, 1] (matches the dataset normalization).
"""
from __future__ import annotations

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim: int = 100, feature_g: int = 32, out_channels: int = 3) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.feature_g = feature_g
        self.init_ch = feature_g * 8                       # 256 channels at 8x16

        self.proj = nn.Linear(z_dim, self.init_ch * 8 * 16)

        self.net = nn.Sequential(
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(inplace=True),
            # (fg*8, 8, 16) -> (fg*4, 16, 32)
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(inplace=True),
            # -> (fg*2, 32, 64)
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(inplace=True),
            # -> (3, 64, 128)
            nn.ConvTranspose2d(feature_g * 2, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        self.apply(dcgan_weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2:
            z = z.view(z.size(0), -1)
        x = self.proj(z).view(z.size(0), self.init_ch, 8, 16)
        return self.net(x)


def dcgan_weights_init(m: nn.Module) -> None:
    """DCGAN init: conv weights N(0, 0.02), BatchNorm weights N(1, 0.02)."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif "Linear" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
