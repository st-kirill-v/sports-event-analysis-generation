"""DCGAN discriminator for 64x128 RGB trajectory cards (H x W).

Plain DCGAN: three stride-2 convolutions reduce (64, 128) to (8, 16), then a
linear head produces one score. BatchNorm + LeakyReLU between conv blocks (no
BatchNorm on the first conv, as in the DCGAN paper).

Defaults match a simple BCE setup: `use_sigmoid=True` so the head outputs a
probability in [0, 1]. Two optional stabilizers are off by default:
    - `spectral=True`  wraps the convs and head in spectral_norm (bounds the
      Lipschitz constant, keeps gradients flowing to G).
    - set `use_sigmoid=False` together with hinge loss to use raw scores.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from .generator import dcgan_weights_init


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        feature_d: int = 32,
        use_sigmoid: bool = True,
        spectral: bool = False,
    ) -> None:
        super().__init__()
        self.use_sigmoid = use_sigmoid

        def conv_block(c_in: int, c_out: int, batch_norm: bool) -> list[nn.Module]:
            conv = nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=not batch_norm)
            if spectral:
                conv = spectral_norm(conv)
            layers: list[nn.Module] = [conv]
            if batch_norm:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.body = nn.Sequential(
            # (3, 64, 128) -> (fd, 32, 64)
            *conv_block(in_channels, feature_d, batch_norm=False),
            # -> (fd*2, 16, 32)
            *conv_block(feature_d, feature_d * 2, batch_norm=True),
            # -> (fd*4, 8, 16)
            *conv_block(feature_d * 2, feature_d * 4, batch_norm=True),
        )
        head = nn.Linear(feature_d * 4 * 8 * 16, 1)
        self.head = spectral_norm(head) if spectral else head
        self.apply(dcgan_weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.body(x).flatten(1)
        out = self.head(feat).squeeze(1)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out
