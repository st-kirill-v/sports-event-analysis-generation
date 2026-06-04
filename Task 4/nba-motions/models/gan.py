"""Composite GAN wrapper: a plain BCE DCGAN with optional hinge/spectral norm.

Defaults are the simple textbook setup: BCE loss, one-sided label smoothing
(real labels at 0.9), Adam(2e-4, beta1=0.5) for both networks. Switching
`loss="hinge"` and `spectral=True` turns on the more stable training regime.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .discriminator import Discriminator
from .generator import Generator


@dataclass
class GANConfig:
    z_dim: int = 100
    feature_g: int = 32
    feature_d: int = 32
    img_channels: int = 3
    img_h: int = 64
    img_w: int = 128
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    loss: str = "bce"                   # "bce" (default) or "hinge"
    spectral: bool = False              # spectral_norm on D (recommended for hinge)
    real_label_smooth: float = 0.9      # only used when loss == "bce"


class DCGAN:
    """Bundles generator, discriminator and Adam optimizers."""

    def __init__(self, cfg: GANConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

        self.G = Generator(
            z_dim=cfg.z_dim, feature_g=cfg.feature_g, out_channels=cfg.img_channels
        ).to(device)
        # BCE wants a probability (sigmoid); hinge wants a raw score.
        self.D = Discriminator(
            in_channels=cfg.img_channels,
            feature_d=cfg.feature_d,
            use_sigmoid=(cfg.loss == "bce"),
            spectral=cfg.spectral,
        ).to(device)

        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))
        self._bce = nn.BCELoss() if cfg.loss == "bce" else None

    # ------------------------------------------------------------------ utils

    def sample_z(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.cfg.z_dim, device=self.device)

    # ------------------------------------------------------------------ steps

    def step_d(self, real: torch.Tensor) -> float:
        bs = real.size(0)
        real = real.to(self.device, non_blocking=True)

        self.opt_d.zero_grad(set_to_none=True)
        z = self.sample_z(bs)
        fake = self.G(z).detach()

        d_real = self.D(real)
        d_fake = self.D(fake)

        if self.cfg.loss == "hinge":
            loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
        else:  # bce
            real_lbl = torch.full_like(d_real, self.cfg.real_label_smooth)
            fake_lbl = torch.zeros_like(d_fake)
            loss = self._bce(d_real, real_lbl) + self._bce(d_fake, fake_lbl)

        loss.backward()
        self.opt_d.step()
        return float(loss.item())

    def step_g(self, batch_size: int) -> float:
        self.opt_g.zero_grad(set_to_none=True)
        z = self.sample_z(batch_size)
        fake = self.G(z)
        d_fake = self.D(fake)

        if self.cfg.loss == "hinge":
            loss = -d_fake.mean()
        else:  # bce
            target = torch.ones_like(d_fake)
            loss = self._bce(d_fake, target)

        loss.backward()
        self.opt_g.step()
        return float(loss.item())

    # ------------------------------------------------------------------ io

    def state_dict(self) -> dict:
        return {
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "cfg": self.cfg.__dict__,
        }

    def load_state_dict(self, state: dict) -> None:
        self.G.load_state_dict(state["G"])
        self.D.load_state_dict(state["D"])
        if "opt_g" in state:
            self.opt_g.load_state_dict(state["opt_g"])
        if "opt_d" in state:
            self.opt_d.load_state_dict(state["opt_d"])
