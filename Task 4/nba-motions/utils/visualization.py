"""Image grid plotting helpers (RGB)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def _to_numpy_grid(images: torch.Tensor, nrow: int, padding: int = 2) -> np.ndarray:
    """images: (B, 3, H, W) in [-1, 1] -> HxWx3 numpy in [0, 1]."""
    if images.dim() == 3:
        images = images.unsqueeze(0)
    images = images.detach().cpu().float()
    images = (images.clamp(-1.0, 1.0) + 1.0) / 2.0
    grid = make_grid(images, nrow=nrow, padding=padding, pad_value=1.0)
    return grid.numpy().transpose(1, 2, 0)


def save_grid(images: torch.Tensor, path: str | Path, nrow: int = 8, title: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = _to_numpy_grid(images, nrow=nrow)
    n = images.size(0) if images.dim() == 4 else 1
    nrows = max(1, n // nrow)
    fig, ax = plt.subplots(figsize=(2 * nrow, nrows))
    ax.imshow(arr, vmin=0.0, vmax=1.0)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_compare_grid(real: torch.Tensor, fake: torch.Tensor, path: str | Path, nrow: int = 8) -> None:
    """Save reals (left) and fakes (right) side-by-side, nrow x nrow each."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = nrow * nrow
    real_arr = _to_numpy_grid(real[:n], nrow=nrow)
    fake_arr = _to_numpy_grid(fake[:n], nrow=nrow)
    fig, axes = plt.subplots(1, 2, figsize=(2 * 2 * nrow, nrow))
    for ax, img, title in zip(axes, [real_arr, fake_arr], ["real", "generated"]):
        ax.imshow(img, vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
