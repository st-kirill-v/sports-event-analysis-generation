"""Inception Score and Frechet Inception Distance.

Inputs are RGB tensors in [-1, 1] (typical generator output) or PNG file lists.
InceptionV3 expects 299x299 RGB with ImageNet normalization, so we resize and
re-normalize inside this module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3


# ----------------------------------------------------------------- preprocess

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _prep_for_inception(x: torch.Tensor) -> torch.Tensor:
    """x: (B, 3, H, W) in [-1,1] or [0,1] -> (B, 3, 299, 299) normalized."""
    if x.dtype != torch.float32:
        x = x.float()
    if x.min() < 0.0:
        x = (x + 1.0) / 2.0
    if x.size(1) == 1:                       # grayscale safety net
        x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    mean = _IMAGENET_MEAN.to(x.device)
    std = _IMAGENET_STD.to(x.device)
    return (x - mean) / std


# ----------------------------------------------------------------- inception

class _InceptionFeatures(nn.Module):
    """Returns (softmax 1000-d, pool 2048-d) for IS + FID."""

    def __init__(self) -> None:
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        net = inception_v3(weights=weights, aux_logits=True)
        net.eval()
        self.fc = net.fc
        net.fc = nn.Identity()
        self.net = net

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pool = self.net(x)
        if isinstance(pool, tuple):
            pool = pool[0]
        logits = self.fc(pool)
        return logits, pool


# ----------------------------------------------------------------- datasets

class _TensorDataset(Dataset):
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def __len__(self) -> int:
        return self.tensor.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensor[idx]


class _PngFolderDataset(Dataset):
    def __init__(self, files: list[Path]) -> None:
        self.files = list(files)
        self.tf = transforms.Compose([
            transforms.ToTensor(),                      # [0,1]
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),  # [-1,1]
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.files[idx]) as img:
            return self.tf(img.convert("RGB"))


def _batched_features(
    model: _InceptionFeatures,
    loader: Iterable[torch.Tensor],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    probs_all: list[np.ndarray] = []
    pool_all: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        batch = _prep_for_inception(batch)
        logits, pool = model(batch)
        probs_all.append(F.softmax(logits, dim=1).cpu().numpy())
        pool_all.append(pool.cpu().numpy())
    return np.concatenate(probs_all, axis=0), np.concatenate(pool_all, axis=0)


# ----------------------------------------------------------------- metrics

def inception_score(probs: np.ndarray, splits: int = 10) -> tuple[float, float]:
    n = probs.shape[0]
    splits = max(1, min(splits, n))
    scores = []
    for i in range(splits):
        part = probs[i * n // splits : (i + 1) * n // splits]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores.append(float(np.exp(np.mean(np.sum(kl, axis=1)))))
    return float(np.mean(scores)), float(np.std(scores))


def frechet_distance(act_real: np.ndarray, act_fake: np.ndarray, eps: float = 1e-6) -> float:
    from scipy import linalg

    mu_r, mu_f = act_real.mean(axis=0), act_fake.mean(axis=0)
    sigma_r = np.cov(act_real, rowvar=False)
    sigma_f = np.cov(act_fake, rowvar=False)

    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sigma_r.dot(sigma_f), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_r.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_r + offset).dot(sigma_f + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_r) + np.trace(sigma_f) - 2.0 * np.trace(covmean))


# ----------------------------------------------------------------- public API

def compute_metrics(
    real_images: list[Path] | torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device,
    batch_size: int = 32,
    is_splits: int = 10,
) -> dict:
    """Compute IS (on fakes) and FID (real vs fake).

    real_images: PNG path list (loaded lazily) or pre-built (N, 3, H, W) tensor in [-1, 1].
    fake_images: tensor (N, 3, H, W) in [-1, 1].
    """
    model = _InceptionFeatures().to(device).eval()

    if isinstance(real_images, torch.Tensor):
        real_loader: DataLoader = DataLoader(_TensorDataset(real_images), batch_size=batch_size)
    else:
        real_loader = DataLoader(_PngFolderDataset(real_images), batch_size=batch_size, num_workers=0)
    fake_loader = DataLoader(_TensorDataset(fake_images), batch_size=batch_size)

    _, act_real = _batched_features(model, real_loader, device)
    probs_fake, act_fake = _batched_features(model, fake_loader, device)

    is_mean, is_std = inception_score(probs_fake, splits=is_splits)
    fid = frechet_distance(act_real, act_fake)

    return {
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
        "fid": fid,
        "n_real": int(act_real.shape[0]),
        "n_fake": int(act_fake.shape[0]),
    }
