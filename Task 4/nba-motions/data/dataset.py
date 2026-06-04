"""Torch dataset for rendered 64x128 RGB trajectory PNGs."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TrajectoryDataset(Dataset):
    """Loads 64x128 RGB PNGs and returns tensors in [-1, 1]."""

    def __init__(self, files: list[Path] | Path | str) -> None:
        if isinstance(files, (str, Path)):
            files = [Path(line.strip()) for line in Path(files).read_text().splitlines() if line.strip()]
        self.files = [Path(p) for p in files]
        self.tf = transforms.Compose([
            transforms.ToTensor(),                          # [0, 1]
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),    # [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.files[idx]) as img:
            return self.tf(img.convert("RGB"))
