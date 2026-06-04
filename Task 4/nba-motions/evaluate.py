"""Compute Inception Score (IS) and Frechet Inception Distance (FID) for a checkpoint.

Also writes a side-by-side compare grid: 64 real test images vs 64 generated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import TrajectoryDataset
from generate import load_model
from utils.metrics import compute_metrics
from utils.visualization import save_compare_grid


def _read_split(p: Path) -> list[Path]:
    return [Path(line.strip()) for line in p.read_text().splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--real-split", required=True, type=Path)
    ap.add_argument("--n-fake", type=int, default=1000)
    ap.add_argument("--n-real", type=int, default=0, help="0 = whole split")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path(r"D:/nba-motions-data/results/metrics.json"))
    ap.add_argument("--compare-grid", type=Path, default=Path(r"D:/nba-motions-data/results/compare.png"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    gan = load_model(args.checkpoint, device)

    real_files = _read_split(args.real_split)
    if args.n_real and args.n_real < len(real_files):
        real_files = real_files[: args.n_real]
    print(f"real: {len(real_files)} images")

    with torch.no_grad():
        z = torch.randn(args.n_fake, gan.cfg.z_dim, device=device)
        chunks = []
        for i in range(0, args.n_fake, args.batch_size):
            chunks.append(gan.G(z[i : i + args.batch_size]).cpu())
        fakes = torch.cat(chunks, dim=0)
    print(f"fakes: {fakes.shape[0]} images")

    metrics = compute_metrics(real_files, fakes, device=device, batch_size=args.batch_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.out}")

    real_ds = TrajectoryDataset(real_files[:64])
    real_batch = torch.stack([real_ds[i] for i in range(min(64, len(real_ds)))])
    args.compare_grid.parent.mkdir(parents=True, exist_ok=True)
    save_compare_grid(real_batch, fakes[:64], args.compare_grid, nrow=8)
    print(f"wrote {args.compare_grid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
