"""Local CPU smoke training for the trajectory DCGAN.

Real training is done on Kaggle GPU (kaggle_notebook_2.ipynb); this script exists
so we can confirm the code runs end-to-end before uploading.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import TrajectoryDataset
from models.gan import DCGAN, GANConfig
from utils.visualization import save_grid


def _list_files(data_dir: Path, split_file: Path | None) -> list[Path]:
    if split_file and split_file.exists():
        return [Path(line.strip()) for line in split_file.read_text().splitlines() if line.strip()]
    return sorted(data_dir.glob("*.png"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--split-file", type=Path, default=Path("splits/train.txt"))
    ap.add_argument("--checkpoints-dir", type=Path, default=Path(r"D:/nba-motions-data/checkpoints"))
    ap.add_argument("--samples-dir", type=Path, default=Path(r"D:/nba-motions-data/samples"))
    ap.add_argument("--logs-dir", type=Path, default=Path(r"D:/nba-motions-data/logs"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--z-dim", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit-batches", type=int, default=3, help=">0 caps batches per epoch (smoke test)")
    ap.add_argument("--checkpoint-every", type=int, default=1)
    ap.add_argument("--sample-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cpu")    # local: force CPU
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)

    files = _list_files(args.data_dir, args.split_file)
    if not files:
        print(f"No images found at {args.data_dir} / {args.split_file}", file=sys.stderr)
        return 1
    print(f"Dataset size: {len(files)}")

    ds = TrajectoryDataset(files)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True, pin_memory=False)

    cfg = GANConfig(z_dim=args.z_dim, lr_g=args.lr, lr_d=args.lr)
    gan = DCGAN(cfg, device=device)
    print(f"G params: {sum(p.numel() for p in gan.G.parameters()):,}")
    print(f"D params: {sum(p.numel() for p in gan.D.parameters()):,}")

    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    args.samples_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.logs_dir / "train_log.csv"
    log_fh = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log_fh)
    if log_path.stat().st_size == 0:
        writer.writerow(["epoch", "iter", "loss_d", "loss_g", "t"])

    fixed_z = torch.randn(16, cfg.z_dim, device=device)

    global_iter = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        d_acc = g_acc = 0.0
        steps = 0
        for batch_idx, real in enumerate(loader):
            if args.limit_batches and batch_idx >= args.limit_batches:
                break
            loss_d = gan.step_d(real)
            loss_g = gan.step_g(real.size(0))
            d_acc += loss_d
            g_acc += loss_g
            steps += 1
            global_iter += 1
            writer.writerow([epoch, global_iter, f"{loss_d:.4f}", f"{loss_g:.4f}", f"{time.time():.1f}"])
            print(f"  epoch {epoch} batch {batch_idx} loss_d={loss_d:.4f} loss_g={loss_g:.4f}", flush=True)
        log_fh.flush()
        dt = time.time() - t0
        print(f"== epoch {epoch} done in {dt:.1f}s | avg d={d_acc/max(1,steps):.4f} g={g_acc/max(1,steps):.4f}", flush=True)

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            gan.G.eval()
            with torch.no_grad():
                fake = gan.G(fixed_z)
            gan.G.train()
            save_grid(fake, args.samples_dir / f"epoch_{epoch:04d}.png", nrow=4, title=f"epoch {epoch}")
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            ckpt = args.checkpoints_dir / f"gan_epoch_{epoch:04d}.pt"
            torch.save(gan.state_dict(), ckpt)
            print(f"  saved {ckpt}")
    log_fh.close()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
