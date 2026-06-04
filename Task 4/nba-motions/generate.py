"""Generate trajectory PNGs from a trained GAN checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from models.gan import DCGAN, GANConfig
from utils.visualization import save_grid


def load_model(ckpt_path: Path, device: torch.device) -> DCGAN:
    state = torch.load(ckpt_path, map_location=device)
    cfg_dict = state.get("cfg", {})
    cfg = GANConfig(**{k: v for k, v in cfg_dict.items() if k in GANConfig.__dataclass_fields__})
    gan = DCGAN(cfg, device=device)
    gan.load_state_dict(state)
    gan.G.eval()
    return gan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path(r"D:/nba-motions-data/output/generated"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    gan = load_model(args.checkpoint, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        z = torch.randn(args.n, gan.cfg.z_dim, device=device)
        fake = gan.G(z)
        imgs = ((fake.clamp(-1, 1) + 1) * 127.5).byte().cpu()

    for i in range(args.n):
        arr = imgs[i].permute(1, 2, 0).numpy()
        Image.fromarray(arr, mode="RGB").save(args.out_dir / f"gen_{i:05d}.png")
    print(f"Saved {args.n} PNGs to {args.out_dir}")

    if args.grid:
        side = int(round(args.n ** 0.5))
        save_grid(fake[: side * side], args.out_dir / "grid.png", nrow=side, title=f"generated x{side*side}")
        print(f"Saved grid -> {args.out_dir / 'grid.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
