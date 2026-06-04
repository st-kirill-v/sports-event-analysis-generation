from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--target-games",
        "400",
        "--max-archives",
        "900",
        "--max-events",
        "900",
        *sys.argv[1:],
    ]
    runpy.run_path(
        str(Path(__file__).resolve().parent / "merge_nba_200.py"),
        run_name="__main__",
    )
