"""
Quickly overlay canonical-view landmarks on RGB frames for a few seeds.

Example:
  python visualise_landmark_and_rgb_canonical.py --seeds_csv rankedseedsall.csv --num_seeds 5 --n_landmarks 10
"""


# python reward_model_training/reward_model_framework/core_modules/data/create_train_data/visualise_landmark_and_rgb_canonical.py \
#   --seeds_csv reward_model_training/reward_model_framework/core_modules/data/create_train_data/rankedseedsall.csv \
#   --num_seeds 3 --n_landmarks 5

# python reward_model_training/reward_model_framework/core_modules/data/create_train_data/visualise_landmark_and_rgb_canonical.py \
#   --seeds_csv reward_model_training/reward_model_framework/core_modules/data/create_train_data/rankedseedsall.csv \
#   --num_seeds 3 --n_landmarks 98

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import autoroot  # noqa: F401
import numpy as np
import torch

from core_modules.data.create_train_data import generation_utils as gen_utils
from core_modules.data.create_train_data.synthesize_landmarks import (
    CANONICAL_VIEW_IDX,
    save_landmark_overlay_for_seed,
)


def _load_seeds_from_csv(csv_path: Path) -> List[int]:
    try:
        import pandas as pd

        df = pd.read_csv(csv_path, index_col=0)
        seeds: List[int] = []
        for col in df.columns:
            seeds.extend(df[col].dropna().astype(int).tolist())
        return seeds
    except Exception:
        return torch.as_tensor(np.loadtxt(csv_path, delimiter=",", dtype=int)).flatten().int().tolist()


def visualise_seeds(
    seeds: Iterable[int],
    n_landmarks: int,
    rgb_dir: Optional[Path],
    landmarks_dir: Optional[Path],
    output_dir: Path,
    max_seeds: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, seed in enumerate(seeds):
        if i >= max_seeds:
            break
        try:
            out_path = save_landmark_overlay_for_seed(
                seed,
                n=n_landmarks,
                rgb_dir=rgb_dir,
                landmarks_dir=landmarks_dir,
                output_dir=output_dir,
            )
            print(f"[OK] seed {seed} view {CANONICAL_VIEW_IDX} -> {out_path}")
        except Exception as exc:
            print(f"[WARN] seed {seed} skipped: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualise canonical-view landmarks over RGB for a few seeds.")
    parser.add_argument("--seeds_csv", type=Path, default=Path(__file__).parent / "rankedseedsall.csv")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of seeds to visualise.")
    parser.add_argument("--n_landmarks", type=int, default=98, help="Top-N landmarks to draw (e.g., 2 or 3 for spot checks).")
    parser.add_argument(
        "--rgb_dir",
        type=Path,
        default=None,
        help="Directory containing triple_rgb_s_* images. Defaults to RWD_DATA_DIR for each seed if not provided.",
    )
    parser.add_argument(
        "--landmarks_dir",
        type=Path,
        default=None,
        help="Directory containing *_lmks_*.pt files. Defaults to rgb_dir.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=gen_utils.DEFAULT_CHECK_DIR,
        help="Where to write the overlaid JPGs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = _load_seeds_from_csv(args.seeds_csv)
    rgb_dir = args.rgb_dir if args.rgb_dir is not None else None
    landmarks_dir = args.landmarks_dir if args.landmarks_dir is not None else rgb_dir
    visualise_seeds(
        seeds=seeds,
        n_landmarks=args.n_landmarks,
        rgb_dir=rgb_dir,
        landmarks_dir=landmarks_dir,
        output_dir=args.output_dir,
        max_seeds=args.num_seeds,
    )


if __name__ == "__main__":
    main()
