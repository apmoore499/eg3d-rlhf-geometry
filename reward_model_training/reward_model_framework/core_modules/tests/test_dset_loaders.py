"""Smoke test for dataset loaders (landmarks + core tensors).
Loads a single seed for specified dtypes and prints tensor shapes."""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import autoroot  # noqa: F401

from core_modules.data.dset_loaders import dset_single_stream_ordered_minimal
from core_modules.data.misc_small_utils import ddir_func, seed_func_default


def run_smoke(dtype: str, seed: int, map_on: str = "cpu") -> bool:
    rankings = torch.tensor([[seed, -1, -1, -1, -1]])
    ds = dset_single_stream_ordered_minimal(
        all_combined_rankings=rankings,
        dtype=dtype,
        ddir_func=ddir_func,
        seed_func=seed_func_default,
        include_goodseed=False,
        map_on=map_on,
    )

    try:
        sample = ds.return_single_example_by_seed(seed)
        shape = getattr(sample, "shape", None)
        print(f"[OK] dtype={dtype} seed={seed} shape={shape}")
        return True
    except Exception:
        print(f"[FAIL] dtype={dtype} seed={seed}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Smoke test dataset loaders (landmarks + core dtypes).")
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=None,
        help="List of dtypes to load. Defaults to a curated subset when omitted.",
    )
    parser.add_argument("--seed", type=int, default=28852, help="Seed to load.")
    parser.add_argument("--map_on", type=str, default="cpu", help="Device for torch.load tensors (default: cpu).")
    args = parser.parse_args()

    # Default: exercise a curated subset of dtypes.
    dtypes = args.dtypes
    if dtypes is None:
        dtypes = [
            "point_cloud_entire",
            "sigma_field_256",
            "triple_dmap",
            "triple_rgb",
            "single_rgb",
            "single_dmap",
            "triple_rgb_lmks_98",
            "canonical_rgb_lmks_98",
            "aw98_3d_lmks",
        ]

    all_ok = True
    for dt in dtypes:
        all_ok = run_smoke(dt, args.seed, map_on=args.map_on) and all_ok

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
