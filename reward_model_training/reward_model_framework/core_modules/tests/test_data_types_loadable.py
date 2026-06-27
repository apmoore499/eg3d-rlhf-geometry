"""Smoke test: every priority representation (dtype) is loadable for a real seed.

Confirms the reward-model data pipeline can obtain each data type from the
shared data dir (RWD_DATA_DIR, default ~/Documents/eg3dredo_data) — either as a
file-backed tensor (sigma_field_*, *_dmap) or derived on the fly (point clouds
from the depth map). Run:

    cd reward_model_training/reward_model_framework
    python -m pytest core_modules/tests/test_data_types_loadable.py -v
"""
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import autoroot  # noqa: F401

from core_modules.data.dset_loaders import dset_single_stream_ordered_minimal
from core_modules.data.misc_small_utils import ddir_func, seed_func_default

# A regular (non-truncated) ranked seed; goodmesh seeds live at 100000-100999.
SEED = 28852

# Representations whose source data is present in eg3dredo_data and that the
# main reward-model pipeline uses. (The stored file-based point clouds are
# intentionally absent — never re-synthed into the shared data dir after the disk
# wipe; sigma_field_256/128 + the depth maps cover the live work.)
PRIORITY_DTYPES = [
    "sigma_field_256",
    "sigma_field_128",
    "single_dmap",
    "triple_dmap",
    "point_cloud_entire",          # derived from the depth map (16384 pts)
    "aw98_patch_rgb_4region_32",   # 4 regions, 32x32 rgb patches (was pcd_centroids)
    "aw98_3d_lmks",                # 98 keypoints sampled from the depth-map PCD
    "aw98_patch_geom_nose_8",      # nose 8x8 geometry patch (was centroids_98_patch_88)
    "aw98_patch_normals_nose_8",   # nose 8x8 + normals (was centroids_98_patch_88_normals)
    "aw98_patch_geom_all98_8",     # all 98 keypoints, 8x8 geometry patches
    "canonical_rgb_lmks_98",
    "pcd_nose_combined",               # fixed: self.center_points/downsample -> self.ept.*
    "sigma_field_64",                  # derived from the full 128^3 volume by trilinear downsampling
]


def _load(dtype: str, seed: int, map_on: str = "cpu"):
    rankings = torch.tensor([[seed, -1, -1, -1, -1]])
    ds = dset_single_stream_ordered_minimal(
        all_combined_rankings=rankings,
        dtype=dtype,
        ddir_func=ddir_func,
        seed_func=seed_func_default,
        include_goodseed=False,
        map_on=map_on,
    )
    return ds.return_single_example_by_seed(seed)


@pytest.mark.parametrize("dtype", PRIORITY_DTYPES)
def test_dtype_loads(dtype):
    sample = _load(dtype, SEED)
    assert sample is not None, f"{dtype}: returned None"
    assert hasattr(sample, "shape"), f"{dtype}: no .shape ({type(sample)})"
    assert sample.numel() > 0, f"{dtype}: empty tensor"
