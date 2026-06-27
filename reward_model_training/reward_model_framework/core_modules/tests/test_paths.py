import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import autoroot  # noqa: F401

from core_modules.utils import camera_utils, depth_to_pcd, meshing_utils

import core_modules


def test_minimal_dataset_class_instantiates():
    minimal_dclass = core_modules.data.dset_loaders.dset_single_stream_ordered_minimal(
        all_combined_rankings=[-1],
        dtype="point_cloud_entire",
        ddir_func=core_modules.data.misc_small_utils.ddir_func,
        seed_func=core_modules.data.misc_small_utils.seed_func_default,
        include_goodseed=False,
        dset_version="three",
    )

    assert minimal_dclass is not None
