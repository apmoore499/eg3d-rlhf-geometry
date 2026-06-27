import os
import sys
import types
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import autoroot  # noqa: F401


# def _install_lightning_stub():
#     """Ensure lightning imports resolve to lightweight stubs."""
#     lightning_mod = types.ModuleType("lightning")
#     lightning_mod.Callback = object
#     pytorch_mod = types.ModuleType("lightning.pytorch")
#     pytorch_mod.Callback = object
#     loggers_mod = types.ModuleType("lightning.pytorch.loggers")
#     loggers_mod.Logger = object
#     callbacks_mod = types.ModuleType("lightning.pytorch.callbacks")
#     callbacks_mod.Callback = object
#     lightning_mod.pytorch = pytorch_mod
#     pytorch_mod.loggers = loggers_mod
#     pytorch_mod.callbacks = callbacks_mod
#     sys.modules["lightning"] = lightning_mod
#     sys.modules["lightning.pytorch"] = pytorch_mod
#     sys.modules["lightning.pytorch.loggers"] = loggers_mod
#     sys.modules["lightning.pytorch.callbacks"] = callbacks_mod


# Only run when explicitly enabled to avoid heavy imports in constrained envs.
# if not os.environ.get("RUN_RLHF_UTIL_SMOKE"):
#     pytest.skip("set RUN_RLHF_UTIL_SMOKE=1 to run utils smoke tests", allow_module_level=True)

# sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# _install_lightning_stub()

from core_modules.utils import camera_utils, depth_to_pcd, meshing_utils


def test_depth_to_pcd_with_stub_ray_sampler():
    # Depth map 2x2 -> 4 rays
    depth = torch.ones(1, 2, 2)

    def stub_ray_sampler(cam2world, intrinsics, nrs):
        origins = torch.zeros(1, nrs * nrs, 3)
        directions = torch.ones(1, nrs * nrs, 3)
        return origins, directions

    pts = depth_to_pcd.modules_depthmap_to_pcd_from_image(
        modules_depthmap_image=depth,
        ray_sampler=stub_ray_sampler,
        gen_c=None,
        nrs=2,
        lim_pts=None,
    )
    # All four rays should produce a point
    assert pts.shape == (4, 3)


def test_meshing_utils_create_samples_small():
    samples, voxel_origin, voxel_size = meshing_utils.create_samples(N=4, voxel_origin=[0, 0, 0], cube_length=2.0)
    # shape: (1, N^3, 3)
    assert samples.shape == (1, 64, 3)
    assert tuple(voxel_origin.tolist()) == (-1.0, -1.0, -1.0)
    assert voxel_size == 2.0 / 3.0
    assert torch.isfinite(samples).all()
