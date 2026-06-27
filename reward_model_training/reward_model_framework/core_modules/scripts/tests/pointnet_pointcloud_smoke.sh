#!/usr/bin/env bash
set -euo pipefail

# Simple smoke test: run PointNetOne on random point clouds with and without
# the pcd_as_pt transforms to catch import/path issues.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="$(cd "${FRAMEWORK_ROOT}/../.." && pwd)"

export PYTHONPATH="${FRAMEWORK_ROOT}:${PROJECT_ROOT}"
export FRAMEWORK_ROOT
export PROJECT_ROOT

python - <<'PY'
import os
import sys
import types
from pathlib import Path

import torch
from omegaconf import OmegaConf
import hydra
import warnings
import importlib.metadata as importlib_metadata

# Provide missing stdlib helper to avoid tensorboard import errors in some envs.
if not hasattr(importlib_metadata, "packages_distributions"):
    importlib_metadata.packages_distributions = lambda: {}

# Stub tensorboard to prevent it from pulling in proto deps during light imports.
tb_stub = types.ModuleType("tensorboard")
tb_stub.compat = types.ModuleType("tensorboard.compat")
sys.modules.setdefault("tensorboard", tb_stub)
sys.modules.setdefault("tensorboard.compat", tb_stub.compat)
sys.modules.setdefault("tensorboard.compat.notf", tb_stub.compat)

warnings.filterwarnings("ignore", message="The torchvision.datapoints.*Beta")

# Stub lightning to avoid importing the real package (which may need semaphores)
# for this lightweight smoke test.
if "lightning" not in sys.modules:
    import torch.nn as nn
    import inspect

    class _DummyLightningModule(nn.Module):
        def save_hyperparameters(self, *args, **kwargs):
            params = {}
            frame = inspect.currentframe()
            if frame and frame.f_back and "kwargs" in frame.f_back.f_locals:
                params = frame.f_back.f_locals["kwargs"]
            self.hparams = types.SimpleNamespace(**params)

    lightning_stub = types.ModuleType("lightning")
    lightning_stub.LightningModule = _DummyLightningModule
    lightning_stub.Callback = object
    sys.modules["lightning"] = lightning_stub

# Stub core_modules.utils.pylogger_c to avoid pulling the full utils package (which imports lightning loggers).
import logging

pylogger_stub = types.SimpleNamespace(RankedLogger=lambda name, rank_zero_only=True: logging.getLogger(name))
utils_stub = types.ModuleType("core_modules.utils")
utils_stub.pylogger_c = pylogger_stub
framework_root = Path(os.environ["FRAMEWORK_ROOT"])
utils_stub.__path__ = [str(framework_root / "core_modules" / "utils")]
sys.modules.setdefault("core_modules.utils", utils_stub)
sys.modules.setdefault("core_modules.utils.pylogger_c", pylogger_stub)

project_root = Path(os.environ["PROJECT_ROOT"])
sys.path.insert(0, str(project_root))

# Minimal package stubs so we can import submodules without triggering heavy __init__ logic.
src_pkg = types.ModuleType("core_modules")
src_pkg.__path__ = [str(framework_root / "core_modules")]
sys.modules.setdefault("core_modules", src_pkg)

data_pkg = types.ModuleType("core_modules.data")
data_pkg.__path__ = [str(framework_root / "core_modules" / "data")]
sys.modules.setdefault("core_modules.data", data_pkg)

models_pkg = types.ModuleType("core_modules.models")
models_pkg.__path__ = [str(framework_root / "core_modules" / "models")]
sys.modules.setdefault("core_modules.models", models_pkg)

from core_modules.data.aug_and_tforms.augtforms_pointcloud import (
    modules_depthmap_to_xyz,
    downsample_pcd_points,
    mean_scale_pts,
    center_points,
)

try:
    import torchvision

    if hasattr(torchvision, "disable_beta_transforms_warning"):
        torchvision.disable_beta_transforms_warning()
except Exception:
    pass
from core_modules.models.modules_pointnet import PointNetOne

# Build transforms directly (mirror pcd_as_pt pipeline)
nrs = 128
depth_to_xyz = modules_depthmap_to_xyz(nrs=nrs)
downsample = downsample_pcd_points(n_points=2048)
center = center_points()
scale = mean_scale_pts()

depth = torch.rand(1, nrs, nrs)
with torch.no_grad():
    pcd = depth_to_xyz(depth)
    pcd = downsample(pcd)
    pcd = center(pcd)
    pcd = scale(pcd)

pcd = pcd.permute(0, 2, 1).contiguous()  # (B, 3, N) for PointNet


def mlp_cfg(input_size, output_size):
    return {
        "_target_": "core_modules.models.utils_base.MLP",
        "input_size": input_size,
        "output_size": output_size,
        "hidden_sizes": [64, 64],
        "dropout_rate": 0.0,
        "use_dropout": False,
        "normalisation_type": "none",
        "activation_type": "relu",
        "norm_first_layer": False,
        "residual": True,
    }


model_cfg = OmegaConf.create(
    {
        "name": "PointNetOne_smoke",
        "agg_type": "max",
        "spatial_transform_3d": False,
        "feature_transform": False,
        "normtype": "batch_norm",
        "act_type": "relu",
        "return_global_embedding": True,
        "external": None,
        "global_feature_size": 64,
        "mlp_global": mlp_cfg(1024, 64),
        "mlp_pairs": mlp_cfg(64 * 2, 2),
        "mlp_BT": mlp_cfg(64, 1),
        "mlp_scalar_rwd": mlp_cfg(64, 1),
        "optimizer": {"_target_": "torch.optim.SGD", "_partial_": True, "lr": 1e-3},
        "scheduler": None,
        "loss": {
            "return_global_embedding": True,
            "lambda_abs_sym_loss": 0.0,
            "lambda_BT": 0.0,
            "lambda_pairs": 0.0,
            "lambda_scalar_rwd": 0.0,
            "lambda_reg_rwd_vals": 0.0,
            "lambda_agg_features_l2": 0.0,
        },
        "compile": False,
    }
)

model = PointNetOne(**OmegaConf.to_container(model_cfg, resolve=True))
model.eval()

with torch.no_grad():
    out_with_tform = model.forward_to_global_feature_vec(pcd)
    out_raw = model.forward_to_global_feature_vec(torch.rand_like(pcd))

print("Transforms output shape:", pcd.shape)
print("PointNetOne with transforms:", out_with_tform.shape, "finite:", torch.isfinite(out_with_tform).all().item())
print("PointNetOne raw input:", out_raw.shape, "finite:", torch.isfinite(out_raw).all().item())
PY
