#!/usr/bin/env bash
set -euo pipefail

# Test: exercise depth-map preprocessors and the resnet50 depth model
# with lightweight stubs (no network downloads).

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
import warnings
import tempfile
from pathlib import Path
from types import SimpleNamespace

import yaml
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
warnings.filterwarnings("ignore", message="The torchvision.datapoints.*Beta")

# Stub tensorboard and lightning to avoid heavy optional deps during import.
tb_stub = types.ModuleType("tensorboard")
tb_stub.compat = types.ModuleType("tensorboard.compat")
sys.modules.setdefault("tensorboard", tb_stub)
sys.modules.setdefault("tensorboard.compat", tb_stub.compat)
sys.modules.setdefault("tensorboard.compat.notf", tb_stub.compat)

# Stub wandb to avoid protobuf/tensorboard pulls.
if "wandb" not in sys.modules:
    wandb_stub = types.SimpleNamespace(init=lambda *args, **kwargs: None)
    sys.modules["wandb"] = wandb_stub

if "lightning" not in sys.modules:
    class _DummyLightningModule(nn.Module):
        def save_hyperparameters(self, *args, **kwargs):
            params = kwargs if kwargs else {}
            self.hparams = SimpleNamespace(**params)

    class _DummyLightningDataModule:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

    lightning_stub = types.ModuleType("lightning")
    lightning_stub.LightningModule = _DummyLightningModule
    lightning_stub.LightningDataModule = _DummyLightningDataModule
    lightning_stub.Trainer = _DummyTrainer
    lightning_stub.Callback = object

    pytorch_stub = types.ModuleType("lightning.pytorch")
    pytorch_loggers_stub = types.ModuleType("lightning.pytorch.loggers")
    pytorch_callbacks_stub = types.ModuleType("lightning.pytorch.callbacks")

    class _DummyLogger:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyCallback:
        def __init__(self, *args, **kwargs):
            pass

    pytorch_loggers_stub.Logger = _DummyLogger
    pytorch_callbacks_stub.Callback = _DummyCallback
    pytorch_stub.loggers = pytorch_loggers_stub
    pytorch_stub.callbacks = pytorch_callbacks_stub

    sys.modules["lightning"] = lightning_stub
    sys.modules["lightning.pytorch"] = pytorch_stub
    sys.modules["lightning.pytorch.loggers"] = pytorch_loggers_stub
    sys.modules["lightning.pytorch.callbacks"] = pytorch_callbacks_stub

# Stub torch.hub.load to avoid downloading pretrained weights.
class _StubResnet(nn.Module):
    def __init__(self, feat_ch=2048):
        super().__init__()
        self.feat_ch = feat_ch
        self.avgpool = nn.Identity()

    def forward(self, x):
        b = x.shape[0]
        # Produce a fixed feature map to exercise downstream shape logic.
        pooled = torch.ones((b, self.feat_ch, 1, 1), device=x.device, dtype=x.dtype)
        return pooled.view(b, -1)

torch.hub.load = lambda *args, **kwargs: _StubResnet()

from core_modules.data.aug_and_tforms.augtforms_dmap import DepthMapPreprocessor, build_depthmap_preprocessors
from core_modules.data.dset_loaders import dset_single_stream_ordered_minimal
from core_modules.models.external.conv2d_backbone import resnet50_to_2048_dmap

maps_cfg = OmegaConf.create(
    {
        "hipass": [False, True],
        "laplace": [False, True],
        "normalise_sides_crop": [False, False],
        "run_lowpass_sides": [False, False],
    }
)

dmap = torch.rand(2, 2, 128, 128)
# ---------------- basic preprocess + forward ---------------- #
preprocessors = build_depthmap_preprocessors(
    maps_transforms=maps_cfg,
    out_size=224,
    normalize_range=True,
    invert=True,
    hp_scale=1.0,
    lp_scale=1.0,
)

processed = [p(dmap[:, i, :, :]) for i, p in enumerate(preprocessors)]
print("Preprocessors:", len(preprocessors))
for i, t in enumerate(processed):
    print(f"  view {i}: shape {tuple(t.shape)}, finite={torch.isfinite(t).all().item()}, min={t.min().item():.4f}, max={t.max().item():.4f}")

model = resnet50_to_2048_dmap(maps_transforms=maps_cfg, transform_stats_to_resnet=False)
model.eval()
with torch.no_grad():
    out = model(dmap)
print("resnet50_to_2048_dmap output:", tuple(out.shape), "finite:", torch.isfinite(out).all().item())
per_map = out.shape[1] // dmap.shape[1]
print(" per-map global feature dim:", per_map)

# ---------------- inspect experiment maps_transforms ---------------- #
framework_root = Path(os.environ["FRAMEWORK_ROOT"])
exp_cfg = framework_root / "core_modules/configs/experiment/single_depth_map_resnet_50.yaml"
with open(exp_cfg, "r") as f:
    exp = yaml.safe_load(f)

mt = exp.get("model", {}).get("external", {}).get("maps_transforms", {})
if isinstance(mt, str):
    aug_name = None
    for entry in exp.get("defaults", []):
        if isinstance(entry, dict):
            for k, v in entry.items():
                if k.replace(" ", "") in ("override/augmentations", "override/augmentations:"):
                    aug_name = v
                if "override /augmentations" in k:
                    aug_name = v
    if aug_name:
        aug_path = framework_root / f"core_modules/configs/data/augmentations/{aug_name}.yaml"
        if aug_path.exists():
            with open(aug_path, "r") as af:
                aug_cfg = yaml.safe_load(af)
            mt = aug_cfg.get("maps_transforms", {})
print("Experiment maps_transforms keys:", list(mt.keys()) if hasattr(mt, "keys") else mt, "hipass len:", len(mt.get("hipass", [])) if isinstance(mt, dict) else "n/a")

# ---------------- transform_stats_to_resnet=True ---------------- #
model_ts = resnet50_to_2048_dmap(maps_transforms=maps_cfg, transform_stats_to_resnet=True)
model_ts.eval()
with torch.no_grad():
    out_ts = model_ts(dmap)
print("resnet50_to_2048_dmap (transform_stats_to_resnet=True) output:", tuple(out_ts.shape), "finite:", torch.isfinite(out_ts).all().item())
print(" per-map global feature dim (stats_to_resnet):", out_ts.shape[1] // dmap.shape[1])

# ---------------- multi-view (3) preprocessors ---------------- #
maps_cfg3 = OmegaConf.create(
    {
        "hipass": [False, True, False],
        "laplace": [False, False, True],
        "normalise_sides_crop": [False, False, False],
        "run_lowpass_sides": [False, False, False],
    }
)
dmap3 = torch.rand(1, 3, 128, 128)
model3 = resnet50_to_2048_dmap(maps_transforms=maps_cfg3, transform_stats_to_resnet=False)
model3.eval()
with torch.no_grad():
    out3 = model3(dmap3)
print("resnet50_to_2048_dmap (3 views) output:", tuple(out3.shape), "finite:", torch.isfinite(out3).all().item())
print(" per-map global feature dim (3 views):", out3.shape[1] // dmap3.shape[1])

# ---------------- loader integration sanity ---------------- #
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    seed = 0
    fn = tmpdir / f"triple_dmap_s_{seed}.pt"
    depth_maps = [torch.full((128, 128), i, dtype=torch.float32) for i in range(3)]
    torch.save(depth_maps, fn)

    aug = DepthMapPreprocessor(out_size=128, normalize_range=False, invert=False)
    ds = dset_single_stream_ordered_minimal(
        all_combined_rankings=[np.array([seed])],
        dtype="single_dmap",
        ddir_func=lambda s: str(tmpdir),
        seed_func=lambda s: int(s),
        augmentations=aug,
        goodmesh_augment=None,
        dset_partition="train",
        include_goodseed=False,
    )
    ds.using_transform = True
    ds.transforms = aug
    o, data = ds[0]
    files = data["files"]
    print("dset_single_stream_ordered_minimal -> dtype:", o, "batch_len:", data["batch_len"], "seed:", data["ordered_seeds"].tolist())
    print(" single_dmap shape:", files[0].shape, "finite:", torch.isfinite(files[0]).all().item())

print("All depthmap test checks finished.")
PY
