#!/usr/bin/env bash
set -euo pipefail

# End-to-end test: depth-map loader -> DepthMap model (global vec + heads) for
# single_dmap and triple_dmap using stubbed pretrained backbones and synthetic data.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="$(cd "${FRAMEWORK_ROOT}/../.." && pwd)"

export PYTHONPATH="${FRAMEWORK_ROOT}:${PROJECT_ROOT}"
export PROJECT_ROOT

python - <<'PY'
import os
import sys
import types
import warnings
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
warnings.filterwarnings("ignore", message="The torchvision.datapoints.*Beta")

# Stubs to avoid optional deps
tb_stub = types.ModuleType("tensorboard")
tb_stub.compat = types.ModuleType("tensorboard.compat")
sys.modules.setdefault("tensorboard", tb_stub)
sys.modules.setdefault("tensorboard.compat", tb_stub.compat)
sys.modules.setdefault("tensorboard.compat.notf", tb_stub.compat)

if "wandb" not in sys.modules:
    sys.modules["wandb"] = types.SimpleNamespace(init=lambda *a, **k: None)

if "lightning" not in sys.modules:
    class _DummyLightningModule(nn.Module):
        def save_hyperparameters(self, *args, **kwargs):
            self.hparams = types.SimpleNamespace(**(kwargs or {}))
    lightning_stub = types.ModuleType("lightning")
    lightning_stub.LightningModule = _DummyLightningModule
    lightning_stub.LightningDataModule = object
    lightning_stub.Trainer = object
    lightning_stub.Callback = object
    sys.modules["lightning"] = lightning_stub
    pytorch_stub = types.ModuleType("lightning.pytorch")
    pytorch_stub.loggers = types.SimpleNamespace(Logger=object)
    pytorch_stub.callbacks = types.SimpleNamespace(Callback=object)
    sys.modules["lightning.pytorch"] = pytorch_stub
    sys.modules["lightning.pytorch.loggers"] = pytorch_stub.loggers
    sys.modules["lightning.pytorch.callbacks"] = pytorch_stub.callbacks

# Stub torch.hub pretrained load
class _StubResnet(nn.Module):
    def __init__(self, feat_ch=2048):
        super().__init__()
        self.avgpool = nn.Identity()
        self.feat_ch = feat_ch

    def forward(self, x):
        b = x.shape[0]
        pooled = torch.ones((b, self.feat_ch, 1, 1), device=x.device, dtype=x.dtype)
        return pooled

torch.hub.load = lambda *args, **kwargs: _StubResnet()

from core_modules.data.aug_and_tforms.augtforms_dmap import DepthMapPreprocessor
from core_modules.data.dset_loaders import dset_single_stream_ordered_minimal
from core_modules.models.modules_depthmap import DepthMap

def make_model_cfg(n_dmaps):
    emb = 2048
    gf = 512
    return OmegaConf.create(
        {
            "_target_": "core_modules.models.modules_depthmap.DepthMap",
            "n_dmaps": n_dmaps,
            "global_feature_size": gf,
            "name": f"DepthMap_resnet50_{n_dmaps}",
            "return_global_embedding": True,
            "compile": False,
            "act_type": "cos",
            "optimizer": {"_target_": "torch.optim.SGD", "_partial_": True, "lr": 1e-3},
            "scheduler": None,
            "external": {
                "_target_": "core_modules.models.external.conv2d_backbone.resnet50_to_2048_dmap",
                "maps_transforms": {"laplace": [False], "hipass": [False], "normalise_sides_crop": [False], "run_lowpass_sides": [False]},
                "transform_stats_to_resnet": False,
            },
            "mlp_global": {
                "_target_": "core_modules.models.utils_base.MLP",
                "input_size": emb * n_dmaps,
                "output_size": gf,
                "hidden_sizes": [512, 512],
                "dropout_rate": 0.0,
                "use_dropout": False,
                "normalisation_type": "none",
                "activation_type": "cos",
                "norm_first_layer": False,
                "residual": True,
            },
            "mlp_pairs": {
                "_target_": "core_modules.models.utils_base.MLP",
                "input_size": gf * 2,
                "output_size": 2,
                "hidden_sizes": [256, 128],
                "dropout_rate": 0.0,
                "use_dropout": False,
                "normalisation_type": "none",
                "activation_type": "cos",
                "norm_first_layer": False,
                "residual": True,
            },
            "mlp_BT": {
                "_target_": "core_modules.models.utils_base.MLP",
                "input_size": gf,
                "output_size": 1,
                "hidden_sizes": [256, 128],
                "dropout_rate": 0.0,
                "use_dropout": False,
                "normalisation_type": "none",
                "activation_type": "cos",
                "norm_first_layer": False,
                "residual": True,
            },
            "mlp_scalar_rwd": {
                "_target_": "core_modules.models.utils_base.MLP",
                "input_size": gf,
                "output_size": 1,
                "hidden_sizes": [256, 128],
                "dropout_rate": 0.0,
                "use_dropout": False,
                "normalisation_type": "none",
                "activation_type": "cos",
                "norm_first_layer": False,
                "residual": True,
            },
            "loss": {
                "return_global_embedding": True,
                "lambda_abs_sym_loss": 0.0,
                "lambda_BT": 0.0,
                "lambda_pairs": 0.0,
                "lambda_scalar_rwd": 0.0,
                "lambda_reg_rwd_vals": 0.0,
                "lambda_agg_features_l2": 0.0,
            },
            "swa": False,
        }
    )

def run_case(n_dmaps, dtype):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        seed = 0
        fn = tmpdir / f"triple_dmap_s_{seed}.pt"
        # store list of depth maps (3 views)
        depth_maps = [torch.full((128, 128), i + 2.5, dtype=torch.float32) for i in range(3)]
        torch.save(depth_maps, fn)

        aug = DepthMapPreprocessor(out_size=128, normalize_range=False, invert=False)
        ds = dset_single_stream_ordered_minimal(
            all_combined_rankings=[torch.tensor([seed])],
            dtype=dtype,
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
        x = data["files"][0]  # shape depends on dtype
        print(f"[{dtype}] loaded tensor shape:", tuple(x.shape))

        model = DepthMap(**OmegaConf.to_container(make_model_cfg(n_dmaps), resolve=True))
        model.eval()
        with torch.no_grad():
            feats = model.external(x)  # (B, emb * n_dmaps)
            gvec = model.forward_to_global_feature_vec(feats)
            scalar = model.forward_to_scalar_reward_from_single_global(gvec)
            bt = model.forward_to_BT_lambda_from_single_global(gvec)
            pairs = model.forward_from_cat_global_vectors(gvec, gvec)
        print(f"[{dtype}] feats {tuple(feats.shape)}, global vec {tuple(gvec.shape)}, scalar {tuple(scalar.shape)}, BT {tuple(bt.shape)}, pairs {tuple(pairs.shape)}")

print("==== Single dmap ====")
run_case(n_dmaps=1, dtype="single_dmap")
print("==== Triple dmap ====")
run_case(n_dmaps=3, dtype="triple_dmap")
PY
