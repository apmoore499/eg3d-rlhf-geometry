"""Smoke test: each reward-model backbone trains+validates+tests for a few batches.

Runs the REAL pipeline (train_rwd_model) per backbone via a subprocess, with the
trainer limited to a couple of batches and a tiny data fraction. This exercises
the production path exactly as a real run would (collate -> run_forward_pass ->
train/val/test steps -> loss -> checkpoint), so a green run vindicates that
backbone end-to-end. No hand-rolled forward / on-the-fly shaping.

    cd reward_model_training/reward_model_framework
    python -m pytest core_modules/tests/test_backbone_train_smoke.py -v

Requires the data in eg3dredo_data and a GPU (the trainer uses cuda:0). Slow:
each backbone is a ~30-60s subprocess.
"""
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The six final experiments (the only configs kept after the Phase 3 prune).
# Each pins its own model + selected_dtype, covering all five live backbones:
# DepthMap (single + triple), Conv3D sigma field, CurveNet, PointNet, PointNet++.
EXPERIMENTS = [
    ("tdmap", "DepthMap/resnet50 <- triple depth map"),
    ("sdmap", "DepthMap/resnet50 <- single depth map"),
    ("sfield_256", "Conv3D/unet3d <- sigma field 256"),
    ("pcd_cvnet_point_cloud_entire", "CurveNet <- point_cloud_entire"),
    ("pcd_pnet_point_cloud_entire", "PointNetOne <- point_cloud_entire"),
    ("pcd_pnet2_point_cloud_entire", "PointNetTwo <- point_cloud_entire"),
]

OVERRIDES = [
    "using_wandb=false",
    "trainer.max_epochs=1",
    "trainer.limit_train_batches=2",
    "trainer.limit_val_batches=2",
    "data.dset_dict.proportion_of_data_to_use=0.02",
]


@pytest.mark.parametrize("experiment,label", EXPERIMENTS)
def test_backbone_train_smoke(experiment, label):
    cmd = [sys.executable, "-m", "core_modules.train_rwd_model", f"experiment={experiment}", *OVERRIDES]
    result = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, (
        f"{label} ({experiment}) failed (rc={result.returncode}):\n"
        f"--- stdout tail ---\n{result.stdout[-3000:]}\n"
        f"--- stderr tail ---\n{result.stderr[-3000:]}"
    )
