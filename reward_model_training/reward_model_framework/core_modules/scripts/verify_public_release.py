from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache" / "eg3d_rlhf_geometry" / "matplotlib"))

if not hasattr(importlib_metadata, "packages_distributions"):
    try:
        import importlib_metadata as importlib_metadata_backport

        importlib_metadata.packages_distributions = importlib_metadata_backport.packages_distributions
    except Exception:
        importlib_metadata.packages_distributions = lambda: {}

import hydra
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict

FRAMEWORK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = FRAMEWORK_ROOT / "core_modules" / "configs"
DEFAULT_EXPERIMENTS = [
    "sfield_256",
    "sdmap",
    "tdmap",
    "pcd_cvnet_point_cloud_entire",
    "pcd_pnet_point_cloud_entire",
    "pcd_pnet2_point_cloud_entire",
]

if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))

import autoroot  # noqa: F401

from core_modules.data.collate_data import CollateVariableShapeBatch
from core_modules.data.dset_loaders import dset_single_stream_ordered_minimal
from core_modules.data.misc_small_utils import ddir_func, seed_func_default
from core_modules.utils.reward_loading import (
    get_datatype_from_model_id,
    load_rwd_model_from_cfg,
    load_tune_augmentation_from_cfg,
)


class SkipStep(RuntimeError):
    pass


class SummaryWriter:
    def __init__(self, path: Path, seeds: Sequence[int], workdir: Path):
        self.path = path
        self.data = {
            "status": "running",
            "workdir": str(workdir),
            "seeds": list(seeds),
            "steps": [],
        }
        self.flush()

    def add(self, step: str, status: str, **details):
        row = {"step": step, "status": status}
        row.update(details)
        self.data["steps"].append(row)
        self.flush()

    def set_status(self, status: str):
        self.data["status"] = status
        self.flush()

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the maintained public-release verification checks.")
    parser.add_argument("--workdir", type=Path, default=REPO_ROOT / "release_verification_outputs" / "public_release_test")
    parser.add_argument("--seeds", type=str, default="28852,28853", help="Two ranked seeds for the test batch, e.g. '28852,28853'.")
    parser.add_argument("--truncation-psi", type=float, default=1.0)
    parser.add_argument("--generator-pkl", type=Path, default=None, help="External untuned EG3D generator checkpoint for data synthesis.")
    parser.add_argument("--reward-model-id", type=str, default="7wnzkgie", help="Released reward-model id to load for the checkpoint test.")
    parser.add_argument("--baseline-pkl", type=Path, default=None, help="Untuned baseline EG3D snapshot for the mesh-bank test.")
    parser.add_argument("--tuned-pkl", type=Path, default=None, help="Fine-tuned EG3D snapshot for the mesh-bank test.")
    parser.add_argument("--mesh-bank-start-seed", type=int, default=9100000)
    parser.add_argument("--mesh-bank-num-seeds", type=int, default=2)
    parser.add_argument("--skip-data-generation", action="store_true")
    parser.add_argument("--skip-loader-test", action="store_true")
    parser.add_argument("--skip-forward-test", action="store_true")
    parser.add_argument("--skip-released-checkpoint-test", action="store_true")
    parser.add_argument("--skip-mesh-bank", action="store_true")
    return parser.parse_args()


def parse_seed_pair(spec: str) -> List[int]:
    seeds = [int(part.strip()) for part in spec.split(",") if part.strip()]
    if len(seeds) != 2:
        raise ValueError(f"Expected exactly 2 comma-separated seeds, got: {spec}")
    if len(set(seeds)) != 2:
        raise ValueError(f"Seed pair must contain two distinct seeds, got: {spec}")
    return seeds


def run_cmd(cmd: Sequence[str], *, cwd: Path, env: Dict[str, str], label: str) -> None:
    print(f"\n[run] {label}")
    print(" ".join(str(c) for c in cmd))
    subprocess.run(list(map(str, cmd)), cwd=str(cwd), env=env, check=True)


def prepare_environment(args: argparse.Namespace, data_dir: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PROJECT_ROOT", str(REPO_ROOT))
    env.setdefault("MPLCONFIGDIR", str(args.workdir / "mplconfig"))
    env["RWD_DATA_DIR"] = str(data_dir)
    env["E3D_RLHF_SAVE_DIR"] = str(data_dir)
    env["E3D_RLHF_SIGMA_DATA_DIR"] = str(data_dir)
    env["E3D_RLHF_CHECK_DIR"] = str(data_dir / "checking")
    if args.generator_pkl is not None:
        env["E3D_RLHF_GENERATOR_PKL"] = str(args.generator_pkl)
        env["EG3D_RLHF_ORIG_PKL"] = str(args.generator_pkl)
    return env


def set_process_env(env: Dict[str, str]) -> None:
    for key, value in env.items():
        os.environ[key] = value


def instantiate_transform(cfg_node):
    if cfg_node is None:
        return torch.nn.Identity()
    return hydra.utils.instantiate(cfg_node)


def compose_experiment_cfg(experiment: str) -> DictConfig:
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="train", overrides=[f"experiment={experiment}", "using_wandb=false"])
    with open_dict(cfg.model):
        cfg.model.loss = cfg.loss
    return cfg


def move_batch_to_device(batch: CollateVariableShapeBatch, device: torch.device) -> CollateVariableShapeBatch:
    batch.file_batch = batch.file_batch.to(device, non_blocking=False)
    batch.lens_batch = batch.lens_batch.to(device, non_blocking=False)
    batch.ordered_seeds = batch.ordered_seeds.to(device, non_blocking=False)
    return batch


def make_ranked_batch(dtype: str, seeds: Sequence[int], *, augmentations, run_dict=None, device: torch.device) -> CollateVariableShapeBatch:
    rankings = torch.tensor([[seeds[0], seeds[1], -1, -1, -1]], dtype=torch.int64)
    dataset = dset_single_stream_ordered_minimal(
        all_combined_rankings=rankings,
        dtype=dtype,
        ddir_func=ddir_func,
        seed_func=seed_func_default,
        augmentations=augmentations,
        include_goodseed=False,
        dset_partition="verify",
        batch_augmentations=None,
        map_on="cpu",
        using_transforms=False,
        dset_version="three",
    )
    if run_dict is not None:
        dataset.update_attrs_from_run_dict(run_dict)
    batch = CollateVariableShapeBatch([dataset[0]])
    return move_batch_to_device(batch, device)


def summarise_losses(losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
    summary = {}
    for key, value in losses.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            summary[key] = float(value.detach().cpu())
    return summary


def run_data_generation_test(seeds: Sequence[int], truncation_psi: float) -> Dict[str, object]:
    from core_modules.tests import test_data_generation as tdg

    tdg.regenerate_triple_rgb(seeds, truncation_psi=truncation_psi)
    tdg.regenerate_triple_dmap(seeds, truncation_psi=truncation_psi)
    tdg.regenerate_sigma_fields(seeds, truncation_psi=truncation_psi)
    tdg.regenerate_landmarks(seeds)
    return {"generated_dir": os.environ["RWD_DATA_DIR"]}


def run_loader_test(seed: int, env: Dict[str, str]) -> Dict[str, object]:
    cmd = [sys.executable, "core_modules/tests/test_dset_loaders.py", "--seed", str(seed), "--map_on", "cpu"]
    run_cmd(cmd, cwd=FRAMEWORK_ROOT, env=env, label="dataset loader test")
    return {"seed": seed}


def run_experiment_forward_test(experiments: Iterable[str], seeds: Sequence[int], device: torch.device) -> Dict[str, object]:
    results = {}
    for experiment in experiments:
        print(f"\n[forward] experiment={experiment}")
        cfg = compose_experiment_cfg(experiment)
        dtype = str(cfg.data.dset_dict.selected_dtypes[0])
        augmentations = instantiate_transform(cfg.data.augmentations.tune)
        model = hydra.utils.instantiate(cfg.model, _recursive_=False).to(device).eval()
        model._trainer = SimpleNamespace(current_epoch=0)
        batch = make_ranked_batch(dtype, seeds, augmentations=augmentations, run_dict=cfg.data.dset_dict[dtype], device=device)
        with torch.no_grad():
            losses, preds = model.run_forward_pass(batch, return_global_vector=True, return_preds=True)
        loss_summary = summarise_losses(losses)
        total_loss = loss_summary.get("total_loss")
        if total_loss is None or not math.isfinite(total_loss):
            raise RuntimeError(f"{experiment}: non-finite total_loss {total_loss}")
        results[experiment] = {
            "dtype": dtype,
            "losses": loss_summary,
            "sum_in_batch": int(preds["sum_in_batch"]),
        }
        del model
        torch.cuda.empty_cache()
    return results


def run_released_checkpoint_test(run_id: str, seeds: Sequence[int], device: torch.device) -> Dict[str, object]:
    print(f"\n[checkpoint] reward_model_id={run_id}")
    dtype = str(get_datatype_from_model_id(run_id))
    tune_aug = instantiate_transform(load_tune_augmentation_from_cfg(run_id))
    model = load_rwd_model_from_cfg(run_id, strict=True).to(device).eval()
    model._trainer = SimpleNamespace(current_epoch=0)
    batch = make_ranked_batch(dtype, seeds, augmentations=tune_aug, run_dict=None, device=device)
    with torch.no_grad():
        losses, preds = model.run_forward_pass(batch, return_global_vector=True, return_preds=True)
    loss_summary = summarise_losses(losses)
    total_loss = loss_summary.get("total_loss")
    if total_loss is None or not math.isfinite(total_loss):
        raise RuntimeError(f"released checkpoint {run_id}: non-finite total_loss {total_loss}")
    del model
    torch.cuda.empty_cache()
    return {
        "reward_model_id": run_id,
        "dtype": dtype,
        "losses": loss_summary,
        "sum_in_batch": int(preds["sum_in_batch"]),
    }


def run_mesh_bank_test(args: argparse.Namespace, env: Dict[str, str]) -> Dict[str, object]:
    if args.baseline_pkl is None or args.tuned_pkl is None:
        raise SkipStep("mesh-bank test requires both --baseline-pkl and --tuned-pkl")
    outdir = args.workdir / "mesh_bank_test"
    if outdir.exists():
        shutil.rmtree(outdir)
    mesh_env = env.copy()
    existing_pythonpath = mesh_env.get("PYTHONPATH", "")
    mesh_env["PYTHONPATH"] = f"{FRAMEWORK_ROOT}:{existing_pythonpath}" if existing_pythonpath else str(FRAMEWORK_ROOT)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "eg3d/reward_tune_analysis/scripts/export_snapshot_mesh_bank.py"),
        "--baseline-pkl",
        str(args.baseline_pkl),
        "--tuned-pkl",
        str(args.tuned_pkl),
        "--outdir",
        str(outdir),
        "--start-seed",
        str(args.mesh_bank_start_seed),
        "--num-seeds",
        str(args.mesh_bank_num_seeds),
        "--methods",
        "legacy_sigma10",
    ]
    run_cmd(cmd, cwd=FRAMEWORK_ROOT, env=mesh_env, label="mesh-bank export test")
    return {"outdir": str(outdir)}


def execute_step(summary: SummaryWriter, step: str, fn, *args, **kwargs):
    started = time.time()
    try:
        details = fn(*args, **kwargs) or {}
        summary.add(step, "ok", seconds=round(time.time() - started, 2), **details)
        return details
    except SkipStep as exc:
        summary.add(step, "skipped", seconds=round(time.time() - started, 2), reason=str(exc))
        print(f"[skip] {step}: {exc}")
        return None
    except Exception as exc:
        summary.add(step, "failed", seconds=round(time.time() - started, 2), error=str(exc))
        summary.set_status("failed")
        raise


def main() -> None:
    args = parse_args()
    seeds = parse_seed_pair(args.seeds)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the maintained public-release verifier")

    args.workdir.mkdir(parents=True, exist_ok=True)
    data_dir = args.workdir / "generated_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    env = prepare_environment(args, data_dir)
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    set_process_env(env)

    summary = SummaryWriter(args.workdir / "verification_summary.json", seeds=seeds, workdir=args.workdir)

    if not args.skip_data_generation:
        execute_step(summary, "data_generation_test", run_data_generation_test, seeds, args.truncation_psi)
    else:
        summary.add("data_generation_test", "skipped", reason="--skip-data-generation")

    if not args.skip_loader_test:
        execute_step(summary, "dataset_loader_test", run_loader_test, seeds[0], env)
    else:
        summary.add("dataset_loader_test", "skipped", reason="--skip-loader-test")

    if not args.skip_forward_test:
        execute_step(summary, "maintained_reward_model_forward_test", run_experiment_forward_test, DEFAULT_EXPERIMENTS, seeds, device)
    else:
        summary.add("maintained_reward_model_forward_test", "skipped", reason="--skip-forward-test")

    if not args.skip_released_checkpoint_test:
        execute_step(summary, "released_reward_model_checkpoint_test", run_released_checkpoint_test, args.reward_model_id, seeds, device)
    else:
        summary.add("released_reward_model_checkpoint_test", "skipped", reason="--skip-released-checkpoint-test")

    if not args.skip_mesh_bank:
        execute_step(summary, "mesh_bank_export_test", run_mesh_bank_test, args, env)
    else:
        summary.add("mesh_bank_export_test", "skipped", reason="--skip-mesh-bank")

    summary.set_status("ok")
    print(f"\nVerification summary: {summary.path}")


if __name__ == "__main__":
    main()
