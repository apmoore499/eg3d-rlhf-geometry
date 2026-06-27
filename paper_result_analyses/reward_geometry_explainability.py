from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import trimesh
from omegaconf import OmegaConf
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_result_analyses.path_defaults import (  # noqa: E402
    EG3D_ROOT,
    RLHF_CORE_ROOT,
    reported_run_dir,
    reward_embedding_analysis_dir,
)

RLHF_SRC_ROOT = RLHF_CORE_ROOT

for _path in (REPO_ROOT, EG3D_ROOT, RLHF_SRC_ROOT.parent):
    _str = str(_path)
    if _str not in sys.path:
        sys.path.insert(0, _str)

try:
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
except Exception:
    pass

from shape_utils import convert_sdf_samples_to_ply as eg3d_convert_sdf_samples_to_ply
import core_modules
from core_modules.data.create_train_data import generation_utils as gen_utils
from core_modules.utils import finetuning_utils
from core_modules.utils import meshing_utils
from core_modules.utils import reward_loading

# Reward run_config.yaml references transforms under the training-time package
# name `src_rlhf`; on disk that package is `core_modules`. Alias so hydra can
# resolve `src_rlhf.data.custom_transforms.*`.
sys.modules.setdefault("src_rlhf", core_modules)


@dataclass
class ExplainabilityConfig:
    reward_model_id: str = "7wnzkgie"
    orig_pkl: str = str(REPO_ROOT / "pkl_pt" / "eg3d_1" / "ffhq512-128.pkl")
    tuned_pkl: str = str(reported_run_dir() / "network-snapshot-002068_LAST.pkl")
    exp3_samples_csv: str = str(
        reward_embedding_analysis_dir()
        / "exp3_orig_vs_tuned"
        / "exp3_orig_vs_tuned_samples.csv"
    )
    results_dir: str = ""  # filled in by main() based on partition_version
    partition_version: str = "aw98_template"  # "legacy" reproduces the original 9-region BB scheme
    shape_res: int = 256
    truncation_psi: float = 0.7
    truncation_cutoff: int = 14
    max_batch: int = 1_000_000
    noise_mode: str = "const"
    device: str = "cuda"
    shapley_permutations: int = 8
    ig_steps: int = 8
    export_top_k: int = 10
    mesh_level: float = 10.0
    mesh_bordermain: int = 30
    mesh_bordersides: int = 60
    mesh_borderback: int = 80
    random_seed: int = 123


RESULTS_DIR_BASE: str = str(reward_embedding_analysis_dir())


LEGACY_REGION_ORDER: Tuple[str, ...] = (
    "brow",
    "left_eye_orbit",
    "right_eye_orbit",
    "nose",
    "left_cheek",
    "right_cheek",
    "mouth",
    "chin_jaw",
    "other",
)


AW98_TEMPLATE_MASKS_PT: Path = Path(RESULTS_DIR_BASE) / "aw98_template_masks" / "region_masks.pt"


REGION_ORDER: Tuple[str, ...] = LEGACY_REGION_ORDER  # overwritten in main() based on partition_version


def _to_serializable(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_serializable(v) for k, v in x.items()}
    return x


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(_to_serializable(payload), f, indent=2)


def _seed_to_z(seed: int, device: torch.device) -> Tuple[torch.Tensor, np.ndarray]:
    z_np = np.random.RandomState(int(seed)).randn(512).astype(np.float32)
    z = torch.from_numpy(z_np).unsqueeze(0).to(device)
    return z, z_np


def _load_reward_assets(cfg: ExplainabilityConfig, device: torch.device):
    model = reward_loading.load_rwd_model_from_cfg(cfg.reward_model_id)
    run_config_path = RLHF_SRC_ROOT / "RWD_MODELS_FOR_TUNING" / cfg.reward_model_id / "run_config.yaml"
    run_config = OmegaConf.load(run_config_path)
    sigma_aug = hydra.utils.instantiate(run_config.data.augmentations.sigma_norm)
    sigma_aug.eval()
    if hasattr(sigma_aug, "to"):
        sigma_aug = sigma_aug.to(device)
    return model.to(device).eval(), sigma_aug


def _load_generator_assets(network_pkl: str, truncation_psi: float, shape_res: int):
    da = gen_utils.load_generator(
        model_path=Path(network_pkl),
        truncation_psi=truncation_psi,
        truncation_cutoff=14,
        shape_res=shape_res,
    )
    cond_path = gen_utils.STATIC_CONFIGS_DIR / "single_dmap_conditioning.pt"
    cond = torch.load(cond_path, map_location=da.device)
    pads_path = gen_utils.STATIC_CONFIGS_DIR / "pads_vals_entire.yaml"
    pads = OmegaConf.load(pads_path)
    return da, cond, pads


def _prepare_sigma_sampling(
    G,
    pads_vals,
    shape_res: int,
) -> Tuple[finetuning_utils.MeshUtilsDataClass, torch.Tensor, Tuple[int, ...], Any]:
    mudc = finetuning_utils.MeshUtilsDataClass()
    samples, shape, tripleaxis_index = mudc.get_samples_coordinates_from_pads_vals_dict(
        pads_vals=pads_vals,
        G=G,
        shape_res=shape_res,
    )
    return mudc, samples, shape, tripleaxis_index


def _sample_sigma_volume(
    mudc: finetuning_utils.MeshUtilsDataClass,
    G,
    conditioning_params: torch.Tensor,
    samples: torch.Tensor,
    shape: Tuple[int, ...],
    seed: int,
    truncation_psi: float,
    truncation_cutoff: int,
    noise_mode: str,
    max_batch: int,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray]:
    z, z_np = _seed_to_z(seed, device=device)
    with torch.no_grad():
        sigmas = mudc.mesh_subset_of_points_from_samples_from_z_with_grad(
            G=G,
            z=z,
            conditioning_params=conditioning_params,
            samples=samples,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            noise_mode=noise_mode,
            update_emas=False,
            max_batch=max_batch,
        )
    volume = sigmas.squeeze(0).squeeze(-1).reshape(shape[1:4]).detach().cpu().float()
    return volume, z_np


def _reward_forward_device(
    reward_model,
    sigma_aug,
    volume_xyz: torch.Tensor,
) -> torch.Tensor:
    aug_volume = sigma_aug(volume_xyz)
    model_input = aug_volume.permute(2, 1, 0).contiguous().unsqueeze(0)
    emb8192 = reward_model.Conv3DModule.forward_to_global_vec(
        model_input, return_global_only=True
    )
    emb512 = reward_model.MLP(emb8192)
    reward = reward_model.forward_to_scalar_reward_from_single_global(emb512)
    return reward.reshape(())


def _coords_grid(shape_xyz: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = torch.linspace(0.0, 1.0, shape_xyz[0])
    ys = torch.linspace(0.0, 1.0, shape_xyz[1])
    zs = torch.linspace(0.0, 1.0, shape_xyz[2])
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    return gx, gy, gz


def _build_region_partition_legacy(shape_xyz: Sequence[int]) -> Dict[str, torch.Tensor]:
    x, y, z = _coords_grid(shape_xyz)
    assigned = torch.zeros(shape_xyz, dtype=torch.bool)
    masks: Dict[str, torch.Tensor] = {}

    defs = {
        "brow": (x >= 0.25) & (x <= 0.75) & (y >= 0.70) & (y <= 0.88) & (z >= 0.45) & (z <= 0.82),
        "left_eye_orbit": (x >= 0.14) & (x < 0.38) & (y >= 0.50) & (y < 0.72) & (z >= 0.52) & (z <= 0.86),
        "right_eye_orbit": (x > 0.62) & (x <= 0.86) & (y >= 0.50) & (y < 0.72) & (z >= 0.52) & (z <= 0.86),
        "nose": (x >= 0.38) & (x <= 0.62) & (y >= 0.38) & (y <= 0.68) & (z >= 0.60) & (z <= 0.94),
        "left_cheek": (x >= 0.08) & (x < 0.34) & (y >= 0.28) & (y < 0.58) & (z >= 0.46) & (z <= 0.82),
        "right_cheek": (x > 0.66) & (x <= 0.92) & (y >= 0.28) & (y < 0.58) & (z >= 0.46) & (z <= 0.82),
        "mouth": (x >= 0.32) & (x <= 0.68) & (y >= 0.20) & (y < 0.40) & (z >= 0.54) & (z <= 0.88),
        "chin_jaw": (x >= 0.18) & (x <= 0.82) & (y >= 0.02) & (y < 0.20) & (z >= 0.40) & (z <= 0.82),
    }
    for region in LEGACY_REGION_ORDER[:-1]:
        mask = defs[region] & ~assigned
        masks[region] = mask
        assigned |= mask
    masks["other"] = ~assigned
    return masks


def _build_region_partition_aw98_template(shape_xyz: Sequence[int]) -> Dict[str, torch.Tensor]:
    """Load AW98-landmark-anchored region masks from disk.

    The masks are built once by `aw98_template_partition.py` from the 100-seed
    averaged WFLW 98-pt landmarks (detected on each seed's canonical-view
    render of the untuned generator), back-projected to world coords via
    cam2world + intrinsics, and grouped by WFLW semantic IDs into anatomical
    AABBs in the σ-cube.
    """
    if not AW98_TEMPLATE_MASKS_PT.exists():
        raise FileNotFoundError(
            f"AW98 template masks not found at {AW98_TEMPLATE_MASKS_PT}. "
            f"Run `python paper_result_analyses/aw98_template_partition.py --stage A` "
            f"then --stage B then --stage C first."
        )
    payload = torch.load(AW98_TEMPLATE_MASKS_PT, map_location="cpu")
    masks_disk: Dict[str, torch.Tensor] = payload["masks"]
    saved_shape = tuple(int(x) for x in payload["cube_shape"])
    request_shape = tuple(int(x) for x in shape_xyz)
    if saved_shape != request_shape:
        raise RuntimeError(
            f"AW98 template mask cube_shape={saved_shape} but explainability "
            f"requested {request_shape}. Re-run aw98_template_partition.py "
            f"with matching SHAPE_RES / pads_vals."
        )
    return {k: v.bool().contiguous() for k, v in masks_disk.items()}


def _aw98_template_region_order() -> Tuple[str, ...]:
    if not AW98_TEMPLATE_MASKS_PT.exists():
        raise FileNotFoundError(
            f"AW98 template masks not found at {AW98_TEMPLATE_MASKS_PT}. "
            f"Run aw98_template_partition.py first."
        )
    payload = torch.load(AW98_TEMPLATE_MASKS_PT, map_location="cpu")
    return tuple(payload["region_priority"])


def _build_region_partition(
    shape_xyz: Sequence[int],
    partition_version: str = "aw98_template",
) -> Dict[str, torch.Tensor]:
    if partition_version == "legacy":
        return _build_region_partition_legacy(shape_xyz)
    if partition_version == "aw98_template":
        return _build_region_partition_aw98_template(shape_xyz)
    raise ValueError(f"Unknown partition_version={partition_version!r}")


def _reward_to_float(
    reward_model,
    sigma_aug,
    volume_xyz: torch.Tensor,
    device: torch.device,
) -> float:
    with torch.no_grad():
        reward = _reward_forward_device(
            reward_model=reward_model,
            sigma_aug=sigma_aug,
            volume_xyz=volume_xyz.to(device),
        )
    return float(reward.detach().cpu().item())


def _estimate_shapley(
    reward_model,
    sigma_aug,
    volume_orig: torch.Tensor,
    volume_tuned: torch.Tensor,
    region_masks: Dict[str, torch.Tensor],
    device: torch.device,
    n_permutations: int,
    rng: np.random.RandomState,
) -> Tuple[Dict[str, float], float, float, float]:
    base = volume_orig.to(device)
    target = volume_tuned.to(device)
    base_reward = _reward_to_float(reward_model, sigma_aug, base, device)
    target_reward = _reward_to_float(reward_model, sigma_aug, target, device)
    contrib = {region: 0.0 for region in REGION_ORDER}
    masks = {region: mask.to(device) for region, mask in region_masks.items()}

    for _ in range(n_permutations):
        coalition = base.clone()
        prev_reward = base_reward
        perm = list(REGION_ORDER)
        rng.shuffle(perm)
        for region in perm:
            coalition = torch.where(masks[region], target, coalition)
            new_reward = _reward_to_float(reward_model, sigma_aug, coalition, device)
            contrib[region] += new_reward - prev_reward
            prev_reward = new_reward

    for region in contrib:
        contrib[region] /= float(n_permutations)
    shapley_sum = float(sum(contrib.values()))
    delta_reward = target_reward - base_reward
    completeness_error = shapley_sum - delta_reward
    return contrib, delta_reward, shapley_sum, completeness_error


def _integrated_gradients(
    reward_model,
    sigma_aug,
    volume_orig: torch.Tensor,
    volume_tuned: torch.Tensor,
    device: torch.device,
    steps: int,
) -> Tuple[torch.Tensor, float]:
    x0 = volume_orig.to(device)
    x1 = volume_tuned.to(device)
    delta = x1 - x0
    total_grad = torch.zeros_like(x0)
    for alpha in torch.linspace(1.0 / steps, 1.0, steps, device=device):
        x = (x0 + alpha * delta).clone().detach().requires_grad_(True)
        reward = _reward_forward_device(
            reward_model=reward_model,
            sigma_aug=sigma_aug,
            volume_xyz=x,
        )
        grad = torch.autograd.grad(reward, x, retain_graph=False, create_graph=False)[0]
        total_grad += grad.detach()
    ig = (delta * total_grad / float(steps)).detach().cpu()
    with torch.no_grad():
        reward_delta = float(
            (
                _reward_forward_device(reward_model, sigma_aug, x1)
                - _reward_forward_device(reward_model, sigma_aug, x0)
            )
            .detach()
            .cpu()
            .item()
        )
    completeness_error = float(ig.sum().item() - reward_delta)
    return ig, completeness_error


def _aggregate_region_scores(
    ig: torch.Tensor,
    region_masks: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for region, mask in region_masks.items():
        vals = ig[mask]
        out[region] = {
            "signed_sum": float(vals.sum().item()),
            "abs_sum": float(vals.abs().sum().item()),
            "positive_sum": float(vals.clamp_min(0).sum().item()),
            "negative_sum": float((-vals.clamp_max(0)).sum().item()),
        }
    return out


def _volume_to_full_cube(
    cropped: torch.Tensor,
    full_res: int,
    tripleaxis_index: Any,
    fill_value: float = 0.0,
) -> torch.Tensor:
    full = torch.full((full_res, full_res, full_res), fill_value, dtype=cropped.dtype)
    rhs = int(tripleaxis_index.ax1horiz.right)
    lhs = int(full_res - tripleaxis_index.ax1horiz.left)
    bot = int(tripleaxis_index.ax2vert.bottom)
    top = int(full_res - tripleaxis_index.ax2vert.top)
    rear = int(tripleaxis_index.ax3depth.rear)
    front = int(full_res - tripleaxis_index.ax3depth.front)
    full[rhs : full_res - lhs, bot : full_res - top, rear : full_res - front] = cropped
    return full


def _sample_mesh_from_seed(
    G,
    conditioning_params: torch.Tensor,
    seed: int,
    cfg: ExplainabilityConfig,
    device: torch.device,
) -> trimesh.Trimesh:
    z, _ = _seed_to_z(seed, device=device)
    with torch.no_grad():
        ws = G.mapping(
            z,
            conditioning_params.to(device),
            truncation_psi=cfg.truncation_psi,
            truncation_cutoff=cfg.truncation_cutoff,
        )
    mesh = meshing_utils.sample_sigmas_to_trimesh_from_ws(
        G=G,
        ws=ws,
        conditioning_params=conditioning_params.to(device),
        shape_res=cfg.shape_res,
        device=device,
        truncation_psi=cfg.truncation_psi,
        truncation_cutoff=cfg.truncation_cutoff,
        noise_mode=cfg.noise_mode,
        bordermain=cfg.mesh_bordermain,
        bordersides=cfg.mesh_bordersides,
        borderback=cfg.mesh_borderback,
        level=cfg.mesh_level,
    )
    return mesh


def _volume_to_visualise_mesh(
    volume: torch.Tensor,
    voxel_size: float = 1.0 / 256.0,
    level: float = 10.0,
) -> trimesh.Trimesh:
    vol_np = volume.detach().cpu().numpy()
    return eg3d_convert_sdf_samples_to_ply(
        vol_np,
        voxel_grid_origin=[-0.5, -0.5, -0.5],
        voxel_size=voxel_size,
        ply_filename_out="unused.ply",
        level=level,
        process=False,
        return_mesh_only=True,
    )


def _sample_volume_nearest_visualise_coords(
    volume: np.ndarray,
    verts: np.ndarray,
    voxel_origin: Tuple[float, float, float] = (-0.5, -0.5, -0.5),
    voxel_size: float = 1.0 / 256.0,
) -> np.ndarray:
    origin = np.asarray(voxel_origin, dtype=np.float32)
    coords = np.rint((verts - origin[None, :]) / float(voxel_size)).astype(np.int64)
    max_idx = np.asarray(volume.shape, dtype=np.int64) - 1
    coords = np.clip(coords, 0, max_idx)
    return volume[coords[:, 0], coords[:, 1], coords[:, 2]]


def _sample_volume_nearest(volume: np.ndarray, verts: np.ndarray) -> np.ndarray:
    res = volume.shape[0]
    coords = ((verts + 0.5) * (res - 1)).round().astype(np.int64)
    coords = np.clip(coords, 0, res - 1)
    return volume[coords[:, 0], coords[:, 1], coords[:, 2]]


def _to_rgba(values: np.ndarray, vmax: Optional[float] = None) -> np.ndarray:
    vals = values.astype(np.float32)
    vmax = float(vmax if vmax is not None else np.max(np.abs(vals)) + 1e-8)
    norm = np.clip((vals / (2 * vmax)) + 0.5, 0.0, 1.0)
    colors = plt.get_cmap("coolwarm")(norm)
    return (colors * 255).astype(np.uint8)


def _export_plotly_mesh(mesh: trimesh.Trimesh, out_path: Path, title: str) -> bool:
    try:
        import plotly.graph_objects as go
    except Exception:
        return False
    verts = mesh.vertices
    faces = mesh.faces
    vc = mesh.visual.vertex_colors
    intensity = vc[:, 0].astype(np.float32)
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                vertexcolor=[f"rgba({r},{g},{b},{a/255.0})" for r, g, b, a in vc],
                intensity=intensity,
                showscale=False,
                flatshading=False,
            )
        ]
    )
    fig.update_layout(title=title, scene_aspectmode="data")
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return True


def _plot_top_seed_panel(
    seed: int,
    reward_orig: float,
    reward_tuned: float,
    shapley: Dict[str, float],
    ig_region: Dict[str, Dict[str, float]],
    sigma_delta: torch.Tensor,
    ig: torch.Tensor,
    out_path: Path,
) -> None:
    midx = sigma_delta.shape[0] // 2
    midy = sigma_delta.shape[1] // 2
    midz = sigma_delta.shape[2] // 2
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    slices = [
        (sigma_delta[midx].numpy(), "sigma delta yz"),
        (sigma_delta[:, midy, :].numpy(), "sigma delta xz"),
        (sigma_delta[:, :, midz].numpy(), "sigma delta xy"),
        (ig[midx].numpy(), "IG yz"),
        (ig[:, midy, :].numpy(), "IG xz"),
        (ig[:, :, midz].numpy(), "IG xy"),
    ]
    for ax, (arr, title) in zip(axes.reshape(-1), slices):
        vmax = np.max(np.abs(arr)) + 1e-8
        im = ax.imshow(arr.T, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"seed={seed} reward_orig={reward_orig:.3f} reward_tuned={reward_tuned:.3f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    shapley_vals = [shapley[r] for r in REGION_ORDER]
    ig_vals = [ig_region[r]["signed_sum"] for r in REGION_ORDER]
    axes[0].bar(REGION_ORDER, shapley_vals)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_title("Region Shapley")
    axes[1].bar(REGION_ORDER, ig_vals)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("Region IG signed sum")
    plt.tight_layout()
    plt.savefig(out_path.with_name(out_path.stem + "_region_bars.png"), dpi=180)
    plt.close(fig)


def _signed_max_abs_projection(volume: np.ndarray, axis: int) -> np.ndarray:
    idx = np.abs(volume).argmax(axis=axis, keepdims=True)
    return np.take_along_axis(volume, idx, axis=axis).squeeze(axis)


def _argmax_abs_index_map(volume: np.ndarray, axis: int) -> np.ndarray:
    return np.abs(volume).argmax(axis=axis).astype(np.float32)


def _plot_fullcube_sigma_delta_panel(
    seed: int,
    reward_orig: float,
    reward_tuned: float,
    sigma_delta_full: np.ndarray,
    out_path: Path,
) -> Dict[str, Any]:
    proj_yz = _signed_max_abs_projection(sigma_delta_full, axis=0)
    proj_xz = _signed_max_abs_projection(sigma_delta_full, axis=1)
    proj_xy = _signed_max_abs_projection(sigma_delta_full, axis=2)
    arg_yz = _argmax_abs_index_map(sigma_delta_full, axis=0)
    arg_xz = _argmax_abs_index_map(sigma_delta_full, axis=1)
    arg_xy = _argmax_abs_index_map(sigma_delta_full, axis=2)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    signed_panels = [
        (proj_yz, "full sigma maxabs yz"),
        (proj_xz, "full sigma maxabs xz"),
        (proj_xy, "full sigma maxabs xy"),
    ]
    arg_panels = [
        (arg_yz, "argmax depth yz"),
        (arg_xz, "argmax depth xz"),
        (arg_xy, "argmax depth xy"),
    ]
    for ax, (arr, title) in zip(axes[0], signed_panels):
        vmax = np.max(np.abs(arr)) + 1e-8
        im = ax.imshow(arr.T, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax, (arr, title) in zip(axes[1], arg_panels):
        im = ax.imshow(arr.T, origin="lower", cmap="viridis")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    max_flat_idx = int(np.abs(sigma_delta_full).argmax())
    max_xyz = np.unravel_index(max_flat_idx, sigma_delta_full.shape)
    max_val = float(sigma_delta_full[max_xyz])
    fig.suptitle(
        f"seed={seed} reward_orig={reward_orig:.3f} reward_tuned={reward_tuned:.3f} "
        f"max|delta|={abs(max_val):.3f} @ xyz={tuple(int(v) for v in max_xyz)}"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return {
        "max_abs_value": float(abs(max_val)),
        "signed_value_at_max_abs": max_val,
        "max_abs_xyz": [int(v) for v in max_xyz],
    }


def _summarise_regions(
    seed_df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    rows = []
    for region in REGION_ORDER:
        vals = seed_df[f"{prefix}_{region}"].to_numpy(dtype=np.float32)
        rows.append(
            {
                "region": region,
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "std": float(vals.std()),
                "frac_positive": float((vals > 0).mean()),
                "mean_abs": float(np.abs(vals).mean()),
            }
        )
    return pd.DataFrame(rows)


def _plot_region_summary(
    shapley_df: pd.DataFrame,
    ig_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(shapley_df["region"], shapley_df["mean"])
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_title("Mean region Shapley")
    axes[1].bar(ig_df["region"], ig_df["mean"])
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("Mean region IG signed sum")
    plt.tight_layout()
    plt.savefig(out_dir / "region_summary_means.png", dpi=180)
    plt.close(fig)


def _run_top_seed_exports(
    top_seeds: Sequence[int],
    seed_metrics: pd.DataFrame,
    reward_model,
    sigma_aug,
    cfg: ExplainabilityConfig,
    device: torch.device,
    out_dir: Path,
) -> None:
    da_orig, cond_orig, pads_orig = _load_generator_assets(cfg.orig_pkl, truncation_psi=1.0, shape_res=cfg.shape_res)
    da_tuned, cond_tuned, pads_tuned = _load_generator_assets(cfg.tuned_pkl, truncation_psi=1.0, shape_res=cfg.shape_res)
    mudc_orig, samples_orig, shape_orig, tri_orig = _prepare_sigma_sampling(da_orig.G, pads_vals=pads_orig, shape_res=cfg.shape_res)
    mudc_tuned, samples_tuned, shape_tuned, tri_tuned = _prepare_sigma_sampling(da_tuned.G, pads_vals=pads_tuned, shape_res=cfg.shape_res)
    region_masks = _build_region_partition(shape_orig[1:4], partition_version=cfg.partition_version)

    export_rows = []
    for rank, seed in enumerate(top_seeds, start=1):
        sample_dir = _ensure_dir(out_dir / f"rank_{rank:02d}_seed_{seed}")
        volume_orig, _ = _sample_sigma_volume(
            mudc=mudc_orig,
            G=da_orig.G,
            conditioning_params=cond_orig,
            samples=samples_orig,
            shape=shape_orig,
            seed=int(seed),
            truncation_psi=cfg.truncation_psi,
            truncation_cutoff=cfg.truncation_cutoff,
            noise_mode=cfg.noise_mode,
            max_batch=cfg.max_batch,
            device=device,
        )
        volume_tuned, _ = _sample_sigma_volume(
            mudc=mudc_tuned,
            G=da_tuned.G,
            conditioning_params=cond_tuned,
            samples=samples_tuned,
            shape=shape_tuned,
            seed=int(seed),
            truncation_psi=cfg.truncation_psi,
            truncation_cutoff=cfg.truncation_cutoff,
            noise_mode=cfg.noise_mode,
            max_batch=cfg.max_batch,
            device=device,
        )
        ig, _ = _integrated_gradients(
            reward_model=reward_model,
            sigma_aug=sigma_aug,
            volume_orig=volume_orig,
            volume_tuned=volume_tuned,
            device=device,
            steps=cfg.ig_steps,
        )
        ig_region = _aggregate_region_scores(ig=ig, region_masks=region_masks)
        row = seed_metrics.loc[seed_metrics["seed"] == seed].iloc[0]
        shapley = {region: float(row[f"shapley_{region}"]) for region in REGION_ORDER}
        reward_orig = float(row["reward_orig"])
        reward_tuned = float(row["reward_tuned"])
        sigma_delta = volume_tuned - volume_orig
        _plot_top_seed_panel(
            seed=int(seed),
            reward_orig=reward_orig,
            reward_tuned=reward_tuned,
            shapley=shapley,
            ig_region=ig_region,
            sigma_delta=sigma_delta,
            ig=ig,
            out_path=sample_dir / "explainability_panel.png",
        )
        sigma_delta_full = _volume_to_full_cube(sigma_delta, cfg.shape_res, tri_orig, fill_value=0.0).numpy()
        fullcube_meta = _plot_fullcube_sigma_delta_panel(
            seed=int(seed),
            reward_orig=reward_orig,
            reward_tuned=reward_tuned,
            sigma_delta_full=sigma_delta_full,
            out_path=sample_dir / "explainability_panel_fullcube_sigma_delta.png",
        )

        mesh_tuned = _volume_to_visualise_mesh(volume_tuned, level=cfg.mesh_level)
        vals = _sample_volume_nearest_visualise_coords(ig.detach().cpu().numpy(), mesh_tuned.vertices)
        mesh_tuned.visual.vertex_colors = _to_rgba(vals)
        mesh_tuned.export(sample_dir / "tuned_mesh_ig_colored.ply")
        html_written = _export_plotly_mesh(
            mesh=mesh_tuned,
            out_path=sample_dir / "tuned_mesh_ig_colored.html",
            title=f"seed {seed} tuned mesh colored by IG",
        )
        export_rows.append(
            {
                "rank": rank,
                "seed": int(seed),
                "reward_orig": reward_orig,
                "reward_tuned": reward_tuned,
                "reward_delta": reward_tuned - reward_orig,
                "top_shapley_region": max(shapley, key=shapley.get),
                "top_ig_region": max(REGION_ORDER, key=lambda r: ig_region[r]["signed_sum"]),
                "fullcube_max_abs_sigma_delta": fullcube_meta["max_abs_value"],
                "fullcube_max_abs_sigma_delta_xyz": str(tuple(fullcube_meta["max_abs_xyz"])),
                "mesh_html_written": bool(html_written),
            }
        )
        _write_json(
            sample_dir / "metadata.json",
            {
                "seed": int(seed),
                "reward_orig": reward_orig,
                "reward_tuned": reward_tuned,
                "reward_delta": reward_tuned - reward_orig,
                "shapley": shapley,
                "ig_region_signed": {r: ig_region[r]["signed_sum"] for r in REGION_ORDER},
                "fullcube_sigma_delta": fullcube_meta,
            },
        )
    pd.DataFrame(export_rows).to_csv(out_dir / "top_seed_exports.csv", index=False)


def main() -> None:
    cfg = ExplainabilityConfig()
    global REGION_ORDER
    if cfg.partition_version == "legacy":
        REGION_ORDER = LEGACY_REGION_ORDER
    elif cfg.partition_version == "aw98_template":
        REGION_ORDER = _aw98_template_region_order()
    else:
        raise ValueError(f"Unknown partition_version={cfg.partition_version!r}")
    if not cfg.results_dir:
        suffix_map = {
            "legacy": "reward_geometry_explainability",
            "aw98_template": "reward_geometry_explainability_aw98",
        }
        cfg.results_dir = f"{RESULTS_DIR_BASE}/{suffix_map[cfg.partition_version]}"
    device = torch.device(cfg.device)
    out_dir = _ensure_dir(Path(cfg.results_dir))
    rng = np.random.RandomState(cfg.random_seed)

    exp3_df = pd.read_csv(cfg.exp3_samples_csv)
    seeds = (
        exp3_df.loc[exp3_df["model_label"] == "orig", "seed"]
        .drop_duplicates()
        .astype(int)
        .sort_values()
        .tolist()
    )

    reward_model, sigma_aug = _load_reward_assets(cfg, device)
    da_orig, cond_orig, pads_orig = _load_generator_assets(cfg.orig_pkl, truncation_psi=1.0, shape_res=cfg.shape_res)
    da_tuned, cond_tuned, pads_tuned = _load_generator_assets(cfg.tuned_pkl, truncation_psi=1.0, shape_res=cfg.shape_res)
    mudc_orig, samples_orig, shape_orig, tri_orig = _prepare_sigma_sampling(da_orig.G, pads_vals=pads_orig, shape_res=cfg.shape_res)
    mudc_tuned, samples_tuned, shape_tuned, _tri_tuned = _prepare_sigma_sampling(da_tuned.G, pads_vals=pads_tuned, shape_res=cfg.shape_res)
    if tuple(shape_orig[1:4]) != tuple(shape_tuned[1:4]):
        raise ValueError(f"orig/tuned shape mismatch: {shape_orig[1:4]} vs {shape_tuned[1:4]}")
    region_masks = _build_region_partition(shape_orig[1:4])

    region_counts = {r: int(region_masks[r].sum().item()) for r in REGION_ORDER}
    _write_json(out_dir / "region_mask_counts.json", region_counts)

    rows: List[Dict[str, Any]] = []
    iterator = tqdm(seeds, desc="reward geometry explainability")
    for seed in iterator:
        volume_orig, _ = _sample_sigma_volume(
            mudc=mudc_orig,
            G=da_orig.G,
            conditioning_params=cond_orig,
            samples=samples_orig,
            shape=shape_orig,
            seed=int(seed),
            truncation_psi=cfg.truncation_psi,
            truncation_cutoff=cfg.truncation_cutoff,
            noise_mode=cfg.noise_mode,
            max_batch=cfg.max_batch,
            device=device,
        )
        volume_tuned, _ = _sample_sigma_volume(
            mudc=mudc_tuned,
            G=da_tuned.G,
            conditioning_params=cond_tuned,
            samples=samples_tuned,
            shape=shape_tuned,
            seed=int(seed),
            truncation_psi=cfg.truncation_psi,
            truncation_cutoff=cfg.truncation_cutoff,
            noise_mode=cfg.noise_mode,
            max_batch=cfg.max_batch,
            device=device,
        )
        shapley, reward_delta, shapley_sum, shapley_err = _estimate_shapley(
            reward_model=reward_model,
            sigma_aug=sigma_aug,
            volume_orig=volume_orig,
            volume_tuned=volume_tuned,
            region_masks=region_masks,
            device=device,
            n_permutations=cfg.shapley_permutations,
            rng=rng,
        )
        ig, ig_err = _integrated_gradients(
            reward_model=reward_model,
            sigma_aug=sigma_aug,
            volume_orig=volume_orig,
            volume_tuned=volume_tuned,
            device=device,
            steps=cfg.ig_steps,
        )
        reward_orig = _reward_to_float(reward_model, sigma_aug, volume_orig, device)
        reward_tuned = _reward_to_float(reward_model, sigma_aug, volume_tuned, device)
        ig_region = _aggregate_region_scores(ig=ig, region_masks=region_masks)
        row: Dict[str, Any] = {
            "seed": int(seed),
            "reward_orig": float(reward_orig),
            "reward_tuned": float(reward_tuned),
            "reward_delta": float(reward_tuned - reward_orig),
            "shapley_sum": float(shapley_sum),
            "shapley_completeness_error": float(shapley_err),
            "ig_sum": float(ig.sum().item()),
            "ig_completeness_error": float(ig_err),
            "sigma_delta_l1": float((volume_tuned - volume_orig).abs().mean().item()),
        }
        for region in REGION_ORDER:
            row[f"shapley_{region}"] = float(shapley[region])
            row[f"ig_signed_{region}"] = float(ig_region[region]["signed_sum"])
            row[f"ig_abs_{region}"] = float(ig_region[region]["abs_sum"])
        row["top_shapley_region"] = max(REGION_ORDER, key=lambda r: row[f"shapley_{r}"])
        row["top_ig_region"] = max(REGION_ORDER, key=lambda r: row[f"ig_signed_{r}"])
        rows.append(row)

    seed_df = pd.DataFrame(rows).sort_values("reward_delta", ascending=False).reset_index(drop=True)
    seed_df.to_csv(out_dir / "seed_level_explainability.csv", index=False)

    shapley_summary = _summarise_regions(seed_df, prefix="shapley")
    ig_summary = _summarise_regions(seed_df, prefix="ig_signed")
    shapley_summary.to_csv(out_dir / "region_shapley_summary.csv", index=False)
    ig_summary.to_csv(out_dir / "region_ig_summary.csv", index=False)
    _plot_region_summary(shapley_summary, ig_summary, out_dir)

    top_seed_counts = (
        seed_df["top_shapley_region"].value_counts().rename_axis("region").reset_index(name="count")
    )
    top_seed_counts.to_csv(out_dir / "top_shapley_region_counts.csv", index=False)
    top_ig_counts = (
        seed_df["top_ig_region"].value_counts().rename_axis("region").reset_index(name="count")
    )
    top_ig_counts.to_csv(out_dir / "top_ig_region_counts.csv", index=False)

    region_alignment_rows = []
    for region in REGION_ORDER:
        a = seed_df[f"shapley_{region}"]
        b = seed_df[f"ig_signed_{region}"]
        region_alignment_rows.append(
            {
                "region": region,
                "pearson": float(a.corr(b, method="pearson")),
                "spearman": float(a.corr(b, method="spearman")),
            }
        )
    region_alignment_df = pd.DataFrame(region_alignment_rows)
    region_alignment_df.to_csv(out_dir / "region_shapley_ig_alignment.csv", index=False)

    summary = {
        "config": asdict(cfg),
        "n_seeds": int(len(seed_df)),
        "reward_delta": {
            "mean": float(seed_df["reward_delta"].mean()),
            "median": float(seed_df["reward_delta"].median()),
            "min": float(seed_df["reward_delta"].min()),
            "max": float(seed_df["reward_delta"].max()),
            "frac_positive": float((seed_df["reward_delta"] > 0).mean()),
        },
        "shapley_completeness_abs_mean": float(seed_df["shapley_completeness_error"].abs().mean()),
        "ig_completeness_abs_mean": float(seed_df["ig_completeness_error"].abs().mean()),
        "top_shapley_region_counts": top_seed_counts.to_dict(orient="records"),
        "top_ig_region_counts": top_ig_counts.to_dict(orient="records"),
        "region_shapley_mean": {
            row["region"]: row["mean"] for row in shapley_summary.to_dict(orient="records")
        },
        "region_ig_mean": {
            row["region"]: row["mean"] for row in ig_summary.to_dict(orient="records")
        },
    }
    _write_json(out_dir / "analysis_summary.json", summary)

    top_seeds = seed_df.head(cfg.export_top_k)["seed"].astype(int).tolist()
    top_dir = _ensure_dir(out_dir / "top_reward_delta_mesh_exports")
    _run_top_seed_exports(
        top_seeds=top_seeds,
        seed_metrics=seed_df,
        reward_model=reward_model,
        sigma_aug=sigma_aug,
        cfg=cfg,
        device=device,
        out_dir=top_dir,
    )

    _write_json(
        out_dir / "manifest.json",
        {
            "analysis_summary": str(out_dir / "analysis_summary.json"),
            "seed_level_explainability": str(out_dir / "seed_level_explainability.csv"),
            "top_seed_exports": str(top_dir / "top_seed_exports.csv"),
        },
    )
    print(json.dumps({"results_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
