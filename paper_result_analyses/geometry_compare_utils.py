from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import itertools
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
EG3D_ROOT = REPO_ROOT / "eg3d"
RLHF_ROOT = REPO_ROOT / "reward_model_training" / "reward_model_framework" / "core_modules"

for _path in (REPO_ROOT, EG3D_ROOT):
    _str = str(_path)
    if _str not in sys.path:
        sys.path.insert(0, _str)

from training.volumetric_rendering.ray_sampler import RaySampler


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GEN_UTILS = _load_module_from_path(
    "nb_generation_utils",
    RLHF_ROOT / "data" / "create_train_data" / "generation_utils.py",
)

_RAY_SAMPLER = RaySampler()


@dataclass
class GeometryCompareConfig:
    orig_pkl: str
    tuned_pkl: str
    dataset_path: Optional[str] = None
    n_samples: int = 100
    latent_seed: int = 0
    depth_resolution: int = 128
    sigma_resolution: int = 64
    rgb_metric_resolution: int = 128
    rgb_model_key: str = "image"
    truncation_psi: float = 1.0
    truncation_cutoff: Optional[int] = 14
    noise_mode: str = "const"
    pointcloud_radius_cutoff: Optional[float] = 4.0
    pointcloud_plot_points: int = 4000
    n_viewpoints: int = 8
    viewpoint_seed: int = 123
    sigma_point_batch: int = 250000
    model_batch_size: int = 4
    sigma_batch_size: int = 2
    identity_batch_size: int = 8
    device: str = "cuda"
    results_dir: Optional[str] = None


def _to_numpy_image(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().float().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _shared_latents(n_samples: int, latent_seed: int, z_dim: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(latent_seed)
    return torch.randn((n_samples, z_dim), generator=gen, device=device)


def _expand_camera(camera_params: torch.Tensor, batch_size: int) -> torch.Tensor:
    if camera_params.shape[0] == batch_size:
        return camera_params
    if camera_params.shape[0] != 1:
        raise ValueError(f"camera batch mismatch: got {camera_params.shape[0]}, wanted 1 or {batch_size}")
    return camera_params.expand(batch_size, -1)


def _render_depth_batch(
    da,
    z_batch: torch.Tensor,
    camera_params: torch.Tensor,
    depth_resolution: int,
    truncation_psi: float,
    truncation_cutoff: int,
    noise_mode: str,
    batch_size: int,
    desc: str,
) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(z_batch), batch_size), desc=desc):
            z_mb = z_batch[start : start + batch_size]
            c_mb = _expand_camera(camera_params, len(z_mb))
            depth = da.G.forward(
                z_mb,
                c=c_mb,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
                neural_rendering_resolution=depth_resolution,
                noise_mode=noise_mode,
            )["image_depth"]
            outs.append(depth[:, 0].detach().cpu())
    return torch.cat(outs, dim=0)


def _render_rgb_batch(
    da,
    z_batch: torch.Tensor,
    camera_params: torch.Tensor,
    conditioning_params: Optional[torch.Tensor],
    neural_rendering_resolution: int,
    truncation_psi: float,
    truncation_cutoff: int,
    noise_mode: str,
    batch_size: int,
    desc: str,
    image_key: str = "image",
    resize_to: Optional[int] = None,
) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(z_batch), batch_size), desc=desc):
            z_mb = z_batch[start : start + batch_size]
            render_c_mb = _expand_camera(camera_params, len(z_mb))
            cond_c_src = conditioning_params if conditioning_params is not None else camera_params
            cond_c_mb = _expand_camera(cond_c_src, len(z_mb))
            ws = da.G.mapping(
                z_mb,
                cond_c_mb,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
            )
            rgb = da.G.synthesis(
                ws,
                render_c_mb,
                neural_rendering_resolution=neural_rendering_resolution,
                noise_mode=noise_mode,
            )[image_key]
            if resize_to is not None and rgb.shape[-1] != resize_to:
                rgb = F.interpolate(rgb, size=(resize_to, resize_to), mode="bilinear", align_corners=False)
            outs.append(rgb.detach().cpu())
    return torch.cat(outs, dim=0)


def _render_rgb_multiview(
    da,
    z_batch: torch.Tensor,
    camera_params_views: torch.Tensor,
    conditioning_params_views: Optional[torch.Tensor],
    neural_rendering_resolution: int,
    truncation_psi: float,
    truncation_cutoff: int,
    noise_mode: str,
    batch_size: int,
    desc: str,
    image_key: str = "image",
    resize_to: Optional[int] = None,
) -> torch.Tensor:
    rendered = []
    for view_idx in range(camera_params_views.shape[0]):
        rendered.append(
            _render_rgb_batch(
                da,
                z_batch,
                camera_params_views[view_idx : view_idx + 1],
                None if conditioning_params_views is None else conditioning_params_views[view_idx : view_idx + 1],
                neural_rendering_resolution=neural_rendering_resolution,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
                noise_mode=noise_mode,
                batch_size=batch_size,
                desc=f"{desc}/view_{view_idx:02d}",
                image_key=image_key,
                resize_to=resize_to,
            )
        )
    return torch.stack(rendered, dim=1)


def _get_single_img_cam_bundle(da) -> Tuple[torch.Tensor, torch.Tensor]:
    camera_params, conditioning_params = _GEN_UTILS.get_triple_img_cams(da)
    return camera_params[1].unsqueeze(0), conditioning_params[1].unsqueeze(0)


def _infer_dataset_path_from_tuned_run(tuned_pkl: str) -> Optional[str]:
    hydra_cfg = Path(tuned_pkl).resolve().parent / "hydra_cfg.yaml"
    if not hydra_cfg.exists():
        return None
    try:
        import yaml
    except Exception:
        return None
    with open(hydra_cfg, "r") as f:
        cfg = yaml.safe_load(f)
    click_args = cfg.get("click_legacy_args", {}) if isinstance(cfg, dict) else {}
    data_path = click_args.get("data")
    return str(data_path) if data_path else None


def _resolve_dataset_path(cfg: GeometryCompareConfig) -> str:
    dataset_path = cfg.dataset_path or _infer_dataset_path_from_tuned_run(cfg.tuned_pkl)
    if not dataset_path:
        raise ValueError(
            "No dataset path available for empirical viewpoint sampling. "
            "Pass GeometryCompareConfig(dataset_path=...) or ensure hydra_cfg.yaml contains click_legacy_args.data."
        )
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    return dataset_path


def _load_dataset_camera_labels(dataset_path: str) -> np.ndarray:
    path = Path(dataset_path)
    if path.is_dir():
        with open(path / "dataset.json", "r") as f:
            labels = json.load(f)["labels"]
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open("dataset.json", "r") as f:
                labels = json.load(f)["labels"]
    else:
        raise ValueError(f"Unsupported dataset path type: {dataset_path}")

    if labels is None:
        raise ValueError(f"Dataset has no labels in dataset.json: {dataset_path}")
    label_map = dict(labels)
    ordered_keys = sorted(label_map.keys())
    matrix = np.asarray([label_map[k] for k in ordered_keys], dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 25:
        raise ValueError(f"Expected camera labels with shape [N,25], got {matrix.shape}")
    return matrix


def _sample_empirical_camera_params(
    dataset_path: str,
    n_viewpoints: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray]:
    label_matrix = _load_dataset_camera_labels(dataset_path)
    rng = np.random.default_rng(seed)
    n_take = min(n_viewpoints, len(label_matrix))
    idx = rng.choice(len(label_matrix), size=n_take, replace=False)
    return torch.from_numpy(label_matrix[idx]).to(device=device, dtype=torch.float32), idx


def _repeat_conditioning_params(conditioning_params: torch.Tensor, n_viewpoints: int) -> torch.Tensor:
    if conditioning_params.shape[0] != 1:
        raise ValueError(f"Expected singleton conditioning params, got {tuple(conditioning_params.shape)}")
    return conditioning_params.expand(n_viewpoints, -1).contiguous()


def _camera_to_rays(camera_params: torch.Tensor, resolution: int) -> Tuple[torch.Tensor, torch.Tensor]:
    cam2world_matrix = camera_params[:, :16].view(-1, 4, 4)
    intrinsics = camera_params[:, 16:25].view(-1, 3, 3)
    return _RAY_SAMPLER(cam2world_matrix, intrinsics, resolution)


def _depths_to_xyz_maps(
    depths: torch.Tensor,
    camera_params: torch.Tensor,
    radius_cutoff: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if depths.ndim != 3:
        raise ValueError(f"Expected depths with shape [B,H,W], got {tuple(depths.shape)}")
    bsz, nrs, _ = depths.shape
    c_mb = _expand_camera(camera_params.to(depths.device), bsz)
    ray_origins, ray_directions = _camera_to_rays(c_mb, nrs)
    flat_depth = depths.reshape(bsz, -1, 1).expand(-1, -1, 3)
    xyz = ray_origins + flat_depth * ray_directions
    xyz = xyz.reshape(bsz, nrs, nrs, 3)
    if radius_cutoff is None:
        mask = torch.ones((bsz, nrs, nrs), dtype=torch.bool, device=depths.device)
    else:
        mask = depths <= radius_cutoff
    return xyz, mask


def _create_sigma_grid(shape_res: int, cube_length: float) -> torch.Tensor:
    voxel_origin = np.array([0, 0, 0]) - cube_length / 2
    voxel_size = cube_length / (shape_res - 1)
    overall_index = torch.arange(0, shape_res**3, 1, dtype=torch.long)
    samples = torch.zeros(shape_res**3, 3, dtype=torch.float32)
    samples[:, 2] = overall_index % shape_res
    samples[:, 1] = (overall_index.float() / shape_res) % shape_res
    samples[:, 0] = ((overall_index.float() / shape_res) / shape_res) % shape_res
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    return samples.unsqueeze(0)


def _sample_sigma_volume_single(
    da,
    z: torch.Tensor,
    conditioning_params: torch.Tensor,
    sigma_grid: torch.Tensor,
    sigma_resolution: int,
    truncation_psi: float,
    truncation_cutoff: Optional[int],
    noise_mode: str,
    sigma_point_batch: int,
) -> torch.Tensor:
    z = z.view(1, -1)
    c_mb = _expand_camera(conditioning_params, 1)
    sample_mb = sigma_grid.to(z.device)
    sigma_flat = torch.empty(sample_mb.shape[1], dtype=torch.float32)

    with torch.no_grad():
        head = 0
        while head < sample_mb.shape[1]:
            take = min(sigma_point_batch, sample_mb.shape[1] - head)
            dirs = torch.zeros((1, take, 3), device=z.device)
            dirs[..., -1] = -1
            sigma_chunk = da.G.sample(
                sample_mb[:, head : head + take],
                dirs,
                z,
                c_mb,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
                noise_mode=noise_mode,
            )["sigma"][0, :, 0]
            sigma_flat[head : head + take] = sigma_chunk.detach().float().cpu()
            head += take
            del sigma_chunk, dirs
            if z.device.type == "cuda":
                torch.cuda.empty_cache()

    return sigma_flat.reshape(sigma_resolution, sigma_resolution, sigma_resolution)


def _extract_sigma_slices_single(volume: torch.Tensor) -> Dict[str, torch.Tensor]:
    mid = volume.shape[-1] // 2
    return {
        "xy": volume[:, :, mid].to(torch.float16),
        "xz": volume[:, mid, :].to(torch.float16),
        "yz": volume[mid, :, :].to(torch.float16),
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat_vals = values.reshape(values.shape[0], -1)
    flat_mask = mask.reshape(mask.shape[0], -1)
    outs = []
    for idx in range(values.shape[0]):
        cur = flat_vals[idx][flat_mask[idx]]
        if cur.numel() == 0:
            outs.append(torch.tensor(float("nan"), device=values.device))
        else:
            outs.append(cur.mean())
    return torch.stack(outs)


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat_vals = values.reshape(values.shape[0], -1)
    flat_mask = mask.reshape(mask.shape[0], -1)
    outs = []
    for idx in range(values.shape[0]):
        cur = flat_vals[idx][flat_mask[idx]]
        if cur.numel() == 0:
            outs.append(torch.tensor(float("nan"), device=values.device))
        else:
            outs.append(cur.max())
    return torch.stack(outs)


def _flat_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().reshape(a.shape[0], -1).mean(dim=1)


def _flat_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(((a - b) ** 2).reshape(a.shape[0], -1).mean(dim=1))


def _flat_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    return F.cosine_similarity(a_flat, b_flat, dim=1)


def _prepare_rgb_for_display(rgb: torch.Tensor) -> np.ndarray:
    arr = rgb.detach().cpu().permute(1, 2, 0).float().numpy()
    return np.clip((arr + 1.0) / 2.0, 0.0, 1.0)


def _prepare_rgb_diff_for_display(rgb_a: torch.Tensor, rgb_b: torch.Tensor, scale: float = 2.0) -> np.ndarray:
    diff = (rgb_b - rgb_a).detach().cpu().permute(1, 2, 0).abs().float().numpy()
    return np.clip(diff * scale, 0.0, 1.0)


def _load_lpips_model(device: torch.device):
    try:
        import lpips
    except Exception as exc:
        raise RuntimeError("LPIPS is required for perceptual consistency checks in this notebook env.") from exc
    model = lpips.LPIPS(net="vgg").to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _compute_lpips_scores(
    model,
    images_a: torch.Tensor,
    images_b: torch.Tensor,
    batch_size: int,
    device: torch.device,
    desc: str,
) -> torch.Tensor:
    if images_a.shape != images_b.shape:
        raise ValueError(f"LPIPS shape mismatch: {images_a.shape} vs {images_b.shape}")
    scores = []
    with torch.no_grad():
        for start in tqdm(range(0, len(images_a), batch_size), desc=desc):
            img_a = images_a[start : start + batch_size].to(device)
            img_b = images_b[start : start + batch_size].to(device)
            scores.append(model(img_a, img_b).reshape(-1).detach().cpu())
    return torch.cat(scores, dim=0)


def _load_face_identity_model(device: torch.device):
    try:
        from facenet_pytorch import InceptionResnetV1
    except Exception as exc:
        raise RuntimeError("facenet_pytorch is required for identity consistency checks in this notebook env.") from exc
    model = InceptionResnetV1(pretrained="vggface2").to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _compute_face_embeddings(
    model,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
    desc: str,
) -> torch.Tensor:
    embeds = []
    with torch.no_grad():
        for start in tqdm(range(0, len(images), batch_size), desc=desc):
            mb = images[start : start + batch_size].to(device)
            if mb.shape[-1] != 160:
                mb = F.interpolate(mb, size=(160, 160), mode="bilinear", align_corners=False)
            emb = F.normalize(model(mb), dim=1)
            embeds.append(emb.detach().cpu())
    return torch.cat(embeds, dim=0)


def _mean_pairwise_view_cosine(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.ndim != 3:
        raise ValueError(f"Expected embeddings [N,V,D], got {tuple(embeddings.shape)}")
    n_views = embeddings.shape[1]
    if n_views < 2:
        return torch.full((embeddings.shape[0],), float("nan"))
    sims = []
    for i, j in itertools.combinations(range(n_views), 2):
        sims.append(F.cosine_similarity(embeddings[:, i], embeddings[:, j], dim=-1))
    return torch.stack(sims, dim=1).mean(dim=1)


def _project_single_delta_against_mean_pull(
    orig: torch.Tensor,
    tuned: torch.Tensor,
    trunc_ref: torch.Tensor,
) -> Dict[str, float]:
    x = orig.reshape(-1).float()
    y = tuned.reshape(-1).float()
    m = trunc_ref.reshape(-1).float()
    delta = y - x
    mean_pull = m - x
    eps = 1e-12
    denom = float(torch.dot(mean_pull, mean_pull).clamp_min(eps).item())
    alpha = float(torch.dot(delta, mean_pull).item() / denom)
    parallel = alpha * mean_pull
    residual = delta - parallel
    delta_norm = float(torch.linalg.norm(delta).clamp_min(eps).item())
    mean_pull_norm = float(torch.linalg.norm(mean_pull).clamp_min(eps).item())
    residual_norm = float(torch.linalg.norm(residual).item())
    cosine = float((torch.dot(delta, mean_pull).item()) / (delta_norm * mean_pull_norm))
    return {
        "alpha": alpha,
        "parallel_norm": float(torch.linalg.norm(parallel).item()),
        "residual_norm": residual_norm,
        "residual_fraction": residual_norm / delta_norm,
        "cosine_to_mean_pull": float(np.clip(cosine, -1.0, 1.0)),
    }


def _run_sigma_analysis_streaming(
    cfg: GeometryCompareConfig,
    device: torch.device,
    orig_da,
    tuned_da,
    z_batch: torch.Tensor,
    z_zero: torch.Tensor,
    orig_conditioning_params: torch.Tensor,
    tuned_conditioning_params: torch.Tensor,
) -> Dict[str, Any]:
    sigma_cube_length = float(orig_da.G.rendering_kwargs["box_warp"])
    sigma_grid = _create_sigma_grid(cfg.sigma_resolution, cube_length=sigma_cube_length)

    baseline_specs = {
        "zeroz_orig": (orig_da, orig_conditioning_params, z_zero, 1.0),
        "zeroz_tuned": (tuned_da, tuned_conditioning_params, z_zero, 1.0),
        "trunc0_orig": (orig_da, orig_conditioning_params, z_zero, 0.0),
        "trunc0_tuned": (tuned_da, tuned_conditioning_params, z_zero, 0.0),
    }
    baseline_volumes: Dict[str, torch.Tensor] = {}
    baseline_slices: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, (da, cond, z_src, trunc_psi) in baseline_specs.items():
        vol = _sample_sigma_volume_single(
            da,
            z_src[0],
            cond,
            sigma_grid=sigma_grid,
            sigma_resolution=cfg.sigma_resolution,
            truncation_psi=trunc_psi,
            truncation_cutoff=cfg.truncation_cutoff,
            noise_mode=cfg.noise_mode,
            sigma_point_batch=cfg.sigma_point_batch,
        )
        baseline_volumes[key] = vol
        baseline_slices[key] = {plane: slc.unsqueeze(0) for plane, slc in _extract_sigma_slices_single(vol).items()}

    slice_store = {
        "orig": {plane: [] for plane in ("xy", "xz", "yz")},
        "tuned": {plane: [] for plane in ("xy", "xz", "yz")},
        **baseline_slices,
    }
    metrics = {
        "delta_l1": [],
        "delta_l2": [],
        "tuned_to_orig_trunc0_l1": [],
        "move_toward_trunc0_cos": [],
        "to_trunc0_orig_l1": [],
        "to_trunc0_tuned_l1": [],
        "to_zeroz_orig_l1": [],
        "to_zeroz_tuned_l1": [],
    }
    projection = {
        "alpha": [],
        "parallel_norm": [],
        "residual_norm": [],
        "residual_fraction": [],
        "cosine_to_mean_pull": [],
    }

    for sample_idx in tqdm(range(len(z_batch)), desc="Sigma/streamed paired analysis"):
        orig_vol = _sample_sigma_volume_single(
            orig_da,
            z_batch[sample_idx],
            orig_conditioning_params,
            sigma_grid=sigma_grid,
            sigma_resolution=cfg.sigma_resolution,
            truncation_psi=cfg.truncation_psi,
            truncation_cutoff=cfg.truncation_cutoff,
            noise_mode=cfg.noise_mode,
            sigma_point_batch=cfg.sigma_point_batch,
        )
        tuned_vol = _sample_sigma_volume_single(
            tuned_da,
            z_batch[sample_idx],
            tuned_conditioning_params,
            sigma_grid=sigma_grid,
            sigma_resolution=cfg.sigma_resolution,
            truncation_psi=cfg.truncation_psi,
            truncation_cutoff=cfg.truncation_cutoff,
            noise_mode=cfg.noise_mode,
            sigma_point_batch=cfg.sigma_point_batch,
        )

        for plane, slc in _extract_sigma_slices_single(orig_vol).items():
            slice_store["orig"][plane].append(slc)
        for plane, slc in _extract_sigma_slices_single(tuned_vol).items():
            slice_store["tuned"][plane].append(slc)

        delta = tuned_vol - orig_vol
        metrics["delta_l1"].append(float(delta.abs().mean().item()))
        metrics["delta_l2"].append(float(torch.sqrt((delta.square()).mean()).item()))
        metrics["tuned_to_orig_trunc0_l1"].append(float((tuned_vol - baseline_volumes["trunc0_orig"]).abs().mean().item()))
        metrics["to_trunc0_orig_l1"].append(float((orig_vol - baseline_volumes["trunc0_orig"]).abs().mean().item()))
        metrics["to_trunc0_tuned_l1"].append(float((tuned_vol - baseline_volumes["trunc0_tuned"]).abs().mean().item()))
        metrics["to_zeroz_orig_l1"].append(float((orig_vol - baseline_volumes["zeroz_orig"]).abs().mean().item()))
        metrics["to_zeroz_tuned_l1"].append(float((tuned_vol - baseline_volumes["zeroz_tuned"]).abs().mean().item()))
        mean_pull = baseline_volumes["trunc0_orig"] - orig_vol
        metrics["move_toward_trunc0_cos"].append(float(F.cosine_similarity(delta.reshape(1, -1), mean_pull.reshape(1, -1)).item()))

        proj = _project_single_delta_against_mean_pull(orig_vol, tuned_vol, baseline_volumes["trunc0_orig"])
        for key, value in proj.items():
            projection[key].append(value)

        del orig_vol, tuned_vol, delta, mean_pull
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for key in ("orig", "tuned"):
        for plane in ("xy", "xz", "yz"):
            slice_store[key][plane] = torch.stack(slice_store[key][plane], dim=0)

    return {
        "streamed": True,
        "resolution": cfg.sigma_resolution,
        "slices": slice_store,
        "metrics": {k: np.asarray(v, dtype=np.float32) for k, v in metrics.items()},
        "projection": {k: np.asarray(v, dtype=np.float32) for k, v in projection.items()},
    }


def _masked_point_metrics(
    xyz_a: torch.Tensor,
    xyz_b: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    common = mask_a & mask_b
    disp = torch.linalg.norm(xyz_a - xyz_b, dim=-1)
    return {
        "common_mask": common,
        "disp": disp,
        "mean_disp": _masked_mean(disp, common),
    }


def _baseline_stack(baseline: torch.Tensor, n_samples: int) -> torch.Tensor:
    return baseline.expand(n_samples, *baseline.shape[1:])


def _sigma_has_full_volumes(sigma: Dict[str, Any]) -> bool:
    return torch.is_tensor(sigma.get("orig")) and sigma["orig"].ndim == 4


def _get_sigma_plane_tensor(sigma: Dict[str, Any], key: str, plane: str) -> torch.Tensor:
    if _sigma_has_full_volumes(sigma):
        return _sigma_mid_slices(sigma[key])[plane]
    return sigma["slices"][key][plane].float()


def _run_perceptual_identity_analysis(
    cfg: GeometryCompareConfig,
    device: torch.device,
    orig_da,
    tuned_da,
    z_batch: torch.Tensor,
    separate_conditioning: bool = True,
) -> Dict[str, Any]:
    dataset_path = _resolve_dataset_path(cfg)
    canonical_camera_orig, canonical_conditioning_orig = _get_single_img_cam_bundle(orig_da)
    canonical_camera_tuned, canonical_conditioning_tuned = _get_single_img_cam_bundle(tuned_da)
    canonical_camera_orig = canonical_camera_orig.to(device)
    canonical_conditioning_orig = canonical_conditioning_orig.to(device)
    canonical_camera_tuned = canonical_camera_tuned.to(device)
    canonical_conditioning_tuned = canonical_conditioning_tuned.to(device)
    empirical_cameras, sampled_view_indices = _sample_empirical_camera_params(
        dataset_path=dataset_path,
        n_viewpoints=cfg.n_viewpoints,
        seed=cfg.viewpoint_seed,
        device=device,
    )
    empirical_conditioning_orig = _repeat_conditioning_params(canonical_conditioning_orig, empirical_cameras.shape[0])
    empirical_conditioning_tuned = _repeat_conditioning_params(canonical_conditioning_tuned, empirical_cameras.shape[0])

    render_kwargs = dict(
        neural_rendering_resolution=cfg.depth_resolution,
        truncation_psi=cfg.truncation_psi,
        truncation_cutoff=cfg.truncation_cutoff,
        noise_mode=cfg.noise_mode,
        batch_size=cfg.model_batch_size,
        image_key=cfg.rgb_model_key,
        resize_to=cfg.rgb_metric_resolution,
    )

    rgb_orig_canonical = _render_rgb_batch(
        orig_da,
        z_batch,
        canonical_camera_orig,
        canonical_conditioning_orig if separate_conditioning else None,
        desc=f"RGB/orig canonical ({'split-c' if separate_conditioning else 'shared-c'})",
        **render_kwargs,
    )
    rgb_tuned_canonical = _render_rgb_batch(
        tuned_da,
        z_batch,
        canonical_camera_tuned,
        canonical_conditioning_tuned if separate_conditioning else None,
        desc=f"RGB/tuned canonical ({'split-c' if separate_conditioning else 'shared-c'})",
        **render_kwargs,
    )
    rgb_orig_views = _render_rgb_multiview(
        orig_da,
        z_batch,
        empirical_cameras,
        empirical_conditioning_orig if separate_conditioning else None,
        desc=f"RGB/orig empirical ({'split-c' if separate_conditioning else 'shared-c'})",
        **render_kwargs,
    )
    rgb_tuned_views = _render_rgb_multiview(
        tuned_da,
        z_batch,
        empirical_cameras,
        empirical_conditioning_tuned if separate_conditioning else None,
        desc=f"RGB/tuned empirical ({'split-c' if separate_conditioning else 'shared-c'})",
        **render_kwargs,
    )

    metric_batch_size = max(1, cfg.identity_batch_size)

    lpips_model = _load_lpips_model(device)
    lpips_canonical = _compute_lpips_scores(
        lpips_model,
        rgb_orig_canonical,
        rgb_tuned_canonical,
        batch_size=metric_batch_size,
        device=device,
        desc="LPIPS canonical",
    )
    lpips_views = _compute_lpips_scores(
        lpips_model,
        rgb_orig_views.reshape(-1, *rgb_orig_views.shape[2:]),
        rgb_tuned_views.reshape(-1, *rgb_tuned_views.shape[2:]),
        batch_size=metric_batch_size,
        device=device,
        desc="LPIPS empirical views",
    ).reshape(len(z_batch), empirical_cameras.shape[0])
    del lpips_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    face_model = _load_face_identity_model(device)
    emb_orig_canonical = _compute_face_embeddings(
        face_model,
        rgb_orig_canonical,
        batch_size=metric_batch_size,
        device=device,
        desc="ID/orig canonical",
    )
    emb_tuned_canonical = _compute_face_embeddings(
        face_model,
        rgb_tuned_canonical,
        batch_size=metric_batch_size,
        device=device,
        desc="ID/tuned canonical",
    )
    emb_orig_views = _compute_face_embeddings(
        face_model,
        rgb_orig_views.reshape(-1, *rgb_orig_views.shape[2:]),
        batch_size=metric_batch_size,
        device=device,
        desc="ID/orig empirical views",
    ).reshape(len(z_batch), empirical_cameras.shape[0], -1)
    emb_tuned_views = _compute_face_embeddings(
        face_model,
        rgb_tuned_views.reshape(-1, *rgb_tuned_views.shape[2:]),
        batch_size=metric_batch_size,
        device=device,
        desc="ID/tuned empirical views",
    ).reshape(len(z_batch), empirical_cameras.shape[0], -1)
    del face_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    orig_view_to_canonical = F.cosine_similarity(
        emb_orig_views,
        emb_orig_canonical.unsqueeze(1).expand(-1, emb_orig_views.shape[1], -1),
        dim=-1,
    )
    tuned_view_to_canonical = F.cosine_similarity(
        emb_tuned_views,
        emb_tuned_canonical.unsqueeze(1).expand(-1, emb_tuned_views.shape[1], -1),
        dim=-1,
    )
    cross_model_view_cos = F.cosine_similarity(emb_orig_views, emb_tuned_views, dim=-1)
    cross_model_canonical_cos = F.cosine_similarity(emb_orig_canonical, emb_tuned_canonical, dim=-1)

    return {
        "dataset_path": dataset_path,
        "separate_conditioning": separate_conditioning,
        "rgb_key": cfg.rgb_model_key,
        "rgb_metric_resolution": cfg.rgb_metric_resolution,
        "viewpoint_seed": cfg.viewpoint_seed,
        "sampled_view_indices": sampled_view_indices,
        "camera_params": {
            "canonical_orig": canonical_camera_orig.detach().cpu(),
            "canonical_tuned": canonical_camera_tuned.detach().cpu(),
            "empirical": empirical_cameras.detach().cpu(),
            "canonical_conditioning_orig": canonical_conditioning_orig.detach().cpu(),
            "canonical_conditioning_tuned": canonical_conditioning_tuned.detach().cpu(),
            "empirical_conditioning_orig": empirical_conditioning_orig.detach().cpu(),
            "empirical_conditioning_tuned": empirical_conditioning_tuned.detach().cpu(),
        },
        "rgb": {
            "orig_canonical": rgb_orig_canonical,
            "tuned_canonical": rgb_tuned_canonical,
            "orig_views": rgb_orig_views,
            "tuned_views": rgb_tuned_views,
        },
        "lpips": {
            "canonical": lpips_canonical,
            "views": lpips_views,
        },
        "identity": {
            "orig_canonical_embeddings": emb_orig_canonical,
            "tuned_canonical_embeddings": emb_tuned_canonical,
            "orig_view_embeddings": emb_orig_views,
            "tuned_view_embeddings": emb_tuned_views,
            "orig_view_to_canonical_cos": orig_view_to_canonical,
            "tuned_view_to_canonical_cos": tuned_view_to_canonical,
            "orig_view_pairwise_cos_mean": _mean_pairwise_view_cosine(emb_orig_views),
            "tuned_view_pairwise_cos_mean": _mean_pairwise_view_cosine(emb_tuned_views),
            "cross_model_canonical_cos": cross_model_canonical_cos,
            "cross_model_view_cos": cross_model_view_cos,
        },
    }


def _build_perceptual_summary(perceptual: Dict[str, Any]) -> pd.DataFrame:
    lpips_scores = perceptual["lpips"]
    identity = perceptual["identity"]
    out = pd.DataFrame(
        {
            "sample_idx": np.arange(lpips_scores["canonical"].shape[0]),
            "lpips_cross_model_canonical": lpips_scores["canonical"].cpu().numpy(),
            "lpips_cross_model_views_mean": lpips_scores["views"].mean(dim=1).cpu().numpy(),
            "lpips_cross_model_views_max": lpips_scores["views"].max(dim=1).values.cpu().numpy(),
            "id_orig_view_to_canonical_cos_mean": identity["orig_view_to_canonical_cos"].mean(dim=1).cpu().numpy(),
            "id_tuned_view_to_canonical_cos_mean": identity["tuned_view_to_canonical_cos"].mean(dim=1).cpu().numpy(),
            "id_orig_view_pairwise_cos_mean": identity["orig_view_pairwise_cos_mean"].cpu().numpy(),
            "id_tuned_view_pairwise_cos_mean": identity["tuned_view_pairwise_cos_mean"].cpu().numpy(),
            "id_cross_model_canonical_cos": identity["cross_model_canonical_cos"].cpu().numpy(),
            "id_cross_model_views_mean": identity["cross_model_view_cos"].mean(dim=1).cpu().numpy(),
            "id_cross_model_views_min": identity["cross_model_view_cos"].min(dim=1).values.cpu().numpy(),
        }
    )
    out["id_view_to_canonical_consistency_delta"] = (
        out["id_tuned_view_to_canonical_cos_mean"] - out["id_orig_view_to_canonical_cos_mean"]
    )
    out["id_view_pairwise_consistency_delta"] = (
        out["id_tuned_view_pairwise_cos_mean"] - out["id_orig_view_pairwise_cos_mean"]
    )
    out["id_cross_model_canonical_drift"] = 1.0 - out["id_cross_model_canonical_cos"]
    out["id_cross_model_views_mean_drift"] = 1.0 - out["id_cross_model_views_mean"]
    return out


def _build_summary(results: Dict[str, Any]) -> pd.DataFrame:
    depth = results["depth"]
    pointcloud = results["pointcloud"]
    sigma = results["sigma"]

    depth_delta = depth["tuned"] - depth["orig"]
    depth_to_orig_trunc0 = _flat_l1(depth["orig"], _baseline_stack(depth["trunc0_orig"], len(depth["orig"])))
    depth_to_tuned_trunc0 = _flat_l1(depth["tuned"], _baseline_stack(depth["trunc0_tuned"], len(depth["tuned"])))
    depth_to_orig_zeroz = _flat_l1(depth["orig"], _baseline_stack(depth["zeroz_orig"], len(depth["orig"])))
    depth_to_tuned_zeroz = _flat_l1(depth["tuned"], _baseline_stack(depth["zeroz_tuned"], len(depth["tuned"])))

    if _sigma_has_full_volumes(sigma):
        sigma_delta = sigma["tuned"] - sigma["orig"]
        sigma_to_orig_trunc0 = _flat_l1(sigma["orig"], _baseline_stack(sigma["trunc0_orig"], len(sigma["orig"])))
        sigma_to_tuned_trunc0 = _flat_l1(sigma["tuned"], _baseline_stack(sigma["trunc0_tuned"], len(sigma["tuned"])))
        sigma_to_orig_zeroz = _flat_l1(sigma["orig"], _baseline_stack(sigma["zeroz_orig"], len(sigma["orig"])))
        sigma_to_tuned_zeroz = _flat_l1(sigma["tuned"], _baseline_stack(sigma["zeroz_tuned"], len(sigma["tuned"])))
        sigma_metrics = {
            "delta_l1": _flat_l1(sigma["orig"], sigma["tuned"]).cpu().numpy(),
            "delta_l2": _flat_l2(sigma["orig"], sigma["tuned"]).cpu().numpy(),
            "tuned_to_orig_trunc0_l1": _flat_l1(
                sigma["tuned"],
                _baseline_stack(sigma["trunc0_orig"], len(sigma["tuned"])),
            ).cpu().numpy(),
            "move_toward_trunc0_cos": _flat_cosine(
                sigma_delta,
                _baseline_stack(sigma["trunc0_orig"], len(sigma["orig"])) - sigma["orig"],
            ).cpu().numpy(),
            "to_trunc0_orig_l1": sigma_to_orig_trunc0.cpu().numpy(),
            "to_trunc0_tuned_l1": sigma_to_tuned_trunc0.cpu().numpy(),
            "to_zeroz_orig_l1": sigma_to_orig_zeroz.cpu().numpy(),
            "to_zeroz_tuned_l1": sigma_to_tuned_zeroz.cpu().numpy(),
        }
    else:
        sigma_metrics = sigma["metrics"]

    point_baseline_orig = _masked_point_metrics(
        pointcloud["xyz_orig"],
        _baseline_stack(pointcloud["xyz_trunc0_orig"], len(pointcloud["xyz_orig"])),
        pointcloud["mask_orig"],
        _baseline_stack(pointcloud["mask_trunc0_orig"], len(pointcloud["mask_orig"])),
    )["mean_disp"]
    point_baseline_tuned = _masked_point_metrics(
        pointcloud["xyz_tuned"],
        _baseline_stack(pointcloud["xyz_trunc0_tuned"], len(pointcloud["xyz_tuned"])),
        pointcloud["mask_tuned"],
        _baseline_stack(pointcloud["mask_trunc0_tuned"], len(pointcloud["mask_tuned"])),
    )["mean_disp"]
    point_baseline_orig_zeroz = _masked_point_metrics(
        pointcloud["xyz_orig"],
        _baseline_stack(pointcloud["xyz_zeroz_orig"], len(pointcloud["xyz_orig"])),
        pointcloud["mask_orig"],
        _baseline_stack(pointcloud["mask_zeroz_orig"], len(pointcloud["mask_orig"])),
    )["mean_disp"]
    point_baseline_tuned_zeroz = _masked_point_metrics(
        pointcloud["xyz_tuned"],
        _baseline_stack(pointcloud["xyz_zeroz_tuned"], len(pointcloud["xyz_tuned"])),
        pointcloud["mask_tuned"],
        _baseline_stack(pointcloud["mask_zeroz_tuned"], len(pointcloud["mask_tuned"])),
    )["mean_disp"]
    point_tuned_to_orig_trunc0 = _masked_point_metrics(
        pointcloud["xyz_tuned"],
        _baseline_stack(pointcloud["xyz_trunc0_orig"], len(pointcloud["xyz_tuned"])),
        pointcloud["mask_tuned"],
        _baseline_stack(pointcloud["mask_trunc0_orig"], len(pointcloud["mask_tuned"])),
    )["mean_disp"]
    z_shift = pointcloud["xyz_tuned"][..., 2] - pointcloud["xyz_orig"][..., 2]
    common_mask = pointcloud["pair_metrics"]["common_mask"]
    mean_abs_z_shift = _masked_mean(z_shift.abs(), common_mask)
    max_abs_z_shift = _masked_max(z_shift.abs(), common_mask)

    out = pd.DataFrame(
        {
            "sample_idx": np.arange(len(depth["orig"])),
            "depth_delta_l1": _flat_l1(depth["orig"], depth["tuned"]).cpu().numpy(),
            "depth_delta_l2": _flat_l2(depth["orig"], depth["tuned"]).cpu().numpy(),
            "depth_tuned_to_orig_trunc0_l1": _flat_l1(
                depth["tuned"],
                _baseline_stack(depth["trunc0_orig"], len(depth["tuned"])),
            ).cpu().numpy(),
            "depth_move_toward_trunc0_cos": _flat_cosine(
                depth_delta,
                _baseline_stack(depth["trunc0_orig"], len(depth["orig"])) - depth["orig"],
            ).cpu().numpy(),
            "depth_to_trunc0_orig_l1": depth_to_orig_trunc0.cpu().numpy(),
            "depth_to_trunc0_tuned_l1": depth_to_tuned_trunc0.cpu().numpy(),
            "depth_to_zeroz_orig_l1": depth_to_orig_zeroz.cpu().numpy(),
            "depth_to_zeroz_tuned_l1": depth_to_tuned_zeroz.cpu().numpy(),
            "point_delta_l2": pointcloud["pair_metrics"]["mean_disp"].cpu().numpy(),
            "point_tuned_to_orig_trunc0_l2": point_tuned_to_orig_trunc0.cpu().numpy(),
            "point_to_trunc0_orig_l2": point_baseline_orig.cpu().numpy(),
            "point_to_trunc0_tuned_l2": point_baseline_tuned.cpu().numpy(),
            "point_to_zeroz_orig_l2": point_baseline_orig_zeroz.cpu().numpy(),
            "point_to_zeroz_tuned_l2": point_baseline_tuned_zeroz.cpu().numpy(),
            "point_mean_abs_delta_z": mean_abs_z_shift.cpu().numpy(),
            "point_max_abs_delta_z": max_abs_z_shift.cpu().numpy(),
            "sigma_delta_l1": sigma_metrics["delta_l1"],
            "sigma_delta_l2": sigma_metrics["delta_l2"],
            "sigma_tuned_to_orig_trunc0_l1": sigma_metrics["tuned_to_orig_trunc0_l1"],
            "sigma_move_toward_trunc0_cos": sigma_metrics["move_toward_trunc0_cos"],
            "sigma_to_trunc0_orig_l1": sigma_metrics["to_trunc0_orig_l1"],
            "sigma_to_trunc0_tuned_l1": sigma_metrics["to_trunc0_tuned_l1"],
            "sigma_to_zeroz_orig_l1": sigma_metrics["to_zeroz_orig_l1"],
            "sigma_to_zeroz_tuned_l1": sigma_metrics["to_zeroz_tuned_l1"],
        }
    )
    out["depth_trunc0_pull_ratio"] = out["depth_to_trunc0_tuned_l1"] / out["depth_to_trunc0_orig_l1"]
    out["point_trunc0_pull_ratio"] = out["point_to_trunc0_tuned_l2"] / out["point_to_trunc0_orig_l2"]
    out["sigma_trunc0_pull_ratio"] = out["sigma_to_trunc0_tuned_l1"] / out["sigma_to_trunc0_orig_l1"]

    if "perceptual" in results:
        out = out.join(_build_perceptual_summary(results["perceptual"]).set_index("sample_idx"), on="sample_idx")
    return out


def _project_delta_against_mean_pull(
    orig: torch.Tensor,
    tuned: torch.Tensor,
    trunc_ref: torch.Tensor,
) -> Dict[str, np.ndarray]:
    x = orig.reshape(orig.shape[0], -1).float()
    y = tuned.reshape(tuned.shape[0], -1).float()
    m = _baseline_stack(trunc_ref, len(orig)).reshape(orig.shape[0], -1).float()

    delta = y - x
    mean_pull = m - x

    eps = 1e-12
    denom = (mean_pull * mean_pull).sum(dim=1).clamp_min(eps)
    alpha = (delta * mean_pull).sum(dim=1) / denom
    parallel = alpha.unsqueeze(1) * mean_pull
    residual = delta - parallel
    delta_norm = torch.linalg.norm(delta, dim=1).clamp_min(eps)
    mean_pull_norm = torch.linalg.norm(mean_pull, dim=1).clamp_min(eps)
    residual_norm = torch.linalg.norm(residual, dim=1)
    parallel_norm = torch.linalg.norm(parallel, dim=1)
    cosine = ((delta * mean_pull).sum(dim=1) / (delta_norm * mean_pull_norm)).clamp(-1.0, 1.0)

    return {
        "alpha": alpha.cpu().numpy(),
        "parallel_norm": parallel_norm.cpu().numpy(),
        "residual_norm": residual_norm.cpu().numpy(),
        "residual_fraction": (residual_norm / delta_norm).cpu().numpy(),
        "cosine_to_mean_pull": cosine.cpu().numpy(),
    }


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(vals, size=vals.size, replace=True)
        boots[i] = sample.mean()
    return {
        "mean": float(vals.mean()),
        "ci_low": float(np.quantile(boots, alpha / 2)),
        "ci_high": float(np.quantile(boots, 1 - alpha / 2)),
    }


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _paired_closeness_stats(
    tuned_to_orig: np.ndarray,
    tuned_to_trunc: np.ndarray,
    projection_stats: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    tuned_to_orig = np.asarray(tuned_to_orig, dtype=float)
    tuned_to_trunc = np.asarray(tuned_to_trunc, dtype=float)
    margin = tuned_to_trunc - tuned_to_orig
    finite = np.isfinite(margin)
    tuned_to_orig = tuned_to_orig[finite]
    tuned_to_trunc = tuned_to_trunc[finite]
    margin = margin[finite]

    out: Dict[str, Any] = {}
    out["n"] = int(len(margin))
    out["fraction_closer_to_original"] = float(np.mean(tuned_to_orig < tuned_to_trunc))
    out["fraction_closer_to_trunc0"] = float(np.mean(tuned_to_trunc < tuned_to_orig))
    out["mean_margin_trunc_minus_orig"] = _bootstrap_mean_ci(margin)
    out["mean_tuned_to_original"] = _bootstrap_mean_ci(tuned_to_orig)
    out["mean_tuned_to_trunc0"] = _bootstrap_mean_ci(tuned_to_trunc)
    out["projection_alpha"] = _bootstrap_mean_ci(projection_stats["alpha"])
    out["residual_fraction"] = _bootstrap_mean_ci(projection_stats["residual_fraction"])
    out["cosine_to_mean_pull"] = _bootstrap_mean_ci(projection_stats["cosine_to_mean_pull"])
    out["interpretation"] = (
        "closer_to_original"
        if out["mean_margin_trunc_minus_orig"]["mean"] > 0
        else "closer_to_trunc0"
    )

    try:
        from scipy import stats

        nz_mask = tuned_to_orig != tuned_to_trunc
        if np.any(nz_mask):
            wilcoxon_res = stats.wilcoxon(tuned_to_trunc[nz_mask], tuned_to_orig[nz_mask], alternative="greater")
            out["wilcoxon_trunc_gt_orig_p"] = _safe_float(wilcoxon_res.pvalue)
        else:
            out["wilcoxon_trunc_gt_orig_p"] = None

        n_success = int(np.sum(tuned_to_orig < tuned_to_trunc))
        n_trials = int(np.sum(tuned_to_orig != tuned_to_trunc))
        if n_trials > 0:
            sign_res = stats.binomtest(n_success, n_trials, p=0.5, alternative="greater")
            out["sign_test_closer_to_original_p"] = _safe_float(sign_res.pvalue)
        else:
            out["sign_test_closer_to_original_p"] = None
    except Exception:
        out["wilcoxon_trunc_gt_orig_p"] = None
        out["sign_test_closer_to_original_p"] = None
    return out


def _paired_delta_stats(a: np.ndarray, b: np.ndarray, seed: int = 0) -> Dict[str, Any]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    delta = b - a

    out: Dict[str, Any] = {
        "n": int(len(delta)),
        "a_mean": _bootstrap_mean_ci(a, seed=seed),
        "b_mean": _bootstrap_mean_ci(b, seed=seed + 1),
        "delta_b_minus_a": _bootstrap_mean_ci(delta, seed=seed + 2),
    }
    try:
        from scipy import stats

        nz_mask = delta != 0
        if np.any(nz_mask):
            res = stats.wilcoxon(b[nz_mask], a[nz_mask])
            out["wilcoxon_p"] = _safe_float(res.pvalue)
        else:
            out["wilcoxon_p"] = None
    except Exception:
        out["wilcoxon_p"] = None
    return out


def _correlation_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    out: Dict[str, Any] = {"n": int(len(x))}
    if len(x) < 3:
        out["spearman_r"] = None
        out["spearman_p"] = None
        out["pearson_r"] = None
        return out
    try:
        from scipy import stats

        spear = stats.spearmanr(x, y)
        out["spearman_r"] = _safe_float(spear.statistic)
        out["spearman_p"] = _safe_float(spear.pvalue)
    except Exception:
        out["spearman_r"] = None
        out["spearman_p"] = None
    pear = np.corrcoef(x, y)[0, 1]
    out["pearson_r"] = _safe_float(pear)
    return out


def compute_perceptual_identity_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    summary = results["summary"]

    out = {
        "lpips_cross_model_canonical": _bootstrap_mean_ci(summary["lpips_cross_model_canonical"].to_numpy()),
        "lpips_cross_model_views_mean": _bootstrap_mean_ci(summary["lpips_cross_model_views_mean"].to_numpy()),
        "identity_view_to_canonical_consistency": {
            "orig_vs_tuned": _paired_delta_stats(
                summary["id_orig_view_to_canonical_cos_mean"].to_numpy(),
                summary["id_tuned_view_to_canonical_cos_mean"].to_numpy(),
            ),
        },
        "identity_view_pairwise_consistency": {
            "orig_vs_tuned": _paired_delta_stats(
                summary["id_orig_view_pairwise_cos_mean"].to_numpy(),
                summary["id_tuned_view_pairwise_cos_mean"].to_numpy(),
                seed=17,
            ),
        },
        "identity_cross_model": {
            "canonical_cos": _bootstrap_mean_ci(summary["id_cross_model_canonical_cos"].to_numpy()),
            "views_mean_cos": _bootstrap_mean_ci(summary["id_cross_model_views_mean"].to_numpy()),
            "views_min_cos": _bootstrap_mean_ci(summary["id_cross_model_views_min"].to_numpy()),
        },
        "identity_vs_geometry": {
            "canonical_drift_vs_depth_delta_l1": _correlation_stats(
                summary["id_cross_model_canonical_drift"].to_numpy(),
                summary["depth_delta_l1"].to_numpy(),
            ),
            "canonical_drift_vs_point_max_abs_delta_z": _correlation_stats(
                summary["id_cross_model_canonical_drift"].to_numpy(),
                summary["point_max_abs_delta_z"].to_numpy(),
            ),
            "canonical_drift_vs_sigma_delta_l1": _correlation_stats(
                summary["id_cross_model_canonical_drift"].to_numpy(),
                summary["sigma_delta_l1"].to_numpy(),
            ),
            "view_mean_drift_vs_depth_delta_l1": _correlation_stats(
                summary["id_cross_model_views_mean_drift"].to_numpy(),
                summary["depth_delta_l1"].to_numpy(),
            ),
            "view_mean_drift_vs_point_max_abs_delta_z": _correlation_stats(
                summary["id_cross_model_views_mean_drift"].to_numpy(),
                summary["point_max_abs_delta_z"].to_numpy(),
            ),
            "view_mean_drift_vs_sigma_delta_l1": _correlation_stats(
                summary["id_cross_model_views_mean_drift"].to_numpy(),
                summary["sigma_delta_l1"].to_numpy(),
            ),
        },
    }
    return out


def _conditioning_mode_metric_stats(
    legacy_summary: pd.DataFrame,
    corrected_summary: pd.DataFrame,
    metric: str,
    seed: int = 0,
) -> Dict[str, Any]:
    legacy = legacy_summary[metric].to_numpy()
    corrected = corrected_summary[metric].to_numpy()
    return {
        "legacy": _bootstrap_mean_ci(legacy, seed=seed),
        "corrected": _bootstrap_mean_ci(corrected, seed=seed + 1),
        "delta_corrected_minus_legacy": _paired_delta_stats(legacy, corrected, seed=seed + 2),
    }


def run_perceptual_conditioning_mode_comparison(cfg: GeometryCompareConfig) -> Dict[str, Any]:
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this Python process. "
            "This comparison expects a GPU-visible kernel for EG3D checkpoint loading and perceptual rendering."
        )

    device = torch.device(cfg.device)
    orig_da = _GEN_UTILS.load_generator(cfg.orig_pkl, truncation_psi=cfg.truncation_psi, truncation_cutoff=cfg.truncation_cutoff)
    tuned_da = _GEN_UTILS.load_generator(cfg.tuned_pkl, truncation_psi=cfg.truncation_psi, truncation_cutoff=cfg.truncation_cutoff)
    z_batch = _shared_latents(cfg.n_samples, cfg.latent_seed, orig_da.G.z_dim, device=device)

    legacy = _run_perceptual_identity_analysis(
        cfg=cfg,
        device=device,
        orig_da=orig_da,
        tuned_da=tuned_da,
        z_batch=z_batch,
        separate_conditioning=False,
    )
    corrected = _run_perceptual_identity_analysis(
        cfg=cfg,
        device=device,
        orig_da=orig_da,
        tuned_da=tuned_da,
        z_batch=z_batch,
        separate_conditioning=True,
    )

    legacy_summary = _build_perceptual_summary(legacy)
    corrected_summary = _build_perceptual_summary(corrected)
    comparison = legacy_summary.merge(corrected_summary, on="sample_idx", suffixes=("_legacy_shared_c", "_corrected_split_c"))

    metrics = [
        "lpips_cross_model_canonical",
        "lpips_cross_model_views_mean",
        "lpips_cross_model_views_max",
        "id_orig_view_to_canonical_cos_mean",
        "id_tuned_view_to_canonical_cos_mean",
        "id_orig_view_pairwise_cos_mean",
        "id_tuned_view_pairwise_cos_mean",
        "id_cross_model_canonical_cos",
        "id_cross_model_views_mean",
        "id_cross_model_views_min",
        "id_view_to_canonical_consistency_delta",
        "id_view_pairwise_consistency_delta",
    ]
    for metric in metrics:
        comparison[f"{metric}_delta_corrected_minus_legacy"] = (
            comparison[f"{metric}_corrected_split_c"] - comparison[f"{metric}_legacy_shared_c"]
        )

    stats = {
        metric: _conditioning_mode_metric_stats(legacy_summary, corrected_summary, metric, seed=17 * (idx + 1))
        for idx, metric in enumerate(metrics)
    }

    return {
        "config": asdict(cfg),
        "results_dir": str(_default_results_dir(cfg)),
        "legacy_shared_c": legacy,
        "corrected_split_c": corrected,
        "legacy_summary": legacy_summary,
        "corrected_summary": corrected_summary,
        "comparison": comparison,
        "stats": stats,
    }


def save_perceptual_conditioning_mode_comparison(results: Dict[str, Any], out_dir: Optional[str] = None) -> Path:
    target = Path(out_dir) if out_dir is not None else Path(results["results_dir"])
    compare_dir = target / "conditioning_mode_comparison"
    if compare_dir.exists():
        shutil.rmtree(compare_dir)
    compare_dir.mkdir(parents=True, exist_ok=True)

    results["legacy_summary"].to_csv(compare_dir / "legacy_shared_c_summary.csv", index=False)
    results["corrected_summary"].to_csv(compare_dir / "corrected_split_c_summary.csv", index=False)
    results["comparison"].to_csv(compare_dir / "conditioning_mode_comparison.csv", index=False)
    with open(compare_dir / "conditioning_mode_comparison_stats.json", "w") as f:
        json.dump(_jsonable({"config": results["config"], "stats": results["stats"]}), f, indent=2)
    return compare_dir


def compute_geometry_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    depth = results["depth"]
    sigma = results["sigma"]
    summary = results["summary"]

    depth_proj = _project_delta_against_mean_pull(depth["orig"], depth["tuned"], depth["trunc0_orig"])
    if _sigma_has_full_volumes(sigma):
        sigma_proj = _project_delta_against_mean_pull(sigma["orig"], sigma["tuned"], sigma["trunc0_orig"])
    else:
        sigma_proj = sigma["projection"]

    stats_out = {
        "depth": _paired_closeness_stats(
            tuned_to_orig=summary["depth_delta_l1"].to_numpy(),
            tuned_to_trunc=summary["depth_tuned_to_orig_trunc0_l1"].to_numpy(),
            projection_stats=depth_proj,
        ),
        "sigma": _paired_closeness_stats(
            tuned_to_orig=summary["sigma_delta_l1"].to_numpy(),
            tuned_to_trunc=summary["sigma_tuned_to_orig_trunc0_l1"].to_numpy(),
            projection_stats=sigma_proj,
        ),
    }
    if "perceptual" in results:
        stats_out["perceptual_identity"] = compute_perceptual_identity_statistics(results)
    return stats_out


def _default_results_dir(cfg: GeometryCompareConfig) -> Path:
    if cfg.results_dir is not None:
        return Path(cfg.results_dir)
    tuned_pkl = Path(cfg.tuned_pkl)
    return tuned_pkl.parent / "geometry_compare_results"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def run_geometry_comparison(cfg: GeometryCompareConfig) -> Dict[str, Any]:
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this Python process. "
            "This notebook expects a GPU-visible kernel for EG3D checkpoint loading and geometry sampling."
        )
    device = torch.device(cfg.device)

    orig_da = _GEN_UTILS.load_generator(cfg.orig_pkl, truncation_psi=cfg.truncation_psi, truncation_cutoff=cfg.truncation_cutoff)
    tuned_da = _GEN_UTILS.load_generator(cfg.tuned_pkl, truncation_psi=cfg.truncation_psi, truncation_cutoff=cfg.truncation_cutoff)

    orig_cam = _GEN_UTILS.get_single_dmap_cam(orig_da)
    tuned_cam = _GEN_UTILS.get_single_dmap_cam(tuned_da)

    z_batch = _shared_latents(cfg.n_samples, cfg.latent_seed, orig_da.G.z_dim, device=device)
    z_zero = torch.zeros((1, orig_da.G.z_dim), device=device)

    depth_orig = _render_depth_batch(
        orig_da,
        z_batch,
        orig_cam.camera_params.to(device),
        depth_resolution=cfg.depth_resolution,
        truncation_psi=cfg.truncation_psi,
        truncation_cutoff=cfg.truncation_cutoff,
        noise_mode=cfg.noise_mode,
        batch_size=cfg.model_batch_size,
        desc="Depth/orig",
    )
    depth_tuned = _render_depth_batch(
        tuned_da,
        z_batch,
        tuned_cam.camera_params.to(device),
        depth_resolution=cfg.depth_resolution,
        truncation_psi=cfg.truncation_psi,
        truncation_cutoff=cfg.truncation_cutoff,
        noise_mode=cfg.noise_mode,
        batch_size=cfg.model_batch_size,
        desc="Depth/tuned",
    )

    depth_zeroz_orig = _render_depth_batch(
        orig_da,
        z_zero,
        orig_cam.camera_params.to(device),
        depth_resolution=cfg.depth_resolution,
        truncation_psi=1.0,
        truncation_cutoff=cfg.truncation_cutoff,
        noise_mode=cfg.noise_mode,
        batch_size=1,
        desc="Depth/orig zero-z",
    )
    depth_zeroz_tuned = _render_depth_batch(
        tuned_da,
        z_zero,
        tuned_cam.camera_params.to(device),
        depth_resolution=cfg.depth_resolution,
        truncation_psi=1.0,
        truncation_cutoff=cfg.truncation_cutoff,
        noise_mode=cfg.noise_mode,
        batch_size=1,
        desc="Depth/tuned zero-z",
    )
    depth_trunc0_orig = _render_depth_batch(
        orig_da,
        z_zero,
        orig_cam.camera_params.to(device),
        depth_resolution=cfg.depth_resolution,
        truncation_psi=0.0,
        truncation_cutoff=cfg.truncation_cutoff,
        noise_mode=cfg.noise_mode,
        batch_size=1,
        desc="Depth/orig trunc0",
    )
    depth_trunc0_tuned = _render_depth_batch(
        tuned_da,
        z_zero,
        tuned_cam.camera_params.to(device),
        depth_resolution=cfg.depth_resolution,
        truncation_psi=0.0,
        truncation_cutoff=cfg.truncation_cutoff,
        noise_mode=cfg.noise_mode,
        batch_size=1,
        desc="Depth/tuned trunc0",
    )

    xyz_orig, mask_orig = _depths_to_xyz_maps(depth_orig.to(device), orig_cam.camera_params.to(device), cfg.pointcloud_radius_cutoff)
    xyz_tuned, mask_tuned = _depths_to_xyz_maps(depth_tuned.to(device), tuned_cam.camera_params.to(device), cfg.pointcloud_radius_cutoff)
    xyz_zeroz_orig, mask_zeroz_orig = _depths_to_xyz_maps(depth_zeroz_orig.to(device), orig_cam.camera_params.to(device), cfg.pointcloud_radius_cutoff)
    xyz_zeroz_tuned, mask_zeroz_tuned = _depths_to_xyz_maps(depth_zeroz_tuned.to(device), tuned_cam.camera_params.to(device), cfg.pointcloud_radius_cutoff)
    xyz_trunc0_orig, mask_trunc0_orig = _depths_to_xyz_maps(depth_trunc0_orig.to(device), orig_cam.camera_params.to(device), cfg.pointcloud_radius_cutoff)
    xyz_trunc0_tuned, mask_trunc0_tuned = _depths_to_xyz_maps(depth_trunc0_tuned.to(device), tuned_cam.camera_params.to(device), cfg.pointcloud_radius_cutoff)

    sigma_results = _run_sigma_analysis_streaming(
        cfg=cfg,
        device=device,
        orig_da=orig_da,
        tuned_da=tuned_da,
        z_batch=z_batch,
        z_zero=z_zero,
        orig_conditioning_params=orig_cam.conditioning_params.to(device),
        tuned_conditioning_params=tuned_cam.conditioning_params.to(device),
    )

    point_pair_metrics = _masked_point_metrics(xyz_orig, xyz_tuned, mask_orig, mask_tuned)
    perceptual = _run_perceptual_identity_analysis(
        cfg=cfg,
        device=device,
        orig_da=orig_da,
        tuned_da=tuned_da,
        z_batch=z_batch,
    )

    results = {
        "config": asdict(cfg),
        "results_dir": str(_default_results_dir(cfg)),
        "latents": z_batch.detach().cpu(),
        "depth": {
            "orig": depth_orig,
            "tuned": depth_tuned,
            "zeroz_orig": depth_zeroz_orig,
            "zeroz_tuned": depth_zeroz_tuned,
            "trunc0_orig": depth_trunc0_orig,
            "trunc0_tuned": depth_trunc0_tuned,
        },
        "pointcloud": {
            "xyz_orig": xyz_orig.detach().cpu(),
            "xyz_tuned": xyz_tuned.detach().cpu(),
            "xyz_zeroz_orig": xyz_zeroz_orig.detach().cpu(),
            "xyz_zeroz_tuned": xyz_zeroz_tuned.detach().cpu(),
            "xyz_trunc0_orig": xyz_trunc0_orig.detach().cpu(),
            "xyz_trunc0_tuned": xyz_trunc0_tuned.detach().cpu(),
            "mask_orig": mask_orig.detach().cpu(),
            "mask_tuned": mask_tuned.detach().cpu(),
            "mask_zeroz_orig": mask_zeroz_orig.detach().cpu(),
            "mask_zeroz_tuned": mask_zeroz_tuned.detach().cpu(),
            "mask_trunc0_orig": mask_trunc0_orig.detach().cpu(),
            "mask_trunc0_tuned": mask_trunc0_tuned.detach().cpu(),
            "pair_metrics": {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in point_pair_metrics.items()},
        },
        "sigma": sigma_results,
        "perceptual": perceptual,
    }
    results["summary"] = _build_summary(results)
    results["stats"] = compute_geometry_statistics(results)
    return results


def plot_depth_aggregates(results: Dict[str, Any], baseline_kind: str = "trunc0", cmap: str = "viridis") -> None:
    depth = results["depth"]
    orig_mean = depth["orig"].mean(0)
    tuned_mean = depth["tuned"].mean(0)
    signed_delta = (depth["tuned"] - depth["orig"]).mean(0)
    abs_delta = (depth["tuned"] - depth["orig"]).abs().mean(0)
    orig_base = depth[f"{baseline_kind}_orig"][0]
    tuned_base = depth[f"{baseline_kind}_tuned"][0]
    orig_base_dist = (depth["orig"] - orig_base.unsqueeze(0)).abs().mean(0)
    tuned_base_dist = (depth["tuned"] - tuned_base.unsqueeze(0)).abs().mean(0)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    items = [
        (orig_mean, "Orig mean depth", cmap),
        (tuned_mean, "Tuned mean depth", cmap),
        (signed_delta, "Mean signed depth delta", "coolwarm"),
        (abs_delta, "Mean abs depth delta", "magma"),
        (orig_base, f"Orig {baseline_kind} depth", cmap),
        (tuned_base, f"Tuned {baseline_kind} depth", cmap),
        (orig_base_dist, f"Orig to {baseline_kind} abs dist", "magma"),
        (tuned_base_dist, f"Tuned to {baseline_kind} abs dist", "magma"),
    ]
    for ax, (img, title, cm) in zip(axes.ravel(), items):
        im = ax.imshow(_to_numpy_image(img), cmap=cm)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()


def plot_depth_sample(results: Dict[str, Any], sample_idx: int, baseline_kind: str = "trunc0") -> None:
    depth = results["depth"]
    orig = depth["orig"][sample_idx]
    tuned = depth["tuned"][sample_idx]
    delta = tuned - orig
    orig_base = depth[f"{baseline_kind}_orig"][0]
    tuned_base = depth[f"{baseline_kind}_tuned"][0]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    items = [
        (orig, f"Orig depth #{sample_idx}", "viridis"),
        (tuned, f"Tuned depth #{sample_idx}", "viridis"),
        (delta, f"Signed delta #{sample_idx}", "coolwarm"),
        ((orig - orig_base).abs(), f"Orig abs dist to {baseline_kind}", "magma"),
        ((tuned - tuned_base).abs(), f"Tuned abs dist to {baseline_kind}", "magma"),
        (delta.abs(), f"Abs delta #{sample_idx}", "magma"),
    ]
    for ax, (img, title, cm) in zip(axes.ravel(), items):
        im = ax.imshow(_to_numpy_image(img), cmap=cm)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()


def plot_pointcloud_aggregates(results: Dict[str, Any], baseline_kind: str = "trunc0") -> None:
    pointcloud = results["pointcloud"]
    common = pointcloud["pair_metrics"]["common_mask"]
    disp = pointcloud["pair_metrics"]["disp"]
    mean_disp = _masked_mean(disp, common)
    _ = mean_disp  # used in summary, keep notebook helper quiet

    mean_disp_map = (disp * common).sum(0) / common.sum(0).clamp_min(1)
    mean_z_shift = ((pointcloud["xyz_tuned"][..., 2] - pointcloud["xyz_orig"][..., 2]) * common).sum(0) / common.sum(0).clamp_min(1)

    orig_base_common = pointcloud["mask_orig"] & _baseline_stack(pointcloud[f"mask_{baseline_kind}_orig"], len(pointcloud["mask_orig"]))
    tuned_base_common = pointcloud["mask_tuned"] & _baseline_stack(pointcloud[f"mask_{baseline_kind}_tuned"], len(pointcloud["mask_tuned"]))
    orig_base_disp = torch.linalg.norm(
        pointcloud["xyz_orig"] - _baseline_stack(pointcloud[f"xyz_{baseline_kind}_orig"], len(pointcloud["xyz_orig"])),
        dim=-1,
    )
    tuned_base_disp = torch.linalg.norm(
        pointcloud["xyz_tuned"] - _baseline_stack(pointcloud[f"xyz_{baseline_kind}_tuned"], len(pointcloud["xyz_tuned"])),
        dim=-1,
    )
    orig_base_map = (orig_base_disp * orig_base_common).sum(0) / orig_base_common.sum(0).clamp_min(1)
    tuned_base_map = (tuned_base_disp * tuned_base_common).sum(0) / tuned_base_common.sum(0).clamp_min(1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    items = [
        (mean_disp_map, "Mean point displacement magnitude", "magma"),
        (mean_z_shift, "Mean point signed z-shift", "coolwarm"),
        (orig_base_map, f"Orig point dist to {baseline_kind}", "magma"),
        (tuned_base_map, f"Tuned point dist to {baseline_kind}", "magma"),
    ]
    for ax, (img, title, cm) in zip(axes.ravel(), items):
        im = ax.imshow(_to_numpy_image(img), cmap=cm)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()


def _subsample_common_points(
    xyz_a: torch.Tensor,
    xyz_b: torch.Tensor,
    common_mask: torch.Tensor,
    max_points: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pts_a = xyz_a[common_mask]
    pts_b = xyz_b[common_mask]
    if pts_a.shape[0] == 0:
        raise ValueError("No common points found for this sample")
    if pts_a.shape[0] > max_points:
        idx = torch.linspace(0, pts_a.shape[0] - 1, max_points).long()
        pts_a = pts_a[idx]
        pts_b = pts_b[idx]
    disp = torch.linalg.norm(pts_b - pts_a, dim=-1)
    return pts_a, pts_b, disp


def plot_pointcloud_sample(results: Dict[str, Any], sample_idx: int, max_points: Optional[int] = None) -> None:
    pointcloud = results["pointcloud"]
    max_points = max_points or results["config"]["pointcloud_plot_points"]
    common_mask = pointcloud["pair_metrics"]["common_mask"][sample_idx]
    pts_orig, pts_tuned, disp = _subsample_common_points(
        pointcloud["xyz_orig"][sample_idx],
        pointcloud["xyz_tuned"][sample_idx],
        common_mask,
        max_points=max_points,
    )

    fig = plt.figure(figsize=(15, 5))
    axes = [
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3, projection="3d"),
    ]

    axes[0].scatter(pts_orig[:, 0], pts_orig[:, 1], pts_orig[:, 2], s=2, c="tab:blue", alpha=0.6)
    axes[0].set_title(f"Orig point cloud #{sample_idx}")
    axes[1].scatter(pts_tuned[:, 0], pts_tuned[:, 1], pts_tuned[:, 2], s=2, c="tab:red", alpha=0.6)
    axes[1].set_title(f"Tuned point cloud #{sample_idx}")
    sc = axes[2].scatter(pts_orig[:, 0], pts_orig[:, 1], pts_orig[:, 2], s=2, c=disp, cmap="magma", alpha=0.8)
    axes[2].set_title(f"Point displacement #{sample_idx}")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=15, azim=45)
    fig.colorbar(sc, ax=axes[2], fraction=0.03, pad=0.02)
    plt.tight_layout()


def _sigma_mid_slices(volumes: torch.Tensor) -> Dict[str, torch.Tensor]:
    mid = volumes.shape[-1] // 2
    return {
        "xy": volumes[:, :, :, mid],
        "xz": volumes[:, :, mid, :],
        "yz": volumes[:, mid, :, :],
    }


def plot_sigma_aggregates(results: Dict[str, Any], baseline_kind: str = "trunc0", plane: str = "xz") -> None:
    sigma = results["sigma"]
    plane_orig = _get_sigma_plane_tensor(sigma, "orig", plane)
    plane_tuned = _get_sigma_plane_tensor(sigma, "tuned", plane)
    plane_orig_base = _get_sigma_plane_tensor(sigma, f"{baseline_kind}_orig", plane)[0]
    plane_tuned_base = _get_sigma_plane_tensor(sigma, f"{baseline_kind}_tuned", plane)[0]
    signed_delta = (plane_tuned - plane_orig).mean(0)
    abs_delta = (plane_tuned - plane_orig).abs().mean(0)
    orig_base_dist = (plane_orig - plane_orig_base.unsqueeze(0)).abs().mean(0)
    tuned_base_dist = (plane_tuned - plane_tuned_base.unsqueeze(0)).abs().mean(0)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    items = [
        (plane_orig.mean(0), f"Orig mean sigma {plane}", "viridis"),
        (plane_tuned.mean(0), f"Tuned mean sigma {plane}", "viridis"),
        (signed_delta, f"Mean signed sigma delta {plane}", "coolwarm"),
        (abs_delta, f"Mean abs sigma delta {plane}", "magma"),
        (plane_orig_base, f"Orig {baseline_kind} sigma {plane}", "viridis"),
        (plane_tuned_base, f"Tuned {baseline_kind} sigma {plane}", "viridis"),
        (orig_base_dist, f"Orig to {baseline_kind} abs dist", "magma"),
        (tuned_base_dist, f"Tuned to {baseline_kind} abs dist", "magma"),
    ]
    for ax, (img, title, cm) in zip(axes.ravel(), items):
        im = ax.imshow(_to_numpy_image(img), cmap=cm)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()


def plot_sigma_sample(results: Dict[str, Any], sample_idx: int, plane: str = "xz", baseline_kind: str = "trunc0") -> None:
    sigma = results["sigma"]
    plane_orig = _get_sigma_plane_tensor(sigma, "orig", plane)[sample_idx]
    plane_tuned = _get_sigma_plane_tensor(sigma, "tuned", plane)[sample_idx]
    plane_delta = plane_tuned - plane_orig
    plane_orig_base = _get_sigma_plane_tensor(sigma, f"{baseline_kind}_orig", plane)[0]
    plane_tuned_base = _get_sigma_plane_tensor(sigma, f"{baseline_kind}_tuned", plane)[0]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    items = [
        (plane_orig, f"Orig sigma {plane} #{sample_idx}", "viridis"),
        (plane_tuned, f"Tuned sigma {plane} #{sample_idx}", "viridis"),
        (plane_delta, f"Signed sigma delta {plane}", "coolwarm"),
        ((plane_orig - plane_orig_base).abs(), f"Orig abs dist to {baseline_kind}", "magma"),
        ((plane_tuned - plane_tuned_base).abs(), f"Tuned abs dist to {baseline_kind}", "magma"),
        (plane_delta.abs(), f"Abs sigma delta {plane}", "magma"),
    ]
    for ax, (img, title, cm) in zip(axes.ravel(), items):
        im = ax.imshow(_to_numpy_image(img), cmap=cm)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()


def plot_pull_to_mean_histograms(results: Dict[str, Any]) -> None:
    summary = results["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    items = [
        ("depth_trunc0_pull_ratio", "Depth pull ratio"),
        ("point_trunc0_pull_ratio", "Point-cloud pull ratio"),
        ("sigma_trunc0_pull_ratio", "Sigma pull ratio"),
    ]
    for ax, (col, title) in zip(axes, items):
        vals = summary[col].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(vals, bins=20)
        ax.axvline(1.0, color="red", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("tuned distance / orig distance")
        ax.set_ylabel("count")
    plt.tight_layout()


def plot_perceptual_identity_aggregates(results: Dict[str, Any]) -> None:
    summary = results["summary"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].hist(summary["lpips_cross_model_canonical"].dropna(), bins=20)
    axes[0, 0].set_title("LPIPS tuned vs orig canonical")
    axes[0, 0].set_xlabel("LPIPS")

    axes[0, 1].hist(summary["lpips_cross_model_views_mean"].dropna(), bins=20)
    axes[0, 1].set_title("LPIPS tuned vs orig train-view mean")
    axes[0, 1].set_xlabel("LPIPS")

    axes[0, 2].hist(summary["id_cross_model_canonical_cos"].dropna(), bins=20)
    axes[0, 2].set_title("Identity cosine tuned vs orig canonical")
    axes[0, 2].set_xlabel("cosine")

    axes[1, 0].hist(summary["id_orig_view_to_canonical_cos_mean"].dropna(), bins=20, alpha=0.6, label="orig")
    axes[1, 0].hist(summary["id_tuned_view_to_canonical_cos_mean"].dropna(), bins=20, alpha=0.6, label="tuned")
    axes[1, 0].set_title("Identity consistency across views")
    axes[1, 0].set_xlabel("mean cosine to canonical")
    axes[1, 0].legend()

    axes[1, 1].scatter(summary["point_max_abs_delta_z"], summary["id_cross_model_canonical_drift"], s=16, alpha=0.7)
    axes[1, 1].set_title("Identity drift vs max abs delta z")
    axes[1, 1].set_xlabel("point max abs delta z")
    axes[1, 1].set_ylabel("1 - canonical cosine")

    axes[1, 2].scatter(summary["depth_delta_l1"], summary["id_cross_model_canonical_drift"], s=16, alpha=0.7)
    axes[1, 2].set_title("Identity drift vs depth delta L1")
    axes[1, 2].set_xlabel("depth delta L1")
    axes[1, 2].set_ylabel("1 - canonical cosine")

    plt.tight_layout()


def _save_identity_view_grid(results: Dict[str, Any], sample_idx: int, out_path: Optional[Path] = None) -> None:
    rgb = results["perceptual"]["rgb"]
    summary_row = results["summary"].loc[sample_idx]
    orig_views = rgb["orig_views"][sample_idx]
    tuned_views = rgb["tuned_views"][sample_idx]
    n_views = orig_views.shape[0]
    fig, axes = plt.subplots(n_views + 1, 3, figsize=(12, 3 * (n_views + 1)))
    row_labels = ["canonical"] + [f"train view {i:02d}" for i in range(n_views)]
    panels = [
        (rgb["orig_canonical"][sample_idx], rgb["tuned_canonical"][sample_idx]),
        *[(orig_views[i], tuned_views[i]) for i in range(n_views)],
    ]

    for row_idx, ((orig_im, tuned_im), row_label) in enumerate(zip(panels, row_labels)):
        axes[row_idx, 0].imshow(_prepare_rgb_for_display(orig_im))
        axes[row_idx, 0].set_ylabel(row_label)
        axes[row_idx, 1].imshow(_prepare_rgb_for_display(tuned_im))
        axes[row_idx, 2].imshow(_prepare_rgb_diff_for_display(orig_im, tuned_im))
        for col_idx, title in enumerate(["orig", "tuned", "abs rgb diff"]):
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(title)

    fig.suptitle(
        (
            f"sample {sample_idx} | id cos canon={summary_row['id_cross_model_canonical_cos']:.4f} "
            f"| lpips canon={summary_row['lpips_cross_model_canonical']:.4f}\n"
            f"depth L1={summary_row['depth_delta_l1']:.4f} | sigma L1={summary_row['sigma_delta_l1']:.4f} "
            f"| max abs delta z={summary_row['point_max_abs_delta_z']:.4f}"
        ),
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _save_geometry_context_grid(results: Dict[str, Any], sample_idx: int, out_path: Optional[Path] = None) -> None:
    depth = results["depth"]
    sigma = results["sigma"]
    sigma_plane_orig = _get_sigma_plane_tensor(sigma, "orig", "xz")[sample_idx]
    sigma_plane_tuned = _get_sigma_plane_tensor(sigma, "tuned", "xz")[sample_idx]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    items = [
        (depth["orig"][sample_idx], "Orig depth", "viridis"),
        (depth["tuned"][sample_idx], "Tuned depth", "viridis"),
        (depth["tuned"][sample_idx] - depth["orig"][sample_idx], "Depth signed delta", "coolwarm"),
        (sigma_plane_orig, "Orig sigma xz", "viridis"),
        (sigma_plane_tuned, "Tuned sigma xz", "viridis"),
        (sigma_plane_tuned - sigma_plane_orig, "Sigma signed delta xz", "coolwarm"),
    ]
    for ax, (img, title, cmap) in zip(axes.ravel(), items):
        im = ax.imshow(_to_numpy_image(img), cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_identity_case(results: Dict[str, Any], sample_idx: int) -> None:
    _save_identity_view_grid(results, sample_idx=sample_idx, out_path=None)
    _save_geometry_context_grid(results, sample_idx=sample_idx, out_path=None)


def save_geometry_artifacts(results: Dict[str, Any], out_dir: Optional[str] = None) -> Path:
    target = Path(out_dir) if out_dir is not None else Path(results["results_dir"])
    target.mkdir(parents=True, exist_ok=True)

    summary = results["summary"]
    summary.to_csv(target / "geometry_summary.csv", index=False)

    stats_payload = {
        "config": results["config"],
        "stats": results["stats"],
    }
    with open(target / "geometry_stats.json", "w") as f:
        json.dump(_jsonable(stats_payload), f, indent=2)

    if "perceptual" in results:
        perceptual = results["perceptual"]
        view_meta = {
            "dataset_path": perceptual["dataset_path"],
            "rgb_key": perceptual["rgb_key"],
            "rgb_metric_resolution": perceptual["rgb_metric_resolution"],
            "viewpoint_seed": perceptual["viewpoint_seed"],
            "sampled_view_indices": perceptual["sampled_view_indices"],
        }
        with open(target / "perceptual_view_metadata.json", "w") as f:
            json.dump(_jsonable(view_meta), f, indent=2)
        np.save(target / "empirical_view_camera_params.npy", perceptual["camera_params"]["empirical"].numpy())

    return target


def _signed_colors_from_values(values: np.ndarray, cmap_name: str = "coolwarm") -> np.ndarray:
    import matplotlib.cm as cm

    max_abs = np.max(np.abs(values))
    if max_abs <= 0 or not np.isfinite(max_abs):
        max_abs = 1.0
    normed = (values / max_abs + 1.0) / 2.0
    rgba = cm.get_cmap(cmap_name)(normed)
    return (rgba[:, :4] * 255).astype(np.uint8)


def _export_plotly_pointcloud_html(
    pts_orig: np.ndarray,
    pts_tuned: np.ndarray,
    z_delta: np.ndarray,
    out_html: Path,
    title: str,
) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return False

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Original", "Tuned", "Delta z on original points"),
    )

    fig.add_trace(
        go.Scatter3d(
            x=pts_orig[:, 0],
            y=pts_orig[:, 1],
            z=pts_orig[:, 2],
            mode="markers",
            marker=dict(size=1.8, color="royalblue", opacity=0.7),
            name="orig",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=pts_tuned[:, 0],
            y=pts_tuned[:, 1],
            z=pts_tuned[:, 2],
            mode="markers",
            marker=dict(size=1.8, color="firebrick", opacity=0.7),
            name="tuned",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=pts_orig[:, 0],
            y=pts_orig[:, 1],
            z=pts_orig[:, 2],
            mode="markers",
            marker=dict(
                size=1.8,
                color=z_delta,
                colorscale="RdBu",
                cmin=-float(np.max(np.abs(z_delta))),
                cmax=float(np.max(np.abs(z_delta))),
                opacity=0.85,
                colorbar=dict(title="delta z"),
            ),
            name="delta_z",
        ),
        row=1,
        col=3,
    )
    fig.update_layout(height=550, width=1700, title=title)
    fig.write_html(str(out_html))
    return True


def export_top_delta_pointclouds(
    results: Dict[str, Any],
    out_dir: Optional[str] = None,
    top_k: int = 10,
    rank_by: str = "point_max_abs_delta_z",
    max_points: Optional[int] = None,
) -> pd.DataFrame:
    import trimesh

    target = save_geometry_artifacts(results, out_dir=out_dir)
    export_dir = target / "top_pointcloud_delta_exports"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    summary = results["summary"].sort_values(rank_by, ascending=False).head(top_k).copy()
    pointcloud = results["pointcloud"]
    max_points = max_points or results["config"]["pointcloud_plot_points"]

    rows = []
    for rank, sample_idx in enumerate(summary["sample_idx"].astype(int).tolist(), start=1):
        common_mask = pointcloud["pair_metrics"]["common_mask"][sample_idx]
        pts_orig, pts_tuned, _ = _subsample_common_points(
            pointcloud["xyz_orig"][sample_idx],
            pointcloud["xyz_tuned"][sample_idx],
            common_mask,
            max_points=max_points,
        )
        z_delta = (pts_tuned[:, 2] - pts_orig[:, 2]).cpu().numpy()
        colors = _signed_colors_from_values(z_delta)

        sample_dir = export_dir / f"rank_{rank:02d}_sample_{sample_idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        orig_np = pts_orig.cpu().numpy()
        tuned_np = pts_tuned.cpu().numpy()

        orig_pc = trimesh.points.PointCloud(orig_np, colors=np.tile(np.array([[160, 160, 160, 255]], dtype=np.uint8), (orig_np.shape[0], 1)))
        tuned_pc = trimesh.points.PointCloud(tuned_np, colors=np.tile(np.array([[160, 160, 160, 255]], dtype=np.uint8), (tuned_np.shape[0], 1)))
        delta_orig_pc = trimesh.points.PointCloud(orig_np, colors=colors)
        delta_tuned_pc = trimesh.points.PointCloud(tuned_np, colors=colors)

        orig_ply = sample_dir / "orig_points_gray.ply"
        tuned_ply = sample_dir / "tuned_points_gray.ply"
        delta_orig_ply = sample_dir / "delta_z_on_orig_positions.ply"
        delta_tuned_ply = sample_dir / "delta_z_on_tuned_positions.ply"
        orig_pc.export(orig_ply)
        tuned_pc.export(tuned_ply)
        delta_orig_pc.export(delta_orig_ply)
        delta_tuned_pc.export(delta_tuned_ply)

        html_path = sample_dir / "pointcloud_delta_view.html"
        html_written = _export_plotly_pointcloud_html(
            orig_np,
            tuned_np,
            z_delta,
            out_html=html_path,
            title=f"Sample {sample_idx} rank {rank} by {rank_by}",
        )

        metadata = {
            "rank": rank,
            "sample_idx": sample_idx,
            "rank_by": rank_by,
            "max_abs_delta_z": float(np.max(np.abs(z_delta))),
            "mean_abs_delta_z": float(np.mean(np.abs(z_delta))),
            "orig_points_gray_ply": str(orig_ply),
            "tuned_points_gray_ply": str(tuned_ply),
            "delta_z_on_orig_positions_ply": str(delta_orig_ply),
            "delta_z_on_tuned_positions_ply": str(delta_tuned_ply),
            "plotly_html": str(html_path) if html_written else None,
        }
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        rows.append(metadata)

    export_df = pd.DataFrame(rows)
    export_df.to_csv(export_dir / "top_pointcloud_exports.csv", index=False)
    return export_df


def export_top_identity_drift_cases(
    results: Dict[str, Any],
    out_dir: Optional[str] = None,
    top_k: int = 10,
    rank_by: str = "id_cross_model_canonical_cos",
) -> pd.DataFrame:
    target = save_geometry_artifacts(results, out_dir=out_dir)
    export_dir = target / "top_identity_drift_exports"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    summary = results["summary"].sort_values(rank_by, ascending=True).head(top_k).copy()

    rows = []
    for rank, sample_idx in enumerate(summary["sample_idx"].astype(int).tolist(), start=1):
        sample_dir = export_dir / f"rank_{rank:02d}_sample_{sample_idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        rgb_grid = sample_dir / "rgb_identity_views.png"
        geom_grid = sample_dir / "geometry_context.png"
        _save_identity_view_grid(results, sample_idx=sample_idx, out_path=rgb_grid)
        _save_geometry_context_grid(results, sample_idx=sample_idx, out_path=geom_grid)

        summary_row = results["summary"].loc[sample_idx].to_dict()
        metadata = {
            "rank": rank,
            "sample_idx": sample_idx,
            "rank_by": rank_by,
            "rgb_identity_views_png": str(rgb_grid),
            "geometry_context_png": str(geom_grid),
            **summary_row,
        }
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(_jsonable(metadata), f, indent=2)
        rows.append(metadata)

    export_df = pd.DataFrame(rows)
    export_df.to_csv(export_dir / "top_identity_drift_cases.csv", index=False)
    return export_df
