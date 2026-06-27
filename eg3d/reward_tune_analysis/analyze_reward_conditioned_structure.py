#!/usr/bin/env python3
"""Analyze reward-conditioned organization in EG3D generator activations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import autoroot  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import legacy
from analyze_snapshot_svd import force_torch_load_map_location_cpu, snapshot_kimg, snapshot_label
from core_modules.utils import camera_utils as rm_camera_utils
from core_modules.utils.reward_loading import load_rwd_model_from_cfg
from training.volumetric_rendering.renderer import generate_planes, sample_from_planes


@dataclass
class ModelResult:
    label: str
    scores: torch.Tensor
    sample_rows: list[dict[str, object]]
    representations: dict[str, torch.Tensor]


def parse_seed_list(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Untuned/base EG3D snapshot.")
    parser.add_argument("--target", required=True, help="Tuned EG3D snapshot.")
    parser.add_argument("--reward-model-id", default="7wnzkgie")
    parser.add_argument("--pads-config", type=Path, default=Path("../reward_model_training/static_configs/pads_vals_entire.yaml"))
    parser.add_argument("--seeds", default="0-47", help="Comma/range seed list.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--top-frac", type=float, default=0.25)
    parser.add_argument("--bottom-frac", type=float, default=0.25)
    parser.add_argument("--reward-truncation-psi", type=float, default=1.0)
    parser.add_argument("--reward-truncation-cutoff", type=int, default=14)
    parser.add_argument("--fixed-grid-res", type=int, default=12)
    parser.add_argument("--plane-pool-res", type=int, default=8)
    parser.add_argument("--geom-downsample-res", type=int, default=16)
    parser.add_argument("--max-batch", type=int, default=500000)
    return parser.parse_args()


def register_omegaconf_resolvers() -> None:
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
    OmegaConf.register_new_resolver("multiply_to_int", lambda x, y: int(x * y), replace=True)
    OmegaConf.register_new_resolver("divide_ceil", lambda x, y: int(math.ceil(x / y)), replace=True)
    OmegaConf.register_new_resolver("divide_floor", lambda x, y: int(math.floor(x / y)), replace=True)


def set_reward_model_env_defaults() -> None:
    project_root = Path(__file__).resolve().parents[1]
    framework_root = project_root / "reward_model_training" / "reward_model_framework"
    os.environ.setdefault("PROJECT_ROOT", str(project_root))
    os.environ.setdefault("STATIC_CONFIGS_DIR", str(project_root / "reward_model_training" / "static_configs"))
    os.environ.setdefault("RWD_MODELS_DIR", str(framework_root / "core_modules" / "RWD_MODELS_FOR_TUNING"))
    os.environ.setdefault("RUNS_SUMMARY_CSV", str(framework_root / "core_modules" / "notebooks" / "runs_summary_for_tune.csv"))


def load_generator(snapshot_path: Path, device: torch.device):
    with snapshot_path.open("rb") as handle:
        with force_torch_load_map_location_cpu():
            data = legacy.load_network_pkl(handle)
    return data["G_ema"].eval().requires_grad_(False).to(device)


def create_samples(N: int, cube_length: float) -> torch.Tensor:
    voxel_origin = np.array([0, 0, 0]) - cube_length / 2
    voxel_size = cube_length / (N - 1)
    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 3)
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    return samples.unsqueeze(0)


def cpad(pc: float, shape_res: int) -> int:
    cp = int(pc * shape_res)
    if cp == 0:
        return 1
    return cp


def make_reward_coords(pads_cfg, device: torch.device) -> tuple[torch.Tensor, tuple[int, int, int]]:
    shape_res = int(pads_cfg.shape_res)
    base = create_samples(shape_res, cube_length=1.0).reshape(1, shape_res, shape_res, shape_res, 3).cpu().numpy()
    base = np.flip(base, 0)
    rhs = cpad(float(pads_cfg.rhs_pad), shape_res)
    lhs = cpad(float(pads_cfg.lhs_pad), shape_res)
    bot = cpad(float(pads_cfg.bot_pad), shape_res)
    top = cpad(float(pads_cfg.top_pad), shape_res)
    rear = cpad(float(pads_cfg.rear_pad), shape_res)
    front = cpad(float(pads_cfg.front_pad), shape_res)
    cropped = base[:, rhs : shape_res - lhs, bot : shape_res - top, rear : shape_res - front, :]
    shape = cropped.shape[1:4]
    coords = torch.from_numpy(cropped.reshape(1, -1, 3)).to(device)
    return coords, shape


def sample_sigma_subset(
    G,
    z: torch.Tensor,
    conditioning_params: torch.Tensor,
    coords: torch.Tensor,
    truncation_psi: float,
    truncation_cutoff: int,
    max_batch: int,
) -> torch.Tensor:
    sigmas = torch.zeros((coords.shape[0], coords.shape[1], 1), device=z.device)
    dirs = torch.zeros((coords.shape[0], max_batch, 3), device=z.device)
    dirs[..., -1] = -1
    head = 0
    while head < coords.shape[1]:
        chunk = min(max_batch, coords.shape[1] - head)
        sigma = G.sample(
            coords[:, head : head + chunk],
            dirs[:, :chunk],
            z,
            conditioning_params,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            noise_mode="const",
        )["sigma"]
        sigmas[:, head : head + chunk] = sigma
        head += chunk
    return sigmas


def normalize_sigma_self(x: torch.Tensor, out_min: float = 0.0, out_max: float = 100.0) -> torch.Tensor:
    x_min = x.min()
    x_max = x.max()
    if float((x_max - x_min).item()) <= 1e-8:
        return torch.full_like(x, out_min)
    return (x - x_min) / (x_max - x_min) * (out_max - out_min) + out_min


def make_fixed_coord_grid(box_warp: float, res: int, device: torch.device) -> torch.Tensor:
    lo = -box_warp / 2.0
    hi = box_warp / 2.0
    coords = torch.linspace(lo, hi, res, device=device)
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1).reshape(1, -1, 3)


def get_generator_representations(
    G,
    z: torch.Tensor,
    c: torch.Tensor,
    coords: torch.Tensor,
    plane_pool_res: int,
    truncation_psi: float,
    truncation_cutoff: int,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        planes_flat = G.backbone.synthesis(ws, noise_mode="const")
        planes = planes_flat.view(len(planes_flat), 3, 32, planes_flat.shape[-2], planes_flat.shape[-1])
        plane_pool = F.adaptive_avg_pool2d(planes.reshape(-1, planes.shape[-2], planes.shape[-1]), output_size=(plane_pool_res, plane_pool_res))
        plane_axes = generate_planes().to(z.device)
        sampled = sample_from_planes(plane_axes, planes, coords, padding_mode="zeros", box_warp=G.rendering_kwargs["box_warp"])
        decoder_input = sampled.mean(1).squeeze(0)
        decoder_hidden = G.decoder.net[1](G.decoder.net[0](decoder_input))
        decoder_final = G.decoder.net[2](decoder_hidden)
    return {
        "planes_pooled": plane_pool.flatten().detach().cpu(),
        "decoder_input": decoder_input.flatten().detach().cpu(),
        "decoder_hidden": decoder_hidden.flatten().detach().cpu(),
        "sigma_output": decoder_final[:, :1].flatten().detach().cpu(),
        "rgb_output_raw": decoder_final[:, 1:].flatten().detach().cpu(),
    }


def sigma_geometry_summary(sigmas_raw: torch.Tensor) -> dict[str, float]:
    density = F.softplus(sigmas_raw - 1.0)
    total_mass = float(density.sum().item())
    occ0 = float((sigmas_raw > 0).float().mean().item())
    occ1 = float((sigmas_raw > 1).float().mean().item())
    occ10 = float((sigmas_raw > 10).float().mean().item())
    center = sigmas_raw[
        sigmas_raw.shape[0] // 4 : (3 * sigmas_raw.shape[0]) // 4,
        sigmas_raw.shape[1] // 4 : (3 * sigmas_raw.shape[1]) // 4,
        sigmas_raw.shape[2] // 4 : (3 * sigmas_raw.shape[2]) // 4,
    ]
    center_density = F.softplus(center - 1.0)
    center_mass_ratio = float((center_density.sum() / density.sum().clamp_min(1e-8)).item())

    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, sigmas_raw.shape[0], device=sigmas_raw.device),
        torch.linspace(-1.0, 1.0, sigmas_raw.shape[1], device=sigmas_raw.device),
        torch.linspace(-1.0, 1.0, sigmas_raw.shape[2], device=sigmas_raw.device),
        indexing="ij",
    )
    denom = density.sum().clamp_min(1e-8)
    com_x = float((density * xx).sum().item() / denom.item())
    com_y = float((density * yy).sum().item() / denom.item())
    com_z = float((density * zz).sum().item() / denom.item())

    front_mass = float(density[:, :, density.shape[2] // 2 :].sum().item())
    rear_mass = float(density[:, :, : density.shape[2] // 2].sum().item())
    front_rear_ratio = front_mass / max(rear_mass, 1e-8)

    return {
        "sigma_min": float(sigmas_raw.min().item()),
        "sigma_max": float(sigmas_raw.max().item()),
        "density_mass": total_mass,
        "occ_gt0": occ0,
        "occ_gt1": occ1,
        "occ_gt10": occ10,
        "center_mass_ratio": center_mass_ratio,
        "density_com_x": com_x,
        "density_com_y": com_y,
        "density_com_z": com_z,
        "front_rear_mass_ratio": front_rear_ratio,
    }


def sample_features_for_model(
    G,
    label: str,
    seeds: list[int],
    reward_model,
    reward_coords: torch.Tensor,
    reward_shape: tuple[int, int, int],
    fixed_coords: torch.Tensor,
    canonical_c: torch.Tensor,
    args: argparse.Namespace,
) -> ModelResult:
    rows: list[dict[str, object]] = []
    reps: dict[str, list[torch.Tensor]] = {
        "planes_pooled": [],
        "decoder_input": [],
        "decoder_hidden": [],
        "sigma_output": [],
        "rgb_output_raw": [],
        "sigma_geometry_downsampled": [],
        "reward_embedding": [],
    }

    for seed in seeds:
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(args.device)
        c = canonical_c.to(args.device)

        with torch.no_grad():
            raw_sigmas = sample_sigma_subset(
                G,
                z,
                c,
                reward_coords,
                truncation_psi=args.reward_truncation_psi,
                truncation_cutoff=args.reward_truncation_cutoff,
                max_batch=args.max_batch,
            ).view(reward_shape)
            sigmas_aug = normalize_sigma_self(raw_sigmas).permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
            reward_score = float(reward_model.forward(sigmas_aug).reshape(-1)[0].item())
            reward_embedding = reward_model.forward_to_global_feature_vec(sigmas_aug).flatten().detach().cpu()
            sigma_geom = F.interpolate(
                sigmas_aug,
                size=(args.geom_downsample_res, args.geom_downsample_res, args.geom_downsample_res),
                mode="trilinear",
                align_corners=True,
            ).flatten().detach().cpu()

        generator_reps = get_generator_representations(
            G,
            z,
            c,
            fixed_coords,
            plane_pool_res=args.plane_pool_res,
            truncation_psi=args.reward_truncation_psi,
            truncation_cutoff=args.reward_truncation_cutoff,
        )
        geom_stats = sigma_geometry_summary(raw_sigmas)

        row = {
            "model_label": label,
            "seed": seed,
            "reward_score": reward_score,
        }
        row.update(geom_stats)
        rows.append(row)

        for key, vec in generator_reps.items():
            reps[key].append(vec)
        reps["sigma_geometry_downsampled"].append(sigma_geom)
        reps["reward_embedding"].append(reward_embedding)

    stacked = {key: torch.stack(values, dim=0) for key, values in reps.items()}
    scores = torch.tensor([float(row["reward_score"]) for row in rows], dtype=torch.float32)
    return ModelResult(label=label, scores=scores, sample_rows=rows, representations=stacked)


def effective_rank_from_centered_matrix(X: torch.Tensor) -> float:
    X = X.float()
    X = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(X)
    s = s[s > 1e-12]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * torch.log(p + 1e-12)).sum()
    return float(torch.exp(entropy).item())


def stable_rank_from_centered_matrix(X: torch.Tensor) -> float:
    X = X.float()
    X = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(X)
    if s.numel() == 0 or float(s[0].item()) == 0.0:
        return 0.0
    return float((s.square().sum() / s[0].square()).item())


def energy_rank_fraction(X: torch.Tensor, fraction: float) -> float:
    X = X.float()
    X = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(X)
    if s.numel() == 0:
        return math.nan
    e = s.square()
    total = float(e.sum().item())
    if total == 0:
        return math.nan
    cume = torch.cumsum(e, dim=0) / total
    idx = int(torch.searchsorted(cume, torch.tensor(fraction)).item()) + 1
    return float(idx / s.numel())


def mean_pairwise_distance(X: torch.Tensor) -> float:
    X = X.float()
    if X.shape[0] < 2:
        return 0.0
    d = torch.cdist(X, X)
    tri = torch.triu_indices(d.shape[0], d.shape[1], offset=1)
    vals = d[tri[0], tri[1]]
    return float(vals.mean().item())


def centroid_distance(Xa: torch.Tensor, Xb: torch.Tensor) -> float:
    ca = Xa.float().mean(0)
    cb = Xb.float().mean(0)
    return float(torch.linalg.norm(ca - cb).item())


def fisher_ratio(Xa: torch.Tensor, Xb: torch.Tensor) -> float:
    within = mean_pairwise_distance(Xa) + mean_pairwise_distance(Xb)
    if within <= 1e-12:
        return math.nan
    return centroid_distance(Xa, Xb) / (within / 2.0)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, args: argparse.Namespace, group_rows: list[dict[str, object]], contrast_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Reward-Conditioned Structure Analysis",
        "",
        f"- Seeds: `{args.seeds}`",
        f"- Reward model: `{args.reward_model_id}`",
        f"- Top fraction: `{args.top_frac}`",
        f"- Bottom fraction: `{args.bottom_frac}`",
        f"- Reward scoring: exact `sigma_field_256` front-slab with self-normalization to `[0, 100]`",
        "",
        "## Key Contrasts",
        "",
    ]
    interesting = []
    for rep_name in ["decoder_hidden", "sigma_output", "sigma_geometry_downsampled", "planes_pooled"]:
        rep_rows = [row for row in contrast_rows if row["representation"] == rep_name]
        if not rep_rows:
            continue
        interesting.append(f"### `{rep_name}`")
        interesting.append("")
        for row in rep_rows:
            interesting.append(
                f"- `{row['model_label']}`: high-vs-low centroid distance `{float(row['centroid_distance']):.4f}`, "
                f"Fisher ratio `{float(row['fisher_ratio']):.4f}`, "
                f"within-tightness ratio low/high `{float(row['within_ratio_low_over_high']):.4f}`."
            )
        interesting.append("")
    lines.extend(interesting)

    lines.extend(
        [
            "## Interpretation Guide",
            "",
            "- `within_ratio_low_over_high > 1` means the high-reward group is tighter than the low-reward group.",
            "- Lower `effective_rank` for the high-reward group suggests a more compact sample manifold.",
            "- Larger `fisher_ratio` means stronger separation between high- and low-reward groups.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    register_omegaconf_resolvers()
    set_reward_model_env_defaults()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    seeds = parse_seed_list(args.seeds)
    n_group = max(4, int(len(seeds) * min(args.top_frac, args.bottom_frac)))

    pads_cfg = OmegaConf.load(args.pads_config)
    reward_coords, reward_shape = make_reward_coords(pads_cfg, device)
    reward_model = load_rwd_model_from_cfg(args.reward_model_id, strict=True)

    base_path = Path(args.base).resolve()
    target_path = Path(args.target).resolve()
    G_base = load_generator(base_path, device)
    G_target = load_generator(target_path, device)

    canonical_c = rm_camera_utils.get_single_dmap_camera().to(device)
    fixed_coords = make_fixed_coord_grid(float(G_base.rendering_kwargs["box_warp"]), args.fixed_grid_res, device)

    base_label = f"untuned_{snapshot_label(base_path)}"
    target_label = f"tuned_{snapshot_label(target_path)}"

    results = [
        sample_features_for_model(G_base, base_label, seeds, reward_model, reward_coords, reward_shape, fixed_coords, canonical_c, args),
        sample_features_for_model(G_target, target_label, seeds, reward_model, reward_coords, reward_shape, fixed_coords, canonical_c, args),
    ]

    all_sample_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []

    for result in results:
        all_sample_rows.extend(result.sample_rows)
        order = torch.argsort(result.scores)
        low_idx = order[:n_group]
        high_idx = order[-n_group:]

        low_seeds = [int(result.sample_rows[int(i)]["seed"]) for i in low_idx]
        high_seeds = [int(result.sample_rows[int(i)]["seed"]) for i in high_idx]

        for rep_name, matrix in result.representations.items():
            low_X = matrix[low_idx].float()
            high_X = matrix[high_idx].float()

            group_rows.append(
                {
                    "model_label": result.label,
                    "representation": rep_name,
                    "group": "low_reward",
                    "n_samples": int(low_X.shape[0]),
                    "seed_list": ",".join(str(s) for s in low_seeds),
                    "reward_mean": float(result.scores[low_idx].mean().item()),
                    "reward_std": float(result.scores[low_idx].std(unbiased=False).item()),
                    "effective_rank": effective_rank_from_centered_matrix(low_X),
                    "stable_rank": stable_rank_from_centered_matrix(low_X),
                    "energy_rank_90_frac": energy_rank_fraction(low_X, 0.90),
                    "energy_rank_95_frac": energy_rank_fraction(low_X, 0.95),
                    "within_pairwise_distance": mean_pairwise_distance(low_X),
                }
            )
            group_rows.append(
                {
                    "model_label": result.label,
                    "representation": rep_name,
                    "group": "high_reward",
                    "n_samples": int(high_X.shape[0]),
                    "seed_list": ",".join(str(s) for s in high_seeds),
                    "reward_mean": float(result.scores[high_idx].mean().item()),
                    "reward_std": float(result.scores[high_idx].std(unbiased=False).item()),
                    "effective_rank": effective_rank_from_centered_matrix(high_X),
                    "stable_rank": stable_rank_from_centered_matrix(high_X),
                    "energy_rank_90_frac": energy_rank_fraction(high_X, 0.90),
                    "energy_rank_95_frac": energy_rank_fraction(high_X, 0.95),
                    "within_pairwise_distance": mean_pairwise_distance(high_X),
                }
            )

            low_within = mean_pairwise_distance(low_X)
            high_within = mean_pairwise_distance(high_X)
            contrast_rows.append(
                {
                    "model_label": result.label,
                    "representation": rep_name,
                    "n_per_group": int(n_group),
                    "low_reward_mean": float(result.scores[low_idx].mean().item()),
                    "high_reward_mean": float(result.scores[high_idx].mean().item()),
                    "centroid_distance": centroid_distance(low_X, high_X),
                    "fisher_ratio": fisher_ratio(low_X, high_X),
                    "within_low": low_within,
                    "within_high": high_within,
                    "within_ratio_low_over_high": low_within / max(high_within, 1e-8),
                    "effective_rank_low": effective_rank_from_centered_matrix(low_X),
                    "effective_rank_high": effective_rank_from_centered_matrix(high_X),
                    "effective_rank_ratio_low_over_high": effective_rank_from_centered_matrix(low_X) / max(effective_rank_from_centered_matrix(high_X), 1e-8),
                }
            )

    summary = {
        "base": {"path": str(base_path), "kimg": snapshot_kimg(base_path)},
        "target": {"path": str(target_path), "kimg": snapshot_kimg(target_path)},
        "reward_model_id": args.reward_model_id,
        "seeds": seeds,
        "n_group": n_group,
        "fixed_grid_res": args.fixed_grid_res,
        "geom_downsample_res": args.geom_downsample_res,
        "plane_pool_res": args.plane_pool_res,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(output_dir / "sample_scores_and_geometry.csv", all_sample_rows)
    write_csv(output_dir / "group_metrics.csv", group_rows)
    write_csv(output_dir / "contrast_metrics.csv", contrast_rows)
    write_report(output_dir / "reward_conditioned_structure_report.md", args, group_rows, contrast_rows)

    print(f"Wrote reward-conditioned analysis to: {output_dir}")


if __name__ == "__main__":
    main()
