#!/usr/bin/env python3
"""Localized sigma-region analysis for reward-conditioned samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import autoroot  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F

import legacy
from analyze_snapshot_svd import force_torch_load_map_location_cpu, snapshot_label
from compare_snapshot_sigma_fields import create_samples
from core_modules.utils import camera_utils as rm_camera_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--scores-csv", type=Path, required=True, help="sample_scores_and_geometry.csv from reward-conditioned run")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shape-res", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--truncation-psi", type=float, default=1.0)
    parser.add_argument("--truncation-cutoff", type=int, default=14)
    parser.add_argument("--max-batch", type=int, default=200000)
    return parser.parse_args()


def load_generator(path: Path, device: torch.device):
    with path.open("rb") as handle:
        with force_torch_load_map_location_cpu():
            data = legacy.load_network_pkl(handle)
    return data["G_ema"].eval().requires_grad_(False).to(device)


def load_reward_groups(csv_path: Path, top_k: int) -> dict[str, dict[str, list[int]]]:
    rows = list(csv.DictReader(csv_path.open()))
    models = sorted({row["model_label"] for row in rows})
    out: dict[str, dict[str, list[int]]] = {}
    for model in models:
        sub = [row for row in rows if row["model_label"] == model]
        sub = sorted(sub, key=lambda row: float(row["reward_score"]))
        out[model] = {
            "low": [int(row["seed"]) for row in sub[:top_k]],
            "high": [int(row["seed"]) for row in sub[-top_k:]],
        }
    return out


def sample_sigma_grid(
    G,
    z: torch.Tensor,
    conditioning_params: torch.Tensor,
    shape_res: int,
    truncation_psi: float,
    truncation_cutoff: int,
    max_batch: int,
) -> torch.Tensor:
    cube_length = float(G.rendering_kwargs["box_warp"])
    samples, _, _ = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=cube_length)
    samples = samples.to(z.device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
    dirs = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
    dirs[..., -1] = -1
    head = 0
    while head < samples.shape[1]:
        chunk = min(max_batch, samples.shape[1] - head)
        sigma = G.sample(
            coordinates=samples[:, head : head + chunk],
            directions=dirs[:, :chunk],
            z=z,
            c=conditioning_params,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            noise_mode="const",
        )["sigma"]
        sigmas[:, head : head + chunk] = sigma
        head += chunk
    return sigmas.reshape(shape_res, shape_res, shape_res)


def mean_pairwise_distance(X: torch.Tensor) -> float:
    if X.shape[0] < 2:
        return 0.0
    d = torch.cdist(X.float(), X.float())
    tri = torch.triu_indices(d.shape[0], d.shape[1], offset=1)
    vals = d[tri[0], tri[1]]
    return float(vals.mean().item())


def centroid_distance(Xa: torch.Tensor, Xb: torch.Tensor) -> float:
    return float(torch.linalg.norm(Xa.float().mean(0) - Xb.float().mean(0)).item())


def fisher_ratio(Xa: torch.Tensor, Xb: torch.Tensor) -> float:
    within = mean_pairwise_distance(Xa) + mean_pairwise_distance(Xb)
    if within <= 1e-12:
        return 0.0
    return centroid_distance(Xa, Xb) / (within / 2.0)


def effective_rank(X: torch.Tensor) -> float:
    X = X.float() - X.float().mean(0, keepdim=True)
    s = torch.linalg.svdvals(X)
    s = s[s > 1e-12]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()).item())


def build_region_masks(shape_res: int, tuned_high_mean_density: torch.Tensor) -> dict[str, torch.Tensor]:
    coords = torch.linspace(-1.0, 1.0, shape_res)
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
    center_box = (xx.abs() <= 0.35) & (yy.abs() <= 0.35) & (zz.abs() <= 0.35)
    front_face = (zz >= 0.10) & (xx.abs() <= 0.45) & (yy.abs() <= 0.50)
    nose = (zz >= 0.20) & (xx.abs() <= 0.20) & (yy.abs() <= 0.22)
    thresh = torch.quantile(tuned_high_mean_density.flatten(), 0.90)
    high_sigma_core = tuned_high_mean_density >= thresh
    return {
        "center_box": center_box,
        "front_face": front_face,
        "nose": nose,
        "high_sigma_core": high_sigma_core,
    }


def flatten_region(density_grids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return density_grids[:, mask].float()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, contrast_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Localized Reward-Conditioned Sigma Analysis",
        "",
        "A stronger localized signal would look like:",
        "- `within_ratio_low_over_high > 1`",
        "- `effective_rank_ratio_low_over_high > 1`",
        "- a noticeably larger effect in tuned `01446` than in untuned EG3D",
        "",
    ]
    for region in ["nose", "front_face", "center_box", "high_sigma_core"]:
        lines.append(f"## {region}")
        lines.append("")
        for row in contrast_rows:
            if row["region"] != region:
                continue
            lines.append(
                f"- `{row['model_label']}`: within ratio `{float(row['within_ratio_low_over_high']):.4f}`, "
                f"effective-rank ratio `{float(row['effective_rank_ratio_low_over_high']):.4f}`, "
                f"Fisher ratio `{float(row['fisher_ratio']):.4f}`."
            )
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.base).resolve()
    target_path = Path(args.target).resolve()
    G_base = load_generator(base_path, device)
    G_target = load_generator(target_path, device)
    c = rm_camera_utils.get_single_dmap_camera().to(device)

    groups = load_reward_groups(args.scores_csv, args.top_k)
    base_label = f"untuned_{snapshot_label(base_path)}"
    target_label = f"tuned_{snapshot_label(target_path)}"

    seed_bank = sorted(set(groups[base_label]["low"] + groups[base_label]["high"] + groups[target_label]["low"] + groups[target_label]["high"]))

    sampled: dict[str, dict[int, torch.Tensor]] = {base_label: {}, target_label: {}}
    for model_label, G in [(base_label, G_base), (target_label, G_target)]:
        for seed in seed_bank:
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            sigma = sample_sigma_grid(
                G,
                z,
                c,
                shape_res=args.shape_res,
                truncation_psi=args.truncation_psi,
                truncation_cutoff=args.truncation_cutoff,
                max_batch=args.max_batch,
            )
            sampled[model_label][seed] = F.softplus(sigma - 1.0).detach().cpu()

    tuned_high_density = torch.stack([sampled[target_label][seed] for seed in groups[target_label]["high"]], dim=0).mean(0)
    masks = build_region_masks(args.shape_res, tuned_high_density)

    summary_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []

    for model_label in [base_label, target_label]:
        low_grids = torch.stack([sampled[model_label][seed] for seed in groups[model_label]["low"]], dim=0)
        high_grids = torch.stack([sampled[model_label][seed] for seed in groups[model_label]["high"]], dim=0)
        for region_name, mask in masks.items():
            low_X = flatten_region(low_grids, mask)
            high_X = flatten_region(high_grids, mask)

            low_within = mean_pairwise_distance(low_X)
            high_within = mean_pairwise_distance(high_X)
            low_rank = effective_rank(low_X)
            high_rank = effective_rank(high_X)

            summary_rows.append(
                {
                    "model_label": model_label,
                    "region": region_name,
                    "group": "low_reward",
                    "n_voxels": int(mask.sum().item()),
                    "within_pairwise_distance": low_within,
                    "effective_rank": low_rank,
                    "mean_density_mass": float(low_grids[:, mask].sum(1).mean().item()),
                    "mean_density_value": float(low_grids[:, mask].mean().item()),
                }
            )
            summary_rows.append(
                {
                    "model_label": model_label,
                    "region": region_name,
                    "group": "high_reward",
                    "n_voxels": int(mask.sum().item()),
                    "within_pairwise_distance": high_within,
                    "effective_rank": high_rank,
                    "mean_density_mass": float(high_grids[:, mask].sum(1).mean().item()),
                    "mean_density_value": float(high_grids[:, mask].mean().item()),
                }
            )
            contrast_rows.append(
                {
                    "model_label": model_label,
                    "region": region_name,
                    "n_voxels": int(mask.sum().item()),
                    "within_low": low_within,
                    "within_high": high_within,
                    "within_ratio_low_over_high": low_within / max(high_within, 1e-8),
                    "effective_rank_low": low_rank,
                    "effective_rank_high": high_rank,
                    "effective_rank_ratio_low_over_high": low_rank / max(high_rank, 1e-8),
                    "centroid_distance": centroid_distance(low_X, high_X),
                    "fisher_ratio": fisher_ratio(low_X, high_X),
                    "mean_density_mass_low": float(low_grids[:, mask].sum(1).mean().item()),
                    "mean_density_mass_high": float(high_grids[:, mask].sum(1).mean().item()),
                    "mean_density_value_low": float(low_grids[:, mask].mean().item()),
                    "mean_density_value_high": float(high_grids[:, mask].mean().item()),
                }
            )

    write_csv(output_dir / "region_group_metrics.csv", summary_rows)
    write_csv(output_dir / "region_contrast_metrics.csv", contrast_rows)
    write_report(output_dir / "region_report.md", contrast_rows)
    (output_dir / "meta.json").write_text(
        json.dumps(
            {
                "base": str(base_path),
                "target": str(target_path),
                "scores_csv": str(args.scores_csv.resolve()),
                "shape_res": args.shape_res,
                "top_k": args.top_k,
                "seed_bank_size": len(seed_bank),
            },
            indent=2,
        )
    )
    print(f"Wrote localized region analysis to: {output_dir}")


if __name__ == "__main__":
    main()
