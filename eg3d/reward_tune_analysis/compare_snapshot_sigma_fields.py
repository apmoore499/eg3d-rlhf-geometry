#!/usr/bin/env python3
"""Compare sigma-field outputs between two EG3D/PanoHead snapshots.

For each requested seed, this script samples matched 3D sigma grids from both
snapshots using the canonical conditioning camera, then computes:

- raw sigma L1/L2/cosine metrics
- density-space metrics after softplus
- occupancy IoU at several sigma thresholds
- positive-flip / negative-flip voxel fractions
- simple slice visualizations of base / target / delta

This is intended as an output-space geometry comparison, complementing the
weight-space SVD analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import autoroot  # noqa: F401
import matplotlib
import numpy as np
import torch
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import legacy
from analyze_snapshot_svd import force_torch_load_map_location_cpu, snapshot_kimg, snapshot_label
from camera_utils import FOV_to_intrinsics, LookAtPoseSampler


def parse_seed_list(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(list(range(int(lo), int(hi) + 1)))
        else:
            seeds.append(int(part))
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Baseline snapshot path.")
    parser.add_argument("--target", required=True, help="Target/tuned snapshot path.")
    parser.add_argument("--seeds", required=True, help="Comma/range seed list, e.g. `0-7`.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for outputs.")
    parser.add_argument("--shape-res", type=int, default=64, help="Sigma grid resolution.")
    parser.add_argument("--max-batch", type=int, default=200000, help="Max query points per `G.sample` call.")
    parser.add_argument("--truncation-psi", type=float, default=0.7, help="Truncation psi.")
    parser.add_argument("--truncation-cutoff", type=int, default=14, help="Truncation cutoff.")
    parser.add_argument("--fov-deg", type=float, default=18.837, help="Camera FOV in degrees.")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--sigma-thresholds",
        default="0,1,10",
        help="Comma-separated sigma thresholds for occupancy IoU.",
    )
    parser.add_argument(
        "--extra-sys-path",
        action="append",
        default=[],
        help="Optional extra sys.path entries for checkpoint unpickling.",
    )
    parser.add_argument(
        "--save-volumes",
        action="store_true",
        help="Save per-seed compressed numpy volumes for reuse in downstream analyses.",
    )
    return parser.parse_args()


def load_generator(snapshot_path: Path, device: torch.device):
    with snapshot_path.open("rb") as handle:
        with force_torch_load_map_location_cpu():
            data = legacy.load_network_pkl(handle)
    G = data["G_ema"].eval().requires_grad_(False).to(device)
    return G


def make_conditioning(G, device: torch.device, fov_deg: float) -> torch.Tensor:
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get("avg_camera_pivot", [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get("avg_camera_radius", 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=device)
    return torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    voxel_origin = np.array(voxel_origin) - cube_length / 2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 3)
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    return samples.unsqueeze(0), voxel_origin, voxel_size


def sample_sigma_grid(
    G,
    z: torch.Tensor,
    conditioning_params: torch.Tensor,
    shape_res: int,
    max_batch: int,
    truncation_psi: float,
    truncation_cutoff: int,
) -> torch.Tensor:
    samples, _, _ = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs["box_warp"] * 1)
    samples = samples.to(z.device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
    transformed_ray_directions = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
    transformed_ray_directions[..., -1] = -1
    head = 0
    with torch.no_grad():
        while head < samples.shape[1]:
            tail = min(head + max_batch, samples.shape[1])
            sigma = G.sample(
                samples[:, head:tail],
                transformed_ray_directions[:, : tail - head],
                z,
                conditioning_params,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
                noise_mode="const",
            )["sigma"]
            sigmas[:, head:tail] = sigma
            head = tail
    return sigmas.reshape(shape_res, shape_res, shape_res)


def tensor_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()


def occupancy_iou(a: torch.Tensor, b: torch.Tensor, threshold: float) -> float:
    ma = a > threshold
    mb = b > threshold
    inter = torch.logical_and(ma, mb).sum().item()
    union = torch.logical_or(ma, mb).sum().item()
    if union == 0:
        return math.nan
    return inter / union


def density_center_of_mass(density: torch.Tensor) -> tuple[float, float, float]:
    N = density.shape[0]
    coords = torch.linspace(-1.0, 1.0, N, device=density.device)
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
    mass = density.sum()
    if mass.item() == 0:
        return math.nan, math.nan, math.nan
    return (
        (xx * density).sum().item() / mass.item(),
        (yy * density).sum().item() / mass.item(),
        (zz * density).sum().item() / mass.item(),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_slice_figure(path: Path, sigma_base: torch.Tensor, sigma_target: torch.Tensor, density_delta: torch.Tensor) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

    idx = sigma_base.shape[0] // 2
    slices = [
        ("axial", sigma_base[idx], sigma_target[idx], density_delta[idx]),
        ("coronal", sigma_base[:, idx, :], sigma_target[:, idx, :], density_delta[:, idx, :]),
        ("sagittal", sigma_base[:, :, idx], sigma_target[:, :, idx], density_delta[:, :, idx]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    for row_idx, (name, base_slice, target_slice, delta_slice) in enumerate(slices):
        panels = [
            (base_slice.detach().cpu().numpy(), f"{name}: base sigma"),
            (target_slice.detach().cpu().numpy(), f"{name}: target sigma"),
            (delta_slice.detach().cpu().numpy(), f"{name}: delta density"),
        ]
        for col_idx, (image, title) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            cmap = "coolwarm" if "delta" in title else "viridis"
            ax.imshow(image, cmap=cmap)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    for extra in args.extra_sys_path:
        if extra not in os.sys.path:
            os.sys.path.insert(0, extra)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seeds = parse_seed_list(args.seeds)
    thresholds = [float(x) for x in args.sigma_thresholds.split(",") if x.strip()]

    base_path = Path(args.base).resolve()
    target_path = Path(args.target).resolve()
    G_base = load_generator(base_path, device)
    G_target = load_generator(target_path, device)
    c_base = make_conditioning(G_base, device, args.fov_deg)
    c_target = make_conditioning(G_target, device, args.fov_deg)

    rows: list[dict[str, object]] = []
    for seed in seeds:
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G_base.z_dim)).to(device)
        sigma_base = sample_sigma_grid(G_base, z, c_base, args.shape_res, args.max_batch, args.truncation_psi, args.truncation_cutoff)
        sigma_target = sample_sigma_grid(G_target, z, c_target, args.shape_res, args.max_batch, args.truncation_psi, args.truncation_cutoff)

        density_base = torch.nn.functional.softplus(sigma_base)
        density_target = torch.nn.functional.softplus(sigma_target)
        sigma_delta = sigma_target - sigma_base
        density_delta = density_target - density_base

        row: dict[str, object] = {
            "seed": seed,
            "base_snapshot_label": snapshot_label(base_path),
            "target_snapshot_label": snapshot_label(target_path),
            "base_snapshot_kimg": snapshot_kimg(base_path),
            "target_snapshot_kimg": snapshot_kimg(target_path),
            "shape_res": args.shape_res,
            "sigma_l1_mean": torch.nn.functional.l1_loss(sigma_target, sigma_base).item(),
            "sigma_rmse": torch.sqrt(torch.mean((sigma_target - sigma_base) ** 2)).item(),
            "sigma_cosine": tensor_cosine(sigma_base, sigma_target),
            "density_l1_mean": torch.nn.functional.l1_loss(density_target, density_base).item(),
            "density_rmse": torch.sqrt(torch.mean((density_target - density_base) ** 2)).item(),
            "density_cosine": tensor_cosine(density_base, density_target),
            "density_total_mass_base": density_base.sum().item(),
            "density_total_mass_target": density_target.sum().item(),
            "density_total_mass_delta": (density_target.sum() - density_base.sum()).item(),
            "positive_flip_fraction_sigma0": torch.logical_and(sigma_base <= 0, sigma_target > 0).float().mean().item(),
            "negative_flip_fraction_sigma0": torch.logical_and(sigma_base > 0, sigma_target <= 0).float().mean().item(),
            "mean_density_delta_positive": torch.clamp_min(density_delta, 0).mean().item(),
            "mean_density_delta_negative": torch.clamp_max(density_delta, 0).mean().item(),
        }
        com_base = density_center_of_mass(density_base)
        com_target = density_center_of_mass(density_target)
        row["density_com_x_base"], row["density_com_y_base"], row["density_com_z_base"] = com_base
        row["density_com_x_target"], row["density_com_y_target"], row["density_com_z_target"] = com_target
        row["density_com_shift"] = math.sqrt(sum((a - b) ** 2 for a, b in zip(com_base, com_target)))

        for threshold in thresholds:
            key = str(threshold).replace(".", "p").replace("-", "m")
            row[f"occupancy_iou_sigma_{key}"] = occupancy_iou(sigma_base, sigma_target, threshold)

        rows.append(row)
        save_slice_figure(args.output_dir / f"seed_{seed:04d}_slices.png", sigma_base, sigma_target, density_delta)
        if args.save_volumes:
            np.savez_compressed(
                args.output_dir / f"seed_{seed:04d}_volumes.npz",
                sigma_base=sigma_base.detach().cpu().numpy(),
                sigma_target=sigma_target.detach().cpu().numpy(),
                density_base=density_base.detach().cpu().numpy(),
                density_target=density_target.detach().cpu().numpy(),
                density_delta=density_delta.detach().cpu().numpy(),
            )

    write_csv(args.output_dir / "sigma_field_metrics.csv", rows)

    summary: dict[str, object] = {
        "base_snapshot": str(base_path),
        "target_snapshot": str(target_path),
        "seeds": seeds,
        "shape_res": args.shape_res,
        "thresholds": thresholds,
    }
    if rows:
        numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
        for key in numeric_keys:
            vals = [float(row[key]) for row in rows if isinstance(row[key], (int, float)) and math.isfinite(float(row[key]))]
            if vals:
                summary[f"{key}_mean"] = float(np.mean(vals))
                summary[f"{key}_std"] = float(np.std(vals))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "base": str(base_path),
                "target": str(target_path),
                "seeds": seeds,
                "shape_res": args.shape_res,
                "max_batch": args.max_batch,
                "truncation_psi": args.truncation_psi,
                "truncation_cutoff": args.truncation_cutoff,
                "fov_deg": args.fov_deg,
                "device": args.device,
                "sigma_thresholds": thresholds,
                "extra_sys_path": args.extra_sys_path,
                "save_volumes": args.save_volumes,
            },
            indent=2,
        )
    )
    print(f"Wrote metrics to: {args.output_dir / 'sigma_field_metrics.csv'}")
    print(f"Wrote summary to: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
