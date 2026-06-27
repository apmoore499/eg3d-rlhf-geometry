#!/usr/bin/env python3
"""Analyze region-masked differences between saved sigma-field comparisons.

Consumes per-seed `.npz` volumes written by `compare_snapshot_sigma_fields.py
--save-volumes` and computes masked metrics for a set of spatial regions.

The front/back axis is inferred from the baseline density mass imbalance so the
analysis is less dependent on hard-coded coordinate assumptions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_dir", type=Path, help="Directory produced by compare_snapshot_sigma_fields.py with --save-volumes.")
    parser.add_argument("--surface-band", type=float, default=2.0, help="Absolute sigma band for the surface-region mask.")
    parser.add_argument("--center-box", type=float, default=0.35, help="Half-width of the central box mask in normalized coordinates.")
    parser.add_argument("--nose-cross", type=float, default=0.22, help="Half-width for non-front axes in the nose-candidate mask.")
    parser.add_argument("--front-quantile", type=float, default=0.25, help="Front-side quantile used for the nose-candidate mask.")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def tensor_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0:
        return math.nan
    return float(np.dot(a_flat, b_flat) / denom)


def make_coords(shape_res: int) -> np.ndarray:
    coords = np.linspace(-1.0, 1.0, shape_res, dtype=np.float32)
    zz, yy, xx = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.stack([xx, yy, zz], axis=0)


def infer_front_axis(density_base: np.ndarray) -> tuple[int, str]:
    total = float(density_base.sum())
    best_axis = 0
    best_dir = "low"
    best_score = -1.0
    for axis in range(3):
        half = density_base.shape[axis] // 2
        low = float(np.take(density_base, indices=range(0, half), axis=axis).sum())
        high = float(np.take(density_base, indices=range(half, density_base.shape[axis]), axis=axis).sum())
        score = abs(low - high) / total if total > 0 else 0.0
        if score > best_score:
            best_score = score
            best_axis = axis
            best_dir = "low" if low >= high else "high"
    return best_axis, best_dir


def build_masks(
    sigma_base: np.ndarray,
    density_base: np.ndarray,
    coords_xyz: np.ndarray,
    surface_band: float,
    center_box: float,
    nose_cross: float,
    front_quantile: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    front_axis_raw, front_dir = infer_front_axis(density_base)
    axis_names = ["z", "y", "x"]  # array axes
    coord_axis_map = {0: 2, 1: 1, 2: 0}  # array axis -> coords_xyz axis
    front_coord_axis = coord_axis_map[front_axis_raw]

    occupied = sigma_base > 1.0
    center_mask = np.max(np.abs(coords_xyz), axis=0) <= center_box
    surface_mask = np.abs(sigma_base) <= surface_band

    coord_front = coords_xyz[front_coord_axis]
    occ_vals = coord_front[occupied]
    if occ_vals.size == 0:
        cutoff = 0.0
    elif front_dir == "low":
        cutoff = float(np.quantile(occ_vals, front_quantile))
    else:
        cutoff = float(np.quantile(occ_vals, 1.0 - front_quantile))

    if front_dir == "low":
        front_half = coord_front <= 0.0
        rear_half = coord_front > 0.0
        nose_front = coord_front <= cutoff
    else:
        front_half = coord_front >= 0.0
        rear_half = coord_front < 0.0
        nose_front = coord_front >= cutoff

    other_axes = [idx for idx in range(3) if idx != front_coord_axis]
    nose_cross_mask = (np.abs(coords_xyz[other_axes[0]]) <= nose_cross) & (np.abs(coords_xyz[other_axes[1]]) <= nose_cross)
    face_core_mask = front_half & (
        (np.abs(coords_xyz[other_axes[0]]) <= center_box) & (np.abs(coords_xyz[other_axes[1]]) <= center_box)
    )
    nose_candidate = nose_front & nose_cross_mask

    masks = {
        "whole": np.ones_like(sigma_base, dtype=bool),
        "occupied_sigma0": sigma_base > 0.0,
        "occupied_sigma1": sigma_base > 1.0,
        "high_density_sigma10": sigma_base > 10.0,
        "surface_band": surface_mask,
        "center_box": center_mask,
        "front_half_auto": front_half,
        "rear_half_auto": rear_half,
        "face_core_auto": face_core_mask,
        "nose_candidate_auto": nose_candidate,
    }
    meta = {
        "front_array_axis": axis_names[front_axis_raw],
        "front_dir": front_dir,
        "front_cutoff": cutoff,
    }
    return masks, meta


def masked_metrics(
    sigma_base: np.ndarray,
    sigma_target: np.ndarray,
    density_base: np.ndarray,
    density_target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    if mask.sum() == 0:
        return {
            "voxel_fraction": 0.0,
            "sigma_l1_mean": math.nan,
            "sigma_rmse": math.nan,
            "sigma_cosine": math.nan,
            "density_l1_mean": math.nan,
            "density_rmse": math.nan,
            "density_cosine": math.nan,
            "density_mass_base": math.nan,
            "density_mass_target": math.nan,
            "density_mass_delta": math.nan,
            "positive_flip_fraction_sigma0": math.nan,
            "negative_flip_fraction_sigma0": math.nan,
        }

    sb = sigma_base[mask]
    st = sigma_target[mask]
    db = density_base[mask]
    dt = density_target[mask]
    return {
        "voxel_fraction": float(mask.mean()),
        "sigma_l1_mean": float(np.mean(np.abs(st - sb))),
        "sigma_rmse": float(np.sqrt(np.mean((st - sb) ** 2))),
        "sigma_cosine": tensor_cosine(sb, st),
        "density_l1_mean": float(np.mean(np.abs(dt - db))),
        "density_rmse": float(np.sqrt(np.mean((dt - db) ** 2))),
        "density_cosine": tensor_cosine(db, dt),
        "density_mass_base": float(db.sum()),
        "density_mass_target": float(dt.sum()),
        "density_mass_delta": float(dt.sum() - db.sum()),
        "positive_flip_fraction_sigma0": float(np.mean((sb <= 0.0) & (st > 0.0))),
        "negative_flip_fraction_sigma0": float(np.mean((sb > 0.0) & (st <= 0.0))),
    }


def main() -> None:
    args = parse_args()
    volume_files = sorted(args.comparison_dir.glob("seed_*_volumes.npz"))
    if not volume_files:
        raise FileNotFoundError(f"No saved volume files found in {args.comparison_dir}. Run compare_snapshot_sigma_fields.py with --save-volumes.")

    per_seed_rows: list[dict[str, object]] = []
    coords_xyz = None

    for volume_file in volume_files:
        seed = int(volume_file.stem.split("_")[1])
        data = np.load(volume_file)
        sigma_base = data["sigma_base"]
        sigma_target = data["sigma_target"]
        density_base = data["density_base"]
        density_target = data["density_target"]
        if coords_xyz is None or coords_xyz.shape[1:] != sigma_base.shape:
            coords_xyz = make_coords(sigma_base.shape[0])

        masks, meta = build_masks(
            sigma_base=sigma_base,
            density_base=density_base,
            coords_xyz=coords_xyz,
            surface_band=args.surface_band,
            center_box=args.center_box,
            nose_cross=args.nose_cross,
            front_quantile=args.front_quantile,
        )
        for mask_name, mask in masks.items():
            row = {
                "seed": seed,
                "mask_name": mask_name,
                **meta,
                **masked_metrics(sigma_base, sigma_target, density_base, density_target, mask),
            }
            per_seed_rows.append(row)

    write_csv(args.comparison_dir / "region_mask_metrics.csv", per_seed_rows)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in per_seed_rows:
        grouped[str(row["mask_name"])].append(row)

    summary_rows: list[dict[str, object]] = []
    numeric_fields = [
        "voxel_fraction",
        "sigma_l1_mean",
        "sigma_rmse",
        "sigma_cosine",
        "density_l1_mean",
        "density_rmse",
        "density_cosine",
        "density_mass_base",
        "density_mass_target",
        "density_mass_delta",
        "positive_flip_fraction_sigma0",
        "negative_flip_fraction_sigma0",
    ]
    for mask_name, rows in sorted(grouped.items()):
        summary = {"mask_name": mask_name}
        axis_dirs = defaultdict(int)
        for row in rows:
            axis_dirs[(row["front_array_axis"], row["front_dir"])] += 1
        dominant = max(axis_dirs.items(), key=lambda item: item[1])[0]
        summary["dominant_front_array_axis"] = dominant[0]
        summary["dominant_front_dir"] = dominant[1]
        for field in numeric_fields:
            vals = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
            summary[f"{field}_mean"] = float(np.mean(vals)) if vals else math.nan
            summary[f"{field}_std"] = float(np.std(vals)) if vals else math.nan
        summary_rows.append(summary)

    write_csv(args.comparison_dir / "region_mask_summary.csv", summary_rows)
    (args.comparison_dir / "region_mask_config.json").write_text(
        json.dumps(
            {
                "surface_band": args.surface_band,
                "center_box": args.center_box,
                "nose_cross": args.nose_cross,
                "front_quantile": args.front_quantile,
            },
            indent=2,
        )
    )
    print(f"Wrote per-seed region metrics to: {args.comparison_dir / 'region_mask_metrics.csv'}")
    print(f"Wrote summary region metrics to: {args.comparison_dir / 'region_mask_summary.csv'}")


if __name__ == "__main__":
    main()
