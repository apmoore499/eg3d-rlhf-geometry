#!/usr/bin/env python3
"""Extract meshes from saved sigma volumes and compare them per seed."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import trimesh
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_dir", type=Path, help="Directory with seed_*_volumes.npz from compare_snapshot_sigma_fields.py --save-volumes")
    parser.add_argument("--level", type=float, default=10.0, help="Sigma isosurface level used for mesh extraction.")
    parser.add_argument("--surface-samples", type=int, default=20000, help="Number of sampled surface points per mesh for distance estimates.")
    parser.add_argument("--save-meshes", action="store_true", help="Export base/target meshes as PLY files.")
    parser.add_argument("--pad-voxels", type=int, default=3, help="Trim border voxels by setting them to a very negative value before marching cubes.")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def trim_volume(volume: np.ndarray, pad_voxels: int) -> np.ndarray:
    if pad_voxels <= 0:
        return volume
    out = volume.copy()
    out[:pad_voxels] = -1e6
    out[-pad_voxels:] = -1e6
    out[:, :pad_voxels] = -1e6
    out[:, -pad_voxels:] = -1e6
    out[:, :, :pad_voxels] = -1e6
    out[:, :, -pad_voxels:] = -1e6
    return out


def largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    parts = list(mesh.split(only_watertight=False))
    if len(parts) == 0:
        return mesh
    return max(parts, key=lambda m: m.area)


def extract_mesh(volume: np.ndarray, level: float, pad_voxels: int) -> trimesh.Trimesh | None:
    volume = trim_volume(volume, pad_voxels)
    try:
        verts, faces, normals, _ = marching_cubes(volume, level=level, spacing=[1.0, 1.0, 1.0])
    except Exception:
        return None
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return largest_component(mesh)


def sample_surface_points(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, count)
    return pts


def symmetric_surface_distances(points_a: np.ndarray, points_b: np.ndarray) -> dict[str, float]:
    tree_b = cKDTree(points_b)
    dist_a, _ = tree_b.query(points_a, k=1)
    tree_a = cKDTree(points_a)
    dist_b, _ = tree_a.query(points_b, k=1)
    chamfer = float(dist_a.mean() + dist_b.mean())
    hausdorff = float(max(dist_a.max(), dist_b.max()))
    return {
        "mean_nn_a_to_b": float(dist_a.mean()),
        "mean_nn_b_to_a": float(dist_b.mean()),
        "rms_nn_a_to_b": float(np.sqrt(np.mean(dist_a**2))),
        "rms_nn_b_to_a": float(np.sqrt(np.mean(dist_b**2))),
        "chamfer_mean_sum": chamfer,
        "hausdorff_approx": hausdorff,
    }


def save_overlay(path: Path, points_a: np.ndarray, points_b: np.ndarray) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    pairs = [
        ((0, 1), "XY"),
        ((0, 2), "XZ"),
        ((1, 2), "YZ"),
    ]
    for ax, ((i, j), title) in zip(axes, pairs):
        ax.scatter(points_a[:, i], points_a[:, j], s=0.2, alpha=0.25, label="base")
        ax.scatter(points_b[:, i], points_b[:, j], s=0.2, alpha=0.25, label="target")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.legend(markerscale=12, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def safe_mesh_volume(mesh: trimesh.Trimesh | None) -> float:
    if mesh is None or not mesh.is_volume:
        return math.nan
    return float(mesh.volume)


def main() -> None:
    args = parse_args()
    volume_files = sorted(args.comparison_dir.glob("seed_*_volumes.npz"))
    if not volume_files:
        raise FileNotFoundError(f"No saved volume files found in {args.comparison_dir}")

    rows: list[dict[str, object]] = []
    for volume_file in volume_files:
        seed = int(volume_file.stem.split("_")[1])
        data = np.load(volume_file)
        sigma_base = data["sigma_base"]
        sigma_target = data["sigma_target"]

        mesh_base = extract_mesh(sigma_base, args.level, args.pad_voxels)
        mesh_target = extract_mesh(sigma_target, args.level, args.pad_voxels)
        if mesh_base is None or mesh_target is None:
            rows.append({"seed": seed, "mesh_available": False})
            continue

        points_base = sample_surface_points(mesh_base, args.surface_samples)
        points_target = sample_surface_points(mesh_target, args.surface_samples)
        dists = symmetric_surface_distances(points_base, points_target)

        if args.save_meshes:
            mesh_base.export(args.comparison_dir / f"seed_{seed:04d}_base_level_{int(args.level)}.ply")
            mesh_target.export(args.comparison_dir / f"seed_{seed:04d}_target_level_{int(args.level)}.ply")
        save_overlay(args.comparison_dir / f"seed_{seed:04d}_mesh_overlay_level_{int(args.level)}.png", points_base, points_target)

        rows.append(
            {
                "seed": seed,
                "mesh_available": True,
                "vertex_count_base": int(len(mesh_base.vertices)),
                "vertex_count_target": int(len(mesh_target.vertices)),
                "face_count_base": int(len(mesh_base.faces)),
                "face_count_target": int(len(mesh_target.faces)),
                "area_base": float(mesh_base.area),
                "area_target": float(mesh_target.area),
                "area_delta": float(mesh_target.area - mesh_base.area),
                "volume_base": safe_mesh_volume(mesh_base),
                "volume_target": safe_mesh_volume(mesh_target),
                "volume_delta": safe_mesh_volume(mesh_target) - safe_mesh_volume(mesh_base)
                if math.isfinite(safe_mesh_volume(mesh_base)) and math.isfinite(safe_mesh_volume(mesh_target))
                else math.nan,
                **dists,
            }
        )

    write_csv(args.comparison_dir / f"mesh_diff_metrics_level_{int(args.level)}.csv", rows)
    print(f"Wrote mesh diff metrics to: {args.comparison_dir / f'mesh_diff_metrics_level_{int(args.level)}.csv'}")


if __name__ == "__main__":
    main()
