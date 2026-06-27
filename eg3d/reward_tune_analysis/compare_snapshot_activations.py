#!/usr/bin/env python3
"""Compare generator activations on fixed seeds and fixed 3D coordinates."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import autoroot  # noqa: F401
import numpy as np
import torch

import legacy
from analyze_snapshot_svd import force_torch_load_map_location_cpu
from camera_utils import FOV_to_intrinsics, LookAtPoseSampler
from training.volumetric_rendering.renderer import generate_planes, sample_from_planes


def parse_seed_list(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return list(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--seeds", default="0-7")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--coord-grid-res", type=int, default=18, help="Resolution of fixed 3D coordinate grid.")
    parser.add_argument("--fov-deg", type=float, default=18.837)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_generator(path: Path, device: torch.device):
    with path.open("rb") as handle:
        with force_torch_load_map_location_cpu():
            data = legacy.load_network_pkl(handle)
    return data["G_ema"].eval().requires_grad_(False).to(device)


def make_conditioning(G, device: torch.device, fov_deg: float) -> torch.Tensor:
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get("avg_camera_pivot", [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get("avg_camera_radius", 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=device)
    return torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)


def make_coord_grid(G, res: int, device: torch.device) -> torch.Tensor:
    cube_length = float(G.rendering_kwargs["box_warp"])
    lo = -cube_length / 2.0
    hi = cube_length / 2.0
    coords = torch.linspace(lo, hi, res, device=device)
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1).reshape(1, -1, 3)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X.float()
    Y = Y.float()
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic = torch.linalg.norm(X.T @ Y, ord="fro") ** 2
    denom = torch.linalg.norm(X.T @ X, ord="fro") * torch.linalg.norm(Y.T @ Y, ord="fro")
    if denom.item() == 0:
        return math.nan
    return float((hsic / denom).item())


def flat_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()


def get_representations(G, z: torch.Tensor, c: torch.Tensor, coords: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        ws = G.mapping(z, c, truncation_psi=0.7, truncation_cutoff=14)
        planes_flat = G.backbone.synthesis(ws, noise_mode="const")
        planes = planes_flat.view(len(planes_flat), 3, 32, planes_flat.shape[-2], planes_flat.shape[-1])
        plane_axes = generate_planes().to(z.device)
        sampled = sample_from_planes(plane_axes, planes, coords, padding_mode="zeros", box_warp=G.rendering_kwargs["box_warp"])
        decoder_input = sampled.mean(1).squeeze(0)
        hidden = G.decoder.net[1](G.decoder.net[0](decoder_input))
        final = G.decoder.net[2](hidden)
        return {
            "planes_features": planes.permute(0, 1, 3, 4, 2).reshape(-1, 32),
            "decoder_input": decoder_input,
            "decoder_hidden": hidden,
            "sigma_output": final[:, :1],
            "rgb_output_raw": final[:, 1:],
        }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    seeds = parse_seed_list(args.seeds)

    G0 = load_generator(Path(args.base).resolve(), device)
    G1 = load_generator(Path(args.target).resolve(), device)
    c0 = make_conditioning(G0, device, args.fov_deg)
    c1 = make_conditioning(G1, device, args.fov_deg)
    coords = make_coord_grid(G0, args.coord_grid_res, device)

    rows: list[dict[str, object]] = []
    for seed in seeds:
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G0.z_dim)).to(device)
        rep0 = get_representations(G0, z, c0, coords)
        rep1 = get_representations(G1, z, c1, coords)
        row = {"seed": seed}
        for key in ["planes_features", "decoder_input", "decoder_hidden", "sigma_output", "rgb_output_raw"]:
            row[f"{key}_cosine"] = flat_cosine(rep0[key], rep1[key])
            row[f"{key}_cka"] = linear_cka(rep0[key], rep1[key])
            row[f"{key}_rmse"] = float(torch.sqrt(torch.mean((rep1[key] - rep0[key]) ** 2)).item())
            row[f"{key}_l1"] = float(torch.mean(torch.abs(rep1[key] - rep0[key])).item())
        rows.append(row)

    write_csv(args.output_csv, rows)
    print(f"Wrote activation comparison to: {args.output_csv}")


if __name__ == "__main__":
    main()
