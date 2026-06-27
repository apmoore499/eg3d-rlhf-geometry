#!/usr/bin/env python3
"""Compare layerwise singular subspaces between two generator checkpoints."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import autoroot  # noqa: F401
import numpy as np
import torch

import legacy
from analyze_snapshot_svd import force_torch_load_map_location_cpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Baseline snapshot path")
    parser.add_argument("--target", required=True, help="Target snapshot path")
    parser.add_argument("--output-csv", type=Path, required=True, help="CSV path for per-layer metrics")
    parser.add_argument("--top-k", type=int, default=16, help="Top-k singular vectors for subspace angle analysis")
    parser.add_argument("--include-regex", action="append", default=[], help="Optional regex filter for parameter names")
    parser.add_argument("--exclude-regex", action="append", default=[], help="Optional regex exclusion for parameter names")
    return parser.parse_args()


def load_generator(path: Path):
    with path.open("rb") as handle:
        with force_torch_load_map_location_cpu():
            data = legacy.load_network_pkl(handle)
    return data["G_ema"].eval()


def matrix_from_tensor(t: torch.Tensor) -> torch.Tensor | None:
    if t.ndim < 2:
        return None
    if t.ndim == 2:
        return t.float()
    return t.reshape(t.shape[0], -1).float()


def compile_regexes(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in patterns]


def allowed(name: str, include: list[re.Pattern[str]], exclude: list[re.Pattern[str]]) -> bool:
    if include and not any(p.search(name) for p in include):
        return False
    if exclude and any(p.search(name) for p in exclude):
        return False
    return True


def principal_angle_stats(A: torch.Tensor, B: torch.Tensor, k: int) -> tuple[float, float]:
    k = min(k, A.shape[1], B.shape[1])
    if k < 1:
        return math.nan, math.nan
    M = A[:, :k].T @ B[:, :k]
    s = torch.linalg.svdvals(M).clamp(-1, 1)
    angles = torch.arccos(torch.clamp(s, -1.0, 1.0)) * (180.0 / math.pi)
    return float(angles.mean().item()), float(angles.max().item())


def tensor_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    include = compile_regexes(args.include_regex)
    exclude = compile_regexes(args.exclude_regex)

    G0 = load_generator(Path(args.base).resolve())
    G1 = load_generator(Path(args.target).resolve())
    sd0 = G0.state_dict()
    sd1 = G1.state_dict()

    rows: list[dict[str, object]] = []
    for name, tensor0 in sd0.items():
        if name not in sd1:
            continue
        if not name.endswith("weight"):
            continue
        if not allowed(name, include, exclude):
            continue
        matrix0 = matrix_from_tensor(tensor0)
        matrix1 = matrix_from_tensor(sd1[name])
        if matrix0 is None or matrix1 is None:
            continue

        U0, S0, Vh0 = torch.linalg.svd(matrix0, full_matrices=False)
        U1, S1, Vh1 = torch.linalg.svd(matrix1, full_matrices=False)
        rank = min(matrix0.shape[0], matrix0.shape[1], matrix1.shape[0], matrix1.shape[1])
        left_mean, left_max = principal_angle_stats(U0, U1, args.top_k)
        right_mean, right_max = principal_angle_stats(Vh0.T, Vh1.T, args.top_k)

        rows.append(
            {
                "parameter_name": name,
                "rows": int(matrix0.shape[0]),
                "cols": int(matrix0.shape[1]),
                "rank_cap": int(rank),
                "fro_rel_change": float((matrix1 - matrix0).norm().item() / matrix0.norm().clamp_min(1e-12).item()),
                "weight_cosine": tensor_cosine(matrix0, matrix1),
                "singular_value_cosine": tensor_cosine(S0, S1),
                "spectral_norm_delta": float(S1[0].item() - S0[0].item()),
                "nuclear_norm_rel_delta": float((S1.sum().item() - S0.sum().item()) / max(S0.sum().item(), 1e-12)),
                "left_subspace_mean_angle_deg": left_mean,
                "left_subspace_max_angle_deg": left_max,
                "right_subspace_mean_angle_deg": right_mean,
                "right_subspace_max_angle_deg": right_max,
            }
        )

    rows.sort(key=lambda row: row["right_subspace_mean_angle_deg"], reverse=True)
    write_csv(args.output_csv, rows)
    print(f"Wrote layerwise subspace metrics to: {args.output_csv}")


if __name__ == "__main__":
    main()
