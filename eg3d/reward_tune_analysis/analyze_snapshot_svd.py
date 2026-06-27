#!/usr/bin/env python3
"""Analyze singular-value spectra for EG3D/PanoHead snapshot weights.

This script is intended for comparing how fine-tuned generator weights change
over a training trajectory, with emphasis on whether the normalized singular
value spectra remain approximately power-law / scale-free.

Default behavior:
- load `G_ema` from each snapshot pickle
- analyze weight parameters only
- flatten convolution kernels to `[out_channels, -1]`
- compute per-layer singular values and summary metrics
- write CSV/JSON outputs and summary plots

Typical usage:

    python analyze_snapshot_svd.py \
      /path/to/network-snapshot-002048.pkl \
      /path/to/network-snapshot-002053.pkl \
      /path/to/network-snapshot-002058.pkl \
      --output-dir outputs/weight_svd/eg3d_run_01446

Optional baseline comparison:

    python analyze_snapshot_svd.py \
      /path/to/tuned/network-snapshot-002048.pkl \
      /path/to/tuned/network-snapshot-002068.pkl \
      --baseline /path/to/untuned/network-snapshot-002000.pkl \
      --output-dir outputs/weight_svd/eg3d_vs_base

For PanoHead or other checkpoints that require additional class paths during
unpickling, provide one or more `--extra-sys-path` entries.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import sys
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import autoroot  # noqa: F401
import numpy as np
import torch

import legacy


SNAPSHOT_RE = re.compile(r"network-snapshot-(\d+)(?:_LAST)?\.pkl$")


@dataclass
class LayerStats:
    snapshot_path: str
    snapshot_label: str
    run_name: str
    snapshot_kimg: Optional[int]
    module_key: str
    parameter_name: str
    original_shape: str
    rows: int
    cols: int
    rank: int
    fro_norm: float
    spectral_norm: float
    nuclear_norm: float
    stable_rank: float
    stable_rank_fraction: float
    effective_rank: float
    effective_rank_fraction: float
    powerlaw_slope: float
    powerlaw_intercept: float
    powerlaw_r2: float
    energy_rank_90: int
    energy_rank_95: int


@dataclass
class SnapshotSummary:
    snapshot_path: str
    snapshot_label: str
    run_name: str
    snapshot_kimg: Optional[int]
    module_key: str
    layer_count: int
    total_rank: int
    total_fro_norm_sq: float
    max_spectral_norm: float
    weighted_mean_powerlaw_slope: float
    weighted_mean_powerlaw_r2: float
    median_powerlaw_slope: float
    median_powerlaw_r2: float
    weighted_mean_stable_rank_fraction: float
    weighted_mean_effective_rank_fraction: float
    baseline_weighted_mean_powerlaw_slope_delta: Optional[float]
    baseline_weighted_mean_powerlaw_r2_delta: Optional[float]
    baseline_weighted_mean_stable_rank_fraction_delta: Optional[float]
    baseline_weighted_mean_effective_rank_fraction_delta: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", nargs="+", help="Snapshot `.pkl` files to analyze.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/weight_svd"), help="Directory for analysis outputs.")
    parser.add_argument("--module-key", default="G_ema", choices=["G", "G_ema", "D"], help="Which object in the snapshot to analyze.")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional baseline snapshot for delta comparisons.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for SVD computation. `auto` prefers CUDA if available.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Matrix dtype used during SVD.",
    )
    parser.add_argument(
        "--include-regex",
        action="append",
        default=[],
        help="Only analyze parameters whose full names match at least one include regex.",
    )
    parser.add_argument(
        "--exclude-regex",
        action="append",
        default=[],
        help="Skip parameters whose full names match any exclude regex.",
    )
    parser.add_argument("--max-layers", type=int, default=None, help="Optional cap on analyzed parameter tensors after filtering.")
    parser.add_argument("--min-rank", type=int, default=2, help="Skip matrices with rank smaller than this.")
    parser.add_argument(
        "--extra-sys-path",
        action="append",
        default=[],
        help="Extra import paths to prepend before unpickling snapshots.",
    )
    parser.add_argument(
        "--skip-spectra-csv",
        action="store_true",
        help="Do not write the long-form singular spectrum CSV.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-layer progress while analyzing.",
    )
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def snapshot_kimg(path: Path) -> Optional[int]:
    match = SNAPSHOT_RE.search(path.name)
    return int(match.group(1)) if match else None


def snapshot_label(path: Path) -> str:
    kimg = snapshot_kimg(path)
    if kimg is None:
        return f"{path.parent.name}:{path.name}"
    return f"{path.parent.name}:{kimg:06d}"


def compile_regexes(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in patterns]


def parameter_allowed(name: str, include: list[re.Pattern[str]], exclude: list[re.Pattern[str]]) -> bool:
    if include and not any(p.search(name) for p in include):
        return False
    if exclude and any(p.search(name) for p in exclude):
        return False
    return True


def matrix_from_parameter(param: torch.Tensor) -> Optional[torch.Tensor]:
    if param.ndim < 2:
        return None
    if param.ndim == 2:
        return param
    return param.reshape(param.shape[0], -1)


def powerlaw_fit(singular_values: np.ndarray) -> tuple[float, float, float]:
    if singular_values.size < 3 or singular_values[0] <= 0:
        return math.nan, math.nan, math.nan

    normalized = singular_values / singular_values[0]
    ranks = np.arange(1, normalized.size + 1, dtype=np.float64)
    mask = normalized > 0
    if int(mask.sum()) < 3:
        return math.nan, math.nan, math.nan

    x = np.log(ranks[mask])
    y = np.log(normalized[mask])
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return float(slope), float(intercept), float(r2)


def effective_rank(singular_values: np.ndarray) -> float:
    total = float(np.sum(singular_values))
    if total <= 0:
        return math.nan
    probs = singular_values / total
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log(probs)))
    return float(np.exp(entropy))


def energy_rank(singular_values: np.ndarray, threshold: float) -> int:
    energy = singular_values**2
    total = float(np.sum(energy))
    if total <= 0:
        return 0
    cumulative = np.cumsum(energy) / total
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def compute_layer_stats(
    matrix: torch.Tensor,
    snapshot_path: Path,
    module_key: str,
    parameter_name: str,
    original_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[LayerStats, np.ndarray]:
    try:
        work = matrix.detach().to(device=device, dtype=dtype)
        s = torch.linalg.svdvals(work)
    except RuntimeError:
        # Fallback to CPU if CUDA SVD fails for a particular layer.
        work = matrix.detach().to(device="cpu", dtype=dtype)
        s = torch.linalg.svdvals(work)

    singular_values = s.detach().cpu().numpy().astype(np.float64, copy=False)
    if singular_values.size == 0:
        raise ValueError(f"No singular values returned for {parameter_name}")

    fro_norm = float(np.linalg.norm(singular_values))
    spectral_norm = float(singular_values[0])
    nuclear_norm = float(np.sum(singular_values))
    stable_rank = float((fro_norm * fro_norm) / (spectral_norm * spectral_norm)) if spectral_norm > 0 else math.nan
    eff_rank = effective_rank(singular_values)
    slope, intercept, r2 = powerlaw_fit(singular_values)
    rank = int(singular_values.size)

    stats = LayerStats(
        snapshot_path=str(snapshot_path),
        snapshot_label=snapshot_label(snapshot_path),
        run_name=snapshot_path.parent.name,
        snapshot_kimg=snapshot_kimg(snapshot_path),
        module_key=module_key,
        parameter_name=parameter_name,
        original_shape="x".join(str(v) for v in original_shape),
        rows=int(matrix.shape[0]),
        cols=int(matrix.shape[1]),
        rank=rank,
        fro_norm=fro_norm,
        spectral_norm=spectral_norm,
        nuclear_norm=nuclear_norm,
        stable_rank=stable_rank,
        stable_rank_fraction=float(stable_rank / rank) if rank > 0 and not math.isnan(stable_rank) else math.nan,
        effective_rank=eff_rank,
        effective_rank_fraction=float(eff_rank / rank) if rank > 0 and not math.isnan(eff_rank) else math.nan,
        powerlaw_slope=slope,
        powerlaw_intercept=intercept,
        powerlaw_r2=r2,
        energy_rank_90=energy_rank(singular_values, 0.90),
        energy_rank_95=energy_rank(singular_values, 0.95),
    )
    return stats, singular_values


@contextmanager
def force_torch_load_map_location_cpu():
    original_torch_load = torch.load

    def torch_load_cpu(*args, **kwargs):
        kwargs.setdefault("map_location", torch.device("cpu"))
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_cpu
    try:
        yield
    finally:
        torch.load = original_torch_load


def load_snapshot_module(snapshot_path: Path, module_key: str):
    with snapshot_path.open("rb") as handle:
        with force_torch_load_map_location_cpu():
            data = legacy.load_network_pkl(handle)
    if module_key not in data or data[module_key] is None:
        raise KeyError(f"{module_key} not found in {snapshot_path}")
    module = data[module_key]
    module.eval()
    return module


def iter_weight_matrices(
    module,
    include: list[re.Pattern[str]],
    exclude: list[re.Pattern[str]],
    max_layers: Optional[int],
    min_rank: int,
):
    yielded = 0
    for name, param in module.named_parameters():
        if not name.endswith("weight"):
            continue
        if not parameter_allowed(name, include, exclude):
            continue
        matrix = matrix_from_parameter(param)
        if matrix is None:
            continue
        if min(matrix.shape) < min_rank:
            continue
        yield name, param.shape, matrix
        yielded += 1
        if max_layers is not None and yielded >= max_layers:
            break


def summarize_snapshot(
    stats_rows: list[LayerStats],
    baseline_summary: Optional[SnapshotSummary],
) -> SnapshotSummary:
    ranks = np.array([row.rank for row in stats_rows], dtype=np.float64)
    weights = ranks / ranks.sum()
    slopes = np.array([row.powerlaw_slope for row in stats_rows], dtype=np.float64)
    r2s = np.array([row.powerlaw_r2 for row in stats_rows], dtype=np.float64)
    stable_fracs = np.array([row.stable_rank_fraction for row in stats_rows], dtype=np.float64)
    eff_fracs = np.array([row.effective_rank_fraction for row in stats_rows], dtype=np.float64)

    def weighted_mean(values: np.ndarray) -> float:
        mask = np.isfinite(values)
        if not np.any(mask):
            return math.nan
        w = weights[mask]
        w = w / w.sum()
        return float(np.sum(values[mask] * w))

    def median(values: np.ndarray) -> float:
        mask = np.isfinite(values)
        return float(np.median(values[mask])) if np.any(mask) else math.nan

    sample = stats_rows[0]
    summary = SnapshotSummary(
        snapshot_path=sample.snapshot_path,
        snapshot_label=sample.snapshot_label,
        run_name=sample.run_name,
        snapshot_kimg=sample.snapshot_kimg,
        module_key=sample.module_key,
        layer_count=len(stats_rows),
        total_rank=int(np.sum(ranks)),
        total_fro_norm_sq=float(sum(row.fro_norm * row.fro_norm for row in stats_rows)),
        max_spectral_norm=float(max(row.spectral_norm for row in stats_rows)),
        weighted_mean_powerlaw_slope=weighted_mean(slopes),
        weighted_mean_powerlaw_r2=weighted_mean(r2s),
        median_powerlaw_slope=median(slopes),
        median_powerlaw_r2=median(r2s),
        weighted_mean_stable_rank_fraction=weighted_mean(stable_fracs),
        weighted_mean_effective_rank_fraction=weighted_mean(eff_fracs),
        baseline_weighted_mean_powerlaw_slope_delta=None,
        baseline_weighted_mean_powerlaw_r2_delta=None,
        baseline_weighted_mean_stable_rank_fraction_delta=None,
        baseline_weighted_mean_effective_rank_fraction_delta=None,
    )
    if baseline_summary is not None:
        summary.baseline_weighted_mean_powerlaw_slope_delta = (
            summary.weighted_mean_powerlaw_slope - baseline_summary.weighted_mean_powerlaw_slope
        )
        summary.baseline_weighted_mean_powerlaw_r2_delta = (
            summary.weighted_mean_powerlaw_r2 - baseline_summary.weighted_mean_powerlaw_r2
        )
        summary.baseline_weighted_mean_stable_rank_fraction_delta = (
            summary.weighted_mean_stable_rank_fraction - baseline_summary.weighted_mean_stable_rank_fraction
        )
        summary.baseline_weighted_mean_effective_rank_fraction_delta = (
            summary.weighted_mean_effective_rank_fraction - baseline_summary.weighted_mean_effective_rank_fraction
        )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_spectra_csv(path: Path, spectra_rows: list[dict]) -> None:
    with gzip.open(path, "wt", newline="") as handle:
        if not spectra_rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(spectra_rows[0].keys()))
        writer.writeheader()
        writer.writerows(spectra_rows)


def sort_summaries(summaries: list[SnapshotSummary]) -> list[SnapshotSummary]:
    return sorted(
        summaries,
        key=lambda row: (
            row.run_name,
            -1 if row.snapshot_kimg is None else row.snapshot_kimg,
            row.snapshot_path,
        ),
    )


def make_aggregate_curve(spectra_by_layer: list[np.ndarray], points: int = 96) -> tuple[np.ndarray, np.ndarray]:
    if not spectra_by_layer:
        return np.array([]), np.array([])

    min_rank_fraction = min(1.0 / singular_values.size for singular_values in spectra_by_layer if singular_values.size > 0)
    q = np.geomspace(min_rank_fraction, 1.0, points)
    curves = []
    for singular_values in spectra_by_layer:
        normalized = singular_values / singular_values[0]
        rank_frac = np.arange(1, singular_values.size + 1, dtype=np.float64) / singular_values.size
        y = np.full_like(q, np.nan, dtype=np.float64)
        mask = q >= rank_frac[0]
        if np.any(mask):
            y[mask] = np.interp(q[mask], rank_frac, normalized)
        curves.append(y)

    stack = np.vstack(curves)
    return q, np.nanmedian(stack, axis=0)


def plot_summary(output_dir: Path, summaries: list[SnapshotSummary], spectra: dict[str, list[np.ndarray]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[str, list[SnapshotSummary]] = defaultdict(list)
    for row in summaries:
        grouped[row.run_name].append(row)

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=False)
    metrics = [
        ("weighted_mean_powerlaw_slope", "Weighted Mean Power-Law Slope"),
        ("weighted_mean_powerlaw_r2", "Weighted Mean Power-Law R^2"),
        ("weighted_mean_effective_rank_fraction", "Weighted Mean Effective Rank Fraction"),
    ]

    for ax, (field, title) in zip(axes, metrics):
        for run_name, rows in grouped.items():
            rows = sorted(rows, key=lambda r: (-1 if r.snapshot_kimg is None else r.snapshot_kimg, r.snapshot_label))
            xs = [row.snapshot_kimg if row.snapshot_kimg is not None else idx for idx, row in enumerate(rows)]
            ys = [getattr(row, field) for row in rows]
            ax.plot(xs, ys, marker="o", label=run_name)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Snapshot kimg")
    fig.tight_layout()
    fig.savefig(output_dir / "summary_metrics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 7))
    for label, layer_spectra in spectra.items():
        q, median_curve = make_aggregate_curve(layer_spectra)
        if q.size == 0:
            continue
        ax.plot(q, median_curve, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Normalized Rank")
    ax.set_ylabel("Median Normalized Singular Value")
    ax.set_title("Aggregate Normalized Spectra by Snapshot")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "aggregate_normalized_spectra.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    for extra in args.extra_sys_path:
        if extra not in sys.path:
            sys.path.insert(0, extra)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    dtype = select_dtype(args.dtype)
    include = compile_regexes(args.include_regex)
    exclude = compile_regexes(args.exclude_regex)

    config = {
        "snapshots": args.snapshots,
        "baseline": str(args.baseline) if args.baseline else None,
        "module_key": args.module_key,
        "device": str(device),
        "dtype": args.dtype,
        "include_regex": args.include_regex,
        "exclude_regex": args.exclude_regex,
        "max_layers": args.max_layers,
        "min_rank": args.min_rank,
        "extra_sys_path": args.extra_sys_path,
    }
    (output_dir / "analysis_config.json").write_text(json.dumps(config, indent=2))

    baseline_summary: Optional[SnapshotSummary] = None
    if args.baseline is not None:
        baseline_module = load_snapshot_module(args.baseline, args.module_key)
        baseline_rows: list[LayerStats] = []
        for name, shape, matrix in iter_weight_matrices(baseline_module, include, exclude, args.max_layers, args.min_rank):
            stats, _ = compute_layer_stats(matrix, args.baseline, args.module_key, name, tuple(shape), device, dtype)
            baseline_rows.append(stats)
        if not baseline_rows:
            raise RuntimeError("Baseline snapshot produced no analyzable weight matrices.")
        baseline_summary = summarize_snapshot(baseline_rows, baseline_summary=None)
        write_csv(output_dir / "baseline_layer_stats.csv", [asdict(row) for row in baseline_rows])
        write_csv(output_dir / "baseline_summary.csv", [asdict(baseline_summary)])

    all_layer_rows: list[LayerStats] = []
    all_spectra_rows: list[dict] = []
    snapshot_summaries: list[SnapshotSummary] = []
    spectra_for_plots: dict[str, list[np.ndarray]] = defaultdict(list)

    for snapshot_str in args.snapshots:
        snapshot_path = Path(snapshot_str).resolve()
        module = load_snapshot_module(snapshot_path, args.module_key)
        layer_rows: list[LayerStats] = []

        for name, shape, matrix in iter_weight_matrices(module, include, exclude, args.max_layers, args.min_rank):
            if args.verbose:
                print(f"[{snapshot_label(snapshot_path)}] {name}  {tuple(shape)} -> {tuple(matrix.shape)}")
            stats, singular_values = compute_layer_stats(matrix, snapshot_path, args.module_key, name, tuple(shape), device, dtype)
            layer_rows.append(stats)
            spectra_for_plots[stats.snapshot_label].append(singular_values)
            if not args.skip_spectra_csv:
                for idx, sigma in enumerate(singular_values, start=1):
                    all_spectra_rows.append(
                        {
                            "snapshot_path": str(snapshot_path),
                            "snapshot_label": stats.snapshot_label,
                            "run_name": stats.run_name,
                            "snapshot_kimg": stats.snapshot_kimg,
                            "module_key": args.module_key,
                            "parameter_name": name,
                            "rank_index": idx,
                            "rank_fraction": idx / singular_values.size,
                            "singular_value": float(sigma),
                            "normalized_singular_value": float(sigma / singular_values[0]),
                        }
                    )

        if not layer_rows:
            raise RuntimeError(f"{snapshot_path} produced no analyzable weight matrices.")

        summary = summarize_snapshot(layer_rows, baseline_summary)
        all_layer_rows.extend(layer_rows)
        snapshot_summaries.append(summary)

    ordered_summaries = sort_summaries(snapshot_summaries)
    write_csv(output_dir / "snapshot_summary.csv", [asdict(row) for row in ordered_summaries])
    write_csv(output_dir / "layer_stats.csv", [asdict(row) for row in all_layer_rows])
    if not args.skip_spectra_csv:
        write_spectra_csv(output_dir / "spectra_long.csv.gz", all_spectra_rows)
    plot_summary(output_dir, ordered_summaries, spectra_for_plots)

    print(f"Wrote snapshot summary to: {output_dir / 'snapshot_summary.csv'}")
    print(f"Wrote layer stats to: {output_dir / 'layer_stats.csv'}")
    if not args.skip_spectra_csv:
        print(f"Wrote spectra rows to: {output_dir / 'spectra_long.csv.gz'}")
    print(f"Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()
