#!/usr/bin/env python3
"""Plot aggregate singular-value spectrum profiles from snapshot SVD outputs.

Consumes one or more directories produced by `analyze_snapshot_svd.py` and
generates comparison plots in several coordinate systems:

- linear rank vs normalized singular value
- semilogy rank vs normalized singular value
- log-log rank vs normalized singular value
- cumulative energy curves

The semilogy view is useful if the spectrum is approximately exponential.
The log-log view is useful if the spectrum is approximately power-law /
scale-free.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dirs", nargs="+", type=Path, help="Directories produced by analyze_snapshot_svd.py")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for output plots and summary CSV")
    parser.add_argument("--points", type=int, default=128, help="Interpolation points for aggregate curves")
    parser.add_argument(
        "--snapshot-mode",
        choices=["last", "all"],
        default="last",
        help="Whether to use only the last snapshot per run or all snapshots.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def maybe_int(value: str) -> int | None:
    if value in ("", "None"):
        return None
    return int(value)


def sort_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            row["run_name"],
            -1 if maybe_int(row["snapshot_kimg"]) is None else maybe_int(row["snapshot_kimg"]),
            row["snapshot_path"],
        ),
    )


def choose_snapshots(summary_rows: list[dict[str, str]], mode: str) -> list[str]:
    ordered = sort_summary_rows(summary_rows)
    if mode == "all":
        return [row["snapshot_path"] for row in ordered]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ordered:
        grouped[row["run_name"]].append(row)
    return [rows[-1]["snapshot_path"] for _, rows in sorted(grouped.items())]


def aggregate_curves(rows: list[dict[str, str]], points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectra_by_layer: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        key = (row["snapshot_label"], row["parameter_name"])
        spectra_by_layer[key].append((float(row["rank_fraction"]), float(row["normalized_singular_value"])))

    min_rank_fraction = 1.0
    for values in spectra_by_layer.values():
        min_rank_fraction = min(min_rank_fraction, min(rank_fraction for rank_fraction, _ in values))
    q = np.geomspace(min_rank_fraction, 1.0, points)
    spectrum_curves = []
    energy_curves = []
    for values in spectra_by_layer.values():
        values = sorted(values, key=lambda item: item[0])
        rank_frac = np.array([v[0] for v in values], dtype=np.float64)
        normalized_sigma = np.array([v[1] for v in values], dtype=np.float64)
        spectrum_curves.append(np.interp(q, rank_frac, normalized_sigma))

        energy = normalized_sigma**2
        energy = energy / energy.sum() if energy.sum() > 0 else energy
        cumulative = np.cumsum(energy)
        energy_curves.append(np.interp(q, rank_frac, cumulative))

    return q, np.median(np.vstack(spectrum_curves), axis=0), np.median(np.vstack(energy_curves), axis=0)


def fit_semilogy_decay(q: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(q) & np.isfinite(y) & (y > 0)
    if mask.sum() < 3:
        return math.nan, math.nan
    x = q[mask]
    logy = np.log(y[mask])
    slope, intercept = np.polyfit(x, logy, 1)
    return float(slope), float(intercept)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

    run_curves: list[dict[str, object]] = []

    for analysis_dir in args.analysis_dirs:
        summary_rows = read_csv(analysis_dir / "snapshot_summary.csv")
        spectra_rows = read_gzip_csv(analysis_dir / "spectra_long.csv.gz")
        selected_paths = set(choose_snapshots(summary_rows, args.snapshot_mode))
        selected_rows = [row for row in spectra_rows if row["snapshot_path"] in selected_paths]
        if not selected_rows:
            continue
        run_name = next(row["run_name"] for row in selected_rows)
        q, spectrum, cumulative_energy = aggregate_curves(selected_rows, args.points)
        exp_slope, exp_intercept = fit_semilogy_decay(q, spectrum)
        run_curves.append(
            {
                "run_name": run_name,
                "q": q,
                "spectrum": spectrum,
                "cumulative_energy": cumulative_energy,
                "semilogy_decay_slope": exp_slope,
                "semilogy_decay_intercept": exp_intercept,
            }
        )

    if not run_curves:
        raise RuntimeError("No curves were produced from the provided analysis dirs.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for row in run_curves:
        q = row["q"]
        spectrum = row["spectrum"]
        cumulative_energy = row["cumulative_energy"]
        label = row["run_name"]

        axes[0, 0].plot(q, spectrum, label=label)
        axes[0, 1].semilogy(q, spectrum, label=label)
        axes[1, 0].loglog(q, spectrum, label=label)
        axes[1, 1].plot(q, cumulative_energy, label=label)

    axes[0, 0].set_title("Linear Spectrum")
    axes[0, 0].set_xlabel("Normalized Rank")
    axes[0, 0].set_ylabel("Median Normalized Singular Value")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Semilogy Spectrum")
    axes[0, 1].set_xlabel("Normalized Rank")
    axes[0, 1].set_ylabel("Median Normalized Singular Value")
    axes[0, 1].grid(True, which="both", alpha=0.3)

    axes[1, 0].set_title("Log-Log Spectrum")
    axes[1, 0].set_xlabel("Normalized Rank")
    axes[1, 0].set_ylabel("Median Normalized Singular Value")
    axes[1, 0].grid(True, which="both", alpha=0.3)

    axes[1, 1].set_title("Cumulative Energy")
    axes[1, 1].set_xlabel("Normalized Rank")
    axes[1, 1].set_ylabel("Cumulative Spectral Energy")
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.output_dir / "spectrum_profile_views.png", dpi=180)
    plt.close(fig)

    summary_rows = []
    for row in run_curves:
        q = row["q"]
        cumulative_energy = row["cumulative_energy"]
        energy_90_idx = int(np.searchsorted(cumulative_energy, 0.90, side="left"))
        energy_95_idx = int(np.searchsorted(cumulative_energy, 0.95, side="left"))
        summary_rows.append(
            {
                "run_name": row["run_name"],
                "semilogy_decay_slope": row["semilogy_decay_slope"],
                "semilogy_decay_intercept": row["semilogy_decay_intercept"],
                "rank_fraction_at_90_energy": float(q[min(energy_90_idx, len(q) - 1)]),
                "rank_fraction_at_95_energy": float(q[min(energy_95_idx, len(q) - 1)]),
            }
        )
    write_csv(args.output_dir / "spectrum_profile_summary.csv", summary_rows)
    print(f"Wrote plots to: {args.output_dir / 'spectrum_profile_views.png'}")
    print(f"Wrote summary to: {args.output_dir / 'spectrum_profile_summary.csv'}")


if __name__ == "__main__":
    main()
