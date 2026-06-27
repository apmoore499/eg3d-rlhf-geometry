#!/usr/bin/env python3
"""Compare snapshot-SVD analysis outputs and write a compact report.

This script consumes one or more output directories produced by
`analyze_snapshot_svd.py` and generates:

- a combined summary CSV across runs
- a per-run trend CSV
- a Markdown report with first/last metrics and major layer changes

It is designed to answer questions like:
- Do normalized singular spectra remain approximately power-law?
- Does fine-tuning make the spectra steeper or flatter?
- Which layers changed most over the tuning trajectory?
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


METRICS = [
    "weighted_mean_powerlaw_slope",
    "weighted_mean_powerlaw_r2",
    "weighted_mean_stable_rank_fraction",
    "weighted_mean_effective_rank_fraction",
    "max_spectral_norm",
    "total_fro_norm_sq",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "analysis_dirs",
        nargs="+",
        type=Path,
        help="Directories produced by analyze_snapshot_svd.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the comparison report outputs.",
    )
    parser.add_argument(
        "--top-layers",
        type=int,
        default=12,
        help="How many most-changed layers to include per run.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def maybe_float(value: str) -> float:
    if value in ("", "None", "nan", "NaN"):
        return math.nan
    return float(value)


def maybe_int(value: str) -> int | None:
    if value in ("", "None"):
        return None
    return int(value)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_num(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def format_signed(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:+.{digits}f}"


def sort_rows_by_kimg(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            row.get("run_name", ""),
            -1 if maybe_int(row.get("snapshot_kimg", "")) is None else maybe_int(row.get("snapshot_kimg", "")),
            row.get("snapshot_path", ""),
        ),
    )


def linear_trend(xs: list[float], ys: list[float]) -> float:
    valid = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(valid) < 2:
        return math.nan
    x_vals = [x for x, _ in valid]
    y_vals = [y for _, y in valid]
    mean_x = sum(x_vals) / len(x_vals)
    mean_y = sum(y_vals) / len(y_vals)
    denom = sum((x - mean_x) ** 2 for x in x_vals)
    if denom == 0:
        return math.nan
    numer = sum((x - mean_x) * (y - mean_y) for x, y in valid)
    return numer / denom


def make_run_trend_row(summary_rows: list[dict[str, str]]) -> dict[str, object]:
    ordered = sort_rows_by_kimg(summary_rows)
    start = ordered[0]
    end = ordered[-1]
    xs = [float(maybe_int(row.get("snapshot_kimg", "")) or idx) for idx, row in enumerate(ordered)]

    out: dict[str, object] = {
        "run_name": start["run_name"],
        "snapshot_count": len(ordered),
        "start_snapshot_kimg": maybe_int(start["snapshot_kimg"]),
        "end_snapshot_kimg": maybe_int(end["snapshot_kimg"]),
    }
    for metric in METRICS:
        ys = [maybe_float(row[metric]) for row in ordered]
        out[f"{metric}_start"] = ys[0]
        out[f"{metric}_end"] = ys[-1]
        out[f"{metric}_delta"] = ys[-1] - ys[0] if math.isfinite(ys[0]) and math.isfinite(ys[-1]) else math.nan
        out[f"{metric}_trend_per_kimg"] = linear_trend(xs, ys)
    return out


def index_layer_rows(layer_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in layer_rows:
        indexed[(row["snapshot_path"], row["parameter_name"])] = row
    return indexed


def compute_top_layer_changes(
    summary_rows: list[dict[str, str]],
    layer_rows: list[dict[str, str]],
    top_n: int,
) -> list[dict[str, object]]:
    ordered = sort_rows_by_kimg(summary_rows)
    first_path = ordered[0]["snapshot_path"]
    last_path = ordered[-1]["snapshot_path"]

    first_layers = index_layer_rows([row for row in layer_rows if row["snapshot_path"] == first_path])
    last_layers = index_layer_rows([row for row in layer_rows if row["snapshot_path"] == last_path])

    shared_names = sorted({name for _, name in first_layers.keys()} & {name for _, name in last_layers.keys()})
    rows: list[dict[str, object]] = []
    for name in shared_names:
        start = first_layers[(first_path, name)]
        end = last_layers[(last_path, name)]
        slope_delta = maybe_float(end["powerlaw_slope"]) - maybe_float(start["powerlaw_slope"])
        stable_frac_delta = maybe_float(end["stable_rank_fraction"]) - maybe_float(start["stable_rank_fraction"])
        eff_frac_delta = maybe_float(end["effective_rank_fraction"]) - maybe_float(start["effective_rank_fraction"])
        spectral_delta = maybe_float(end["spectral_norm"]) - maybe_float(start["spectral_norm"])
        rows.append(
            {
                "parameter_name": name,
                "slope_delta": slope_delta,
                "stable_rank_fraction_delta": stable_frac_delta,
                "effective_rank_fraction_delta": eff_frac_delta,
                "spectral_norm_delta": spectral_delta,
                "abs_slope_delta": abs(slope_delta) if math.isfinite(slope_delta) else -1.0,
            }
        )

    rows.sort(key=lambda row: row["abs_slope_delta"], reverse=True)
    return rows[:top_n]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report(
    output_path: Path,
    combined_summary_rows: list[dict[str, str]],
    trend_rows: list[dict[str, object]],
    layer_changes_by_run: dict[str, list[dict[str, object]]],
) -> None:
    lines: list[str] = []
    lines.append("# Snapshot SVD Comparison Report")
    lines.append("")
    lines.append("This report compares normalized singular-value spectra across tuning snapshots.")
    lines.append("More negative power-law slope means a steeper spectrum. Higher power-law R^2 means the normalized spectrum is better approximated by a log-log line.")
    lines.append("")

    trend_table = []
    for row in trend_rows:
        trend_table.append(
            [
                str(row["run_name"]),
                str(row["snapshot_count"]),
                str(row["start_snapshot_kimg"]),
                str(row["end_snapshot_kimg"]),
                format_num(float(row["weighted_mean_powerlaw_slope_start"])),
                format_num(float(row["weighted_mean_powerlaw_slope_end"])),
                format_signed(float(row["weighted_mean_powerlaw_slope_delta"])),
                format_num(float(row["weighted_mean_powerlaw_r2_start"])),
                format_num(float(row["weighted_mean_powerlaw_r2_end"])),
                format_signed(float(row["weighted_mean_powerlaw_r2_delta"])),
                format_num(float(row["weighted_mean_effective_rank_fraction_start"])),
                format_num(float(row["weighted_mean_effective_rank_fraction_end"])),
                format_signed(float(row["weighted_mean_effective_rank_fraction_delta"])),
            ]
        )
    lines.append("## Run-Level Summary")
    lines.append("")
    lines.append(
        markdown_table(
            [
                "Run",
                "N",
                "Start kimg",
                "End kimg",
                "Slope start",
                "Slope end",
                "Slope delta",
                "R2 start",
                "R2 end",
                "R2 delta",
                "Eff-rank frac start",
                "Eff-rank frac end",
                "Eff-rank frac delta",
            ],
            trend_table,
        )
    )
    lines.append("")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in combined_summary_rows:
        grouped[row["run_name"]].append(row)

    for run_name in sorted(grouped):
        ordered = sort_rows_by_kimg(grouped[run_name])
        lines.append(f"## {run_name}")
        lines.append("")
        lines.append("Snapshot-by-snapshot metrics:")
        lines.append("")
        per_snapshot = []
        for row in ordered:
            per_snapshot.append(
                [
                    str(row["snapshot_kimg"]),
                    format_num(maybe_float(row["weighted_mean_powerlaw_slope"])),
                    format_num(maybe_float(row["weighted_mean_powerlaw_r2"])),
                    format_num(maybe_float(row["weighted_mean_stable_rank_fraction"])),
                    format_num(maybe_float(row["weighted_mean_effective_rank_fraction"])),
                    format_num(maybe_float(row["max_spectral_norm"])),
                ]
            )
        lines.append(
            markdown_table(
                [
                    "kimg",
                    "Weighted slope",
                    "Weighted R2",
                    "Stable-rank frac",
                    "Eff-rank frac",
                    "Max spectral norm",
                ],
                per_snapshot,
            )
        )
        lines.append("")
        lines.append("Largest layer-level slope changes from first to last snapshot:")
        lines.append("")
        top_layers = []
        for row in layer_changes_by_run[run_name]:
            top_layers.append(
                [
                    str(row["parameter_name"]),
                    format_signed(float(row["slope_delta"])),
                    format_signed(float(row["stable_rank_fraction_delta"])),
                    format_signed(float(row["effective_rank_fraction_delta"])),
                    format_signed(float(row["spectral_norm_delta"])),
                ]
            )
        lines.append(
            markdown_table(
                [
                    "Layer",
                    "Slope delta",
                    "Stable-rank frac delta",
                    "Eff-rank frac delta",
                    "Spectral norm delta",
                ],
                top_layers,
            )
        )
        lines.append("")

    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined_summary_rows: list[dict[str, str]] = []
    trend_rows: list[dict[str, object]] = []
    layer_changes_by_run: dict[str, list[dict[str, object]]] = {}

    for analysis_dir in args.analysis_dirs:
        summary_path = analysis_dir / "snapshot_summary.csv"
        layer_path = analysis_dir / "layer_stats.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
        if not layer_path.exists():
            raise FileNotFoundError(f"Missing layer CSV: {layer_path}")

        summary_rows = read_csv(summary_path)
        layer_rows = read_csv(layer_path)
        if not summary_rows:
            raise RuntimeError(f"No summary rows in: {summary_path}")

        combined_summary_rows.extend(summary_rows)
        trend_rows.append(make_run_trend_row(summary_rows))
        run_name = summary_rows[0]["run_name"]
        layer_changes_by_run[run_name] = compute_top_layer_changes(summary_rows, layer_rows, args.top_layers)

    combined_summary_rows = sort_rows_by_kimg(combined_summary_rows)
    trend_rows = sorted(trend_rows, key=lambda row: str(row["run_name"]))

    write_csv(args.output_dir / "combined_snapshot_summary.csv", combined_summary_rows)
    write_csv(args.output_dir / "run_trends.csv", trend_rows)
    write_report(args.output_dir / "comparison_report.md", combined_summary_rows, trend_rows, layer_changes_by_run)

    print(f"Wrote combined summary to: {args.output_dir / 'combined_snapshot_summary.csv'}")
    print(f"Wrote run trends to: {args.output_dir / 'run_trends.csv'}")
    print(f"Wrote markdown report to: {args.output_dir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
