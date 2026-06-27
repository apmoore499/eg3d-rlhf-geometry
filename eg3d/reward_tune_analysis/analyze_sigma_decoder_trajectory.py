#!/usr/bin/env python3
"""Analyze trajectory changes in the EG3D density decoder.

Focuses on the sigma-specific path in EG3D/PanoHead triplane generators:
- `decoder.net.2.weight[0]` is the explicit sigma output row
- `decoder.net.2.bias[0]` is the sigma bias
- `decoder.net.0.weight` is the shared hidden layer feeding both sigma and RGB

This is intended as a targeted follow-up to `analyze_snapshot_svd.py` when the
question is specifically about whether geometry/density changes are concentrated
in the sigma readout or in shared upstream weights.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import autoroot  # noqa: F401
import torch

import legacy
from analyze_snapshot_svd import force_torch_load_map_location_cpu, snapshot_kimg, snapshot_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", nargs="+", help="Snapshot `.pkl` files ordered along a trajectory.")
    parser.add_argument("--module-key", default="G_ema", choices=["G", "G_ema"], help="Which generator object to inspect.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV/JSON outputs.")
    parser.add_argument(
        "--proxy-layer",
        default="backbone.synthesis.b256.torgb.weight",
        help="Optional upstream plane-producing weight used as a rough proxy for triplane changes.",
    )
    return parser.parse_args()


def load_generator(path: Path, module_key: str):
    with path.open("rb") as handle:
        with force_torch_load_map_location_cpu():
            data = legacy.load_network_pkl(handle)
    return data[module_key].eval()


def rel_change(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((b - a).norm() / a.norm().clamp_min(1e-12)).item()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()


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

    snapshot_paths = [Path(s).resolve() for s in args.snapshots]
    generators = [load_generator(path, args.module_key) for path in snapshot_paths]
    state_dicts = [g.state_dict() for g in generators]

    base_sd = state_dicts[0]
    if "decoder.net.2.weight" not in base_sd:
        raise KeyError("Expected `decoder.net.2.weight` in generator state_dict.")
    if "decoder.net.0.weight" not in base_sd:
        raise KeyError("Expected `decoder.net.0.weight` in generator state_dict.")

    sigma_row_base = base_sd["decoder.net.2.weight"][0].float()
    rgb_rows_base = base_sd["decoder.net.2.weight"][1:].float()
    sigma_bias_base = base_sd["decoder.net.2.bias"][0].float()
    shared_base = base_sd["decoder.net.0.weight"].float()
    proxy_base = base_sd.get(args.proxy_layer)
    if proxy_base is not None:
        proxy_base = proxy_base.float()

    rows: list[dict[str, object]] = []
    for path, sd in zip(snapshot_paths, state_dicts):
        final_weight = sd["decoder.net.2.weight"].float()
        final_bias = sd["decoder.net.2.bias"].float()
        shared = sd["decoder.net.0.weight"].float()

        sigma_row = final_weight[0]
        rgb_rows = final_weight[1:]
        row_rel_changes = ((final_weight - base_sd["decoder.net.2.weight"].float()).norm(dim=1) / base_sd["decoder.net.2.weight"].float().norm(dim=1).clamp_min(1e-12))
        sigma_rank = 1 + int((row_rel_changes > row_rel_changes[0]).sum().item())

        row = {
            "snapshot_path": str(path),
            "snapshot_label": snapshot_label(path),
            "snapshot_kimg": snapshot_kimg(path),
            "sigma_row_rel_change": rel_change(sigma_row_base, sigma_row),
            "sigma_row_cosine": cosine(sigma_row_base, sigma_row),
            "sigma_bias_delta": (final_bias[0] - sigma_bias_base).item(),
            "rgb_rows_mean_rel_change": row_rel_changes[1:].mean().item(),
            "rgb_rows_median_rel_change": row_rel_changes[1:].median().item(),
            "rgb_rows_mean_cosine": torch.nn.functional.cosine_similarity(rgb_rows_base, rgb_rows, dim=1).mean().item(),
            "sigma_row_rel_change_rank_among_outputs": sigma_rank,
            "output_count": int(final_weight.shape[0]),
            "shared_decoder_rel_change": rel_change(shared_base, shared),
            "shared_decoder_cosine": cosine(shared_base, shared),
        }
        if args.proxy_layer in sd:
            proxy = sd[args.proxy_layer].float()
            row["proxy_layer"] = args.proxy_layer
            row["proxy_layer_rel_change"] = rel_change(proxy_base, proxy) if proxy_base is not None else None
            row["proxy_layer_cosine"] = cosine(proxy_base, proxy) if proxy_base is not None else None
        rows.append(row)

    write_csv(args.output_dir / "sigma_decoder_trajectory.csv", rows)
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "snapshots": [str(p) for p in snapshot_paths],
                "module_key": args.module_key,
                "proxy_layer": args.proxy_layer,
            },
            indent=2,
        )
    )
    print(f"Wrote sigma decoder trajectory to: {args.output_dir / 'sigma_decoder_trajectory.csv'}")


if __name__ == "__main__":
    main()
