"""Top-N / bottom-N reward-tail visualisation for any generator. RGB-only
tiles labelled with per-seed reward + seed id.

Reuses the layout from visualise_panohead_reward_tails.py but accepts any
RGB dir + reward CSV via CLI args.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

TILE_RES = 320
COLS = 5
SEP_H = 26
LABEL_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size: int):
    try:
        return ImageFont.truetype(LABEL_FONT, size)
    except Exception:
        return ImageFont.load_default()


def _tile(rgb_path: Path, seed: int, reward: float) -> np.ndarray:
    img = Image.open(rgb_path).resize((TILE_RES, TILE_RES), Image.LANCZOS).convert("RGB")
    band = Image.new("RGBA", (TILE_RES, 44), (0, 0, 0, 180))
    rgba = img.convert("RGBA")
    rgba.paste(band, (0, TILE_RES - 44), band)
    draw = ImageDraw.Draw(rgba)
    draw.text((8, TILE_RES - 40), f"r = {reward:+.3f}",
              fill=(255, 255, 0), font=_font(24))
    draw.text((8, TILE_RES - 16), f"seed {seed}",
              fill=(220, 220, 220), font=_font(14))
    return np.asarray(rgba.convert("RGB"))


def _group_strip(rows: pd.DataFrame, rgb_dir: Path, header: str) -> np.ndarray:
    n = len(rows)
    n_grid_rows = max(1, (n + COLS - 1) // COLS)
    strip_h = SEP_H + TILE_RES * n_grid_rows
    strip_w = TILE_RES * COLS
    canvas = np.full((strip_h, strip_w, 3), 12, dtype=np.uint8)
    band = Image.fromarray(canvas[:SEP_H].copy())
    draw = ImageDraw.Draw(band)
    draw.text((10, 3), header, fill=(255, 255, 255), font=_font(18))
    canvas[:SEP_H] = np.asarray(band)
    for i, (_, r) in enumerate(rows.iterrows()):
        row = i // COLS
        col = i % COLS
        seed = int(r["seed"])
        reward = float(r["reward"])
        # try a few naming conventions in case the dir uses different ones
        candidates = [
            rgb_dir / f"rgb_seed_{seed}.jpg",
            rgb_dir / f"rgb_seed_{seed}.png",
            rgb_dir / f"seed_{seed}.jpg",
        ]
        rgb_path = next((p for p in candidates if p.exists()), None)
        if rgb_path is None:
            print(f"  missing RGB for seed {seed} under {rgb_dir}")
            tile = np.full((TILE_RES, TILE_RES, 3), 80, dtype=np.uint8)
        else:
            tile = _tile(rgb_path, seed, reward)
        y0 = SEP_H + row * TILE_RES
        x0 = col * TILE_RES
        canvas[y0:y0 + TILE_RES, x0:x0 + TILE_RES] = tile
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-dir", required=True)
    ap.add_argument("--reward-csv", required=True)
    ap.add_argument("--reward-col", required=True,
                    help="reward column name in the CSV (e.g. reward_hyplanehead)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True, help="generator label for header")
    ap.add_argument("--n-each", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.reward_csv)
    if args.reward_col not in df.columns:
        # special-case: exp3 CSV uses 'reward_score' and has model_label
        if "reward_score" in df.columns and "model_label" in df.columns:
            raise SystemExit(
                f"{args.reward_csv} has model_label rows; pre-filter to "
                f"orig or tuned and write a single-row-per-seed CSV first."
            )
        raise SystemExit(f"missing {args.reward_col} in {args.reward_csv}; "
                         f"have {df.columns.tolist()}")
    df = df.dropna(subset=[args.reward_col]).copy()
    df = df.rename(columns={args.reward_col: "reward"}).sort_values(
        "reward", ascending=False).reset_index(drop=True)
    n = args.n_each
    top = df.head(n)
    bot = df.tail(n).iloc[::-1]

    print(f"[tails:{args.label}] top-{n} reward range "
          f"[{top['reward'].min():+.3f}, {top['reward'].max():+.3f}], "
          f"bot-{n} range [{bot['reward'].min():+.3f}, {bot['reward'].max():+.3f}]")

    rgb_dir = Path(args.rgb_dir)
    top_hdr = f"Top {n}   (highest σ_XYZ reward — {args.label})"
    bot_hdr = f"Bottom {n}   (lowest σ_XYZ reward — {args.label})"
    top_strip = _group_strip(top, rgb_dir, top_hdr)
    bot_strip = _group_strip(bot, rgb_dir, bot_hdr)
    sep = np.full((20, top_strip.shape[1], 3), 0, dtype=np.uint8)
    final = np.concatenate([top_strip, sep, bot_strip], axis=0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final).save(args.out, quality=92)
    print(f"[tails:{args.label}] saved {args.out}  shape={final.shape}")


if __name__ == "__main__":
    main()
