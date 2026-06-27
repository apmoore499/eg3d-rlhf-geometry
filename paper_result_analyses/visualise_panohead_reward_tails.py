"""Make a top-N / bottom-N PanoHead RGB tile, with σ_XYZ reward scores
labelled per sample, to inspect whether the reward-rank ordering is
visually meaningful within PanoHead's own distribution.

Reads:
  reward_embedding_analysis/panohead_reward_transfer/
      panohead_trunc{psi}/per_seed_rewards.csv
  PanoHead/panohead_sigma_cubes_for_reward/trunc{psi}/rgb_canonical/
      rgb_seed_{seed}.jpg

Writes (one image per truncation level):
  PanoHead/panohead_sigma_cubes_for_reward/trunc{psi}/
      reward_tails_top_vs_bottom.jpg

Layout: top-N samples sorted by descending σ_XYZ reward (left = highest,
right = lowest of the top-N), then a wide black separator, then bottom-N
samples (left = lowest, right = highest of the bottom-N). 5 tiles per row,
two rows per group → up to 10 tiles per group. Each tile shows the
canonical RGB render with the reward score and seed overlaid.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_result_analyses.path_defaults import (  # noqa: E402
    panohead_root,
    reward_embedding_analysis_dir,
)

TILE_RES = 320
COLS = 5
SEP_H = 26
LABEL_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(LABEL_FONT, size)
    except Exception:
        return ImageFont.load_default()


def _tile(rgb_path: Path, seed: int, reward: float) -> np.ndarray:
    """Resize one RGB to TILE_RES and overlay the reward score + seed."""
    img = Image.open(rgb_path).resize((TILE_RES, TILE_RES), Image.LANCZOS).convert("RGB")
    draw = ImageDraw.Draw(img)
    big = _font(24)
    small = _font(14)
    # semi-transparent bar at bottom for legibility
    band = Image.new("RGBA", (TILE_RES, 44), (0, 0, 0, 180))
    img_rgba = img.convert("RGBA")
    img_rgba.paste(band, (0, TILE_RES - 44), band)
    draw = ImageDraw.Draw(img_rgba)
    draw.text((8, TILE_RES - 40), f"r = {reward:+.3f}",
              fill=(255, 255, 0), font=big)
    draw.text((8, TILE_RES - 16), f"seed {seed}",
              fill=(220, 220, 220), font=small)
    return np.asarray(img_rgba.convert("RGB"))


def _group_strip(rows: pd.DataFrame, rgb_dir: Path, header: str) -> np.ndarray:
    """Build a (header + 2 × COLS) strip for one tail group (top or bottom)."""
    n = len(rows)
    n_rows = max(1, (n + COLS - 1) // COLS)
    row_h = TILE_RES
    strip_h = SEP_H + row_h * n_rows
    strip_w = TILE_RES * COLS
    canvas = np.full((strip_h, strip_w, 3), 12, dtype=np.uint8)
    # write header band
    band = Image.fromarray(canvas[:SEP_H].copy())
    draw = ImageDraw.Draw(band)
    draw.text((10, 3), header, fill=(255, 255, 255), font=_font(18))
    canvas[:SEP_H] = np.asarray(band)
    for i, (_, r) in enumerate(rows.iterrows()):
        row = i // COLS
        col = i % COLS
        seed = int(r["seed"])
        reward = float(r["reward_panohead"])
        rgb_path = rgb_dir / f"rgb_seed_{seed}.jpg"
        if not rgb_path.exists():
            print(f"  missing {rgb_path}, drawing blank tile")
            tile = np.full((TILE_RES, TILE_RES, 3), 80, dtype=np.uint8)
        else:
            tile = _tile(rgb_path, seed, reward)
        y0 = SEP_H + row * row_h
        x0 = col * TILE_RES
        canvas[y0:y0 + TILE_RES, x0:x0 + TILE_RES] = tile
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truncation-psi", type=float, default=0.7)
    ap.add_argument("--n-each", type=int, default=10,
                    help="how many seeds from top and bottom of distribution")
    args = ap.parse_args()
    psi = args.truncation_psi
    trunc_str = f"{psi:.2f}"

    csv_path = (
        reward_embedding_analysis_dir()
        / "panohead_reward_transfer"
        / f"panohead_trunc{trunc_str}"
        / "per_seed_rewards.csv"
    )
    rgb_dir = (
        panohead_root()
        / "panohead_sigma_cubes_for_reward"
        / f"trunc{trunc_str}"
        / "rgb_canonical"
    )
    out_path = (
        panohead_root()
        / "panohead_sigma_cubes_for_reward"
        / f"trunc{trunc_str}"
        / "reward_tails_top_vs_bottom.jpg"
    )
    if not csv_path.exists():
        raise SystemExit(f"missing rewards CSV at {csv_path}")
    df = pd.read_csv(csv_path).dropna(subset=["reward_panohead"]).copy()
    df = df.sort_values("reward_panohead", ascending=False).reset_index(drop=True)
    n = args.n_each
    top = df.head(n)
    bot = df.tail(n).iloc[::-1]  # lowest first when read left-to-right
    print(f"[tails] trunc={trunc_str}: top-{n} reward range "
          f"[{top['reward_panohead'].min():+.3f}, {top['reward_panohead'].max():+.3f}], "
          f"bot-{n} range [{bot['reward_panohead'].min():+.3f}, {bot['reward_panohead'].max():+.3f}]")

    top_hdr = (f"Top {n}  (highest σ_XYZ reward in PanoHead distribution, "
               f"trunc_psi={psi})")
    bot_hdr = (f"Bottom {n}  (lowest σ_XYZ reward in PanoHead distribution, "
               f"trunc_psi={psi})")
    top_strip = _group_strip(top, rgb_dir, top_hdr)
    bot_strip = _group_strip(bot, rgb_dir, bot_hdr)
    big_sep = np.full((20, top_strip.shape[1], 3), 0, dtype=np.uint8)
    final = np.concatenate([top_strip, big_sep, bot_strip], axis=0)
    Image.fromarray(final).save(out_path, quality=92)
    print(f"[tails] saved {out_path}  shape={final.shape}")


if __name__ == "__main__":
    main()
