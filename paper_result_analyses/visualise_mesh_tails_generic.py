"""Top-N / bottom-N mesh tails for any generator with pre-extracted σ-cubes
on disk. Loads σ at 256³, marching-cubes, renders canonical-frontal view,
labels with reward + seed, assembles a 4-row grid (2 rows top, 2 rows
bottom of 5 tiles each = 20 mesh renders).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyrender
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont
from skimage import measure

TILE_RES = 320
COLS = 5
SEP_H = 26
LABEL_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MARCHING_CUBES_LEVEL = 10.0
PAD_RATIO = 30 / 256
MESH_CANONICAL_ANGLE_DEG = -90.0


def _font(size: int):
    try:
        return ImageFont.truetype(LABEL_FONT, size)
    except Exception:
        return ImageFont.load_default()


def render_mesh_tile(sigma_path: Path, seed: int, reward: float) -> np.ndarray:
    sigmas = torch.load(sigma_path, map_location="cpu").numpy().astype(np.float32)
    pad = max(1, int(round(PAD_RATIO * sigmas.shape[0])))
    sigmas[:pad] = sigmas[-pad:] = -1000
    sigmas[:, :pad] = sigmas[:, -pad:] = -1000
    sigmas[:, :, :pad] = sigmas[:, :, -pad:] = -1000
    try:
        verts, faces, _, _ = measure.marching_cubes(
            np.transpose(sigmas, (2, 1, 0)),
            level=MARCHING_CUBES_LEVEL, spacing=[1, 1, 1],
        )
    except Exception as e:
        print(f"  marching_cubes failed for seed {seed}: {e}; blank tile")
        return _blank_tile(seed, reward, note="(no surface at level=10)")
    mesh = trimesh.Trimesh(vertices=verts.copy(), faces=faces.copy())
    mesh.fix_normals()
    mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
    s = float(np.max(np.abs(mesh.vertices)))
    if s > 0:
        mesh.vertices = mesh.vertices / s
    rot = trimesh.transformations.rotation_matrix(
        np.radians(MESH_CANONICAL_ANGLE_DEG), [0, 1, 0],
    )
    mesh.apply_transform(rot)
    mesh_pr = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(bg_color=[20, 20, 20, 255], ambient_light=[80, 80, 80])
    scene.add(mesh_pr)
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4, aspectRatio=1.0)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = [0.0, 0.0, 2.4]
    scene.add(cam, pose=cam_pose)
    light_pose = np.eye(4)
    light_pose[:3, 3] = [0.5, 0.5, 2.4]
    scene.add(pyrender.PointLight(color=[255, 255, 255], intensity=10.0),
              pose=light_pose)
    r = pyrender.OffscreenRenderer(TILE_RES, TILE_RES)
    color, _ = r.render(scene)
    r.delete()
    return _draw_labels(color, seed, reward)


def _draw_labels(rgb: np.ndarray, seed: int, reward: float) -> np.ndarray:
    img = Image.fromarray(rgb).convert("RGBA")
    band = Image.new("RGBA", (TILE_RES, 44), (0, 0, 0, 180))
    img.paste(band, (0, TILE_RES - 44), band)
    draw = ImageDraw.Draw(img)
    draw.text((8, TILE_RES - 40), f"r = {reward:+.3f}",
              fill=(255, 255, 0), font=_font(20))
    draw.text((8, TILE_RES - 16), f"seed {seed}",
              fill=(220, 220, 220), font=_font(14))
    return np.asarray(img.convert("RGB"))


def _blank_tile(seed: int, reward: float, note: str = "") -> np.ndarray:
    canvas = np.full((TILE_RES, TILE_RES, 3), 80, dtype=np.uint8)
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), note, fill=(255, 200, 200), font=_font(14))
    return _draw_labels(np.asarray(img), seed, reward)


def _group_strip(rows: pd.DataFrame, sigma_dir: Path, header: str) -> np.ndarray:
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
        sigma_path = sigma_dir / f"sigma_seed_{seed}.pt"
        if not sigma_path.exists():
            tile = _blank_tile(seed, reward, note="(σ cube missing)")
        else:
            print(f"  [tile {i+1}/{n}] {sigma_dir.name} seed {seed}")
            tile = render_mesh_tile(sigma_path, seed, reward)
        y0 = SEP_H + row * TILE_RES
        x0 = col * TILE_RES
        canvas[y0:y0 + TILE_RES, x0:x0 + TILE_RES] = tile
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-dir", required=True,
                    help="dir containing sigma_seed_{seed}.pt files (any "
                         "shape; passed to marching_cubes as-is)")
    ap.add_argument("--reward-csv", required=True)
    ap.add_argument("--reward-col", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-each", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.reward_csv).dropna(subset=[args.reward_col]).copy()
    df = df.rename(columns={args.reward_col: "reward"}).sort_values(
        "reward", ascending=False).reset_index(drop=True)
    n = args.n_each
    top = df.head(n)
    bot = df.tail(n).iloc[::-1]
    print(f"[mesh-tails:{args.label}] top-{n} reward "
          f"[{top['reward'].min():+.3f}, {top['reward'].max():+.3f}], "
          f"bot-{n} [{bot['reward'].min():+.3f}, {bot['reward'].max():+.3f}]")

    sigma_dir = Path(args.sigma_dir)
    top_hdr = f"Top {n}  (σ mesh — highest reward, {args.label})"
    bot_hdr = f"Bottom {n}  (σ mesh — lowest reward, {args.label})"
    top_strip = _group_strip(top, sigma_dir, top_hdr)
    bot_strip = _group_strip(bot, sigma_dir, bot_hdr)
    sep = np.full((20, top_strip.shape[1], 3), 0, dtype=np.uint8)
    final = np.concatenate([top_strip, sep, bot_strip], axis=0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final).save(args.out, quality=92)
    print(f"[mesh-tails:{args.label}] saved {args.out}  shape={final.shape}")


if __name__ == "__main__":
    main()
