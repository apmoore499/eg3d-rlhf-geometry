"""Render a single comparison strip showing one representative
marching-cubes mesh per 360° full-head generator (PanoHead, HyPlaneHead,
SphereHead) at canonical-frontal view, side-by-side.

σ-cubes are already on disk; this script does not touch the
generators. Output: a 3-tile horizontal strip JPG in the paper dir.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import pyrender
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont
from skimage import measure

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_result_analyses.path_defaults import (  # noqa: E402
    generated_figure_dir,
    hyplanehead_root,
    panohead_root,
    spherehead_root,
)

# (label, σ-cube dir, seed). The seed is the top-reward sample per generator.
TILES: List[Tuple[str, Path, int]] = [
    ("PanoHead", panohead_root() / "panohead_sigma_cubes_for_reward" / "trunc0.70", 200091),
    ("HyPlaneHead", hyplanehead_root() / "hyplanehead_sigma_cubes_for_reward" / "trunc0.70", 200060),
    ("SphereHead", spherehead_root() / "spherehead_sigma_cubes_for_reward" / "trunc0.70", 200060),
]
TILE_RES = 512
PAD_RATIO = 30 / 256
MARCHING_CUBES_LEVEL = 10.0
MESH_CANONICAL_ANGLE_DEG = -90.0
OUT = generated_figure_dir() / "fig_multimodel_mesh_comparison.jpg"


def render_one(sigma_dir: Path, seed: int, title: str) -> np.ndarray:
    pt = sigma_dir / f"sigma_seed_{seed}.pt"
    if not pt.exists():
        # fall back to any seed in the dir
        alt = sorted(sigma_dir.glob("sigma_seed_*.pt"))
        if not alt:
            raise SystemExit(f"no σ cubes in {sigma_dir}")
        pt = alt[0]
        seed = int(pt.stem.replace("sigma_seed_", ""))
    sigmas = torch.load(pt, map_location="cpu").numpy().astype(np.float32)
    pad = max(1, int(round(PAD_RATIO * sigmas.shape[0])))
    sigmas[:pad] = sigmas[-pad:] = -1000
    sigmas[:, :pad] = sigmas[:, -pad:] = -1000
    sigmas[:, :, :pad] = sigmas[:, :, -pad:] = -1000
    verts, faces, _, _ = measure.marching_cubes(
        np.transpose(sigmas, (2, 1, 0)),
        level=MARCHING_CUBES_LEVEL, spacing=[1, 1, 1],
    )
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
    img = Image.fromarray(color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22
        )
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, TILE_RES, 40), fill=(0, 0, 0))
    draw.text((10, 6), f"{title}  seed {seed}  ({len(verts)} verts)",
              fill=(255, 255, 255), font=font)
    return np.asarray(img)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tiles = []
    for label, sigma_dir, seed in TILES:
        print(f"[multimodel] rendering {label} seed {seed}")
        tiles.append(render_one(sigma_dir, seed, label))
    strip = np.concatenate(tiles, axis=1)
    Image.fromarray(strip).save(OUT, quality=92)
    print(f"[multimodel] saved {OUT}")


if __name__ == "__main__":
    main()
