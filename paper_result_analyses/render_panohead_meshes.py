"""Marching-cubes mesh visualisations for PanoHead σ cubes — 3-view trimesh
renders saved alongside the σ cubes. Produces JPGs for the canonical visual
sanity check that what PanoHead is generating *as 3D geometry* is a coherent
head, in contrast to its low score under the EG3D-trained σ reward
(Section §4.3.6).
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pyrender
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont
from skimage import measure

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_result_analyses.path_defaults import panohead_root  # noqa: E402

SEEDS = (200000, 200050, 200060)
TRUNCS = (0.70, 0.25, 0.00)
ROOT = panohead_root() / "panohead_sigma_cubes_for_reward"
MARCHING_CUBES_LEVEL = 10.0
PAD = 30  # match gen_meshes.py border-zero convention
# After np.transpose(sigmas, (2,1,0)) the face sits at high pyrender-X (right
# of image). Rotating by -90° around Y brings the face to face the camera
# (frontal). Subsequent angles are 3/4 and side-profile.
VIEW_ANGLES_DEG = (-90, -45, 0)
TILE_RES = 512


def render_three_views(verts: np.ndarray, faces: np.ndarray, title: str,
                       out_path: Path) -> None:
    """3-view pyrender render: rotate mesh about world-Y by VIEW_ANGLES_DEG,
    place an orthographic-ish perspective camera looking at -Z, lit with a
    single point light at the camera position. Output is a horizontal strip."""
    mesh = trimesh.Trimesh(vertices=verts.copy(), faces=faces.copy())
    mesh.fix_normals()
    mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
    scale = float(np.max(np.abs(mesh.vertices)))
    if scale > 0:
        mesh.vertices = mesh.vertices / scale

    tiles = []
    for angle_deg in VIEW_ANGLES_DEG:
        m = mesh.copy()
        rot = trimesh.transformations.rotation_matrix(np.radians(angle_deg), [0, 1, 0])
        m.apply_transform(rot)
        mesh_pr = pyrender.Mesh.from_trimesh(m, smooth=False)
        scene = pyrender.Scene(bg_color=[20, 20, 20, 255], ambient_light=[60, 60, 60])
        scene.add(mesh_pr)
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 4, aspectRatio=1.0)
        cam_pose = np.eye(4)
        cam_pose[:3, 3] = [0.0, 0.0, 2.4]
        scene.add(cam, pose=cam_pose)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0.5, 0.5, 2.4]
        scene.add(pyrender.PointLight(color=[255, 255, 255], intensity=12.0), pose=light_pose)
        r = pyrender.OffscreenRenderer(TILE_RES, TILE_RES)
        color, _ = r.render(scene)
        r.delete()
        tiles.append(color)
    strip = np.concatenate(tiles, axis=1)
    img = Image.fromarray(strip)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18,
        )
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 10), title, fill=(255, 255, 255), font=font)
    img.save(out_path, quality=92)
    print(f"saved {out_path}")


def main():
    for trunc in TRUNCS:
        out_dir = ROOT / f"trunc{trunc:.2f}" / "mesh_renders"
        out_dir.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            cube_pt = ROOT / f"trunc{trunc:.2f}" / f"sigma_seed_{seed}.pt"
            if not cube_pt.exists():
                print(f"missing {cube_pt}")
                continue
            sigmas = torch.load(cube_pt, map_location="cpu").numpy().astype(np.float32)
            sigmas[:PAD] = sigmas[-PAD:] = -1000
            sigmas[:, :PAD] = sigmas[:, -PAD:] = -1000
            sigmas[:, :, :PAD] = sigmas[:, :, -PAD:] = -1000
            verts, faces, _, _ = measure.marching_cubes(
                np.transpose(sigmas, (2, 1, 0)),
                level=MARCHING_CUBES_LEVEL, spacing=[1, 1, 1],
            )
            out_path = out_dir / f"mesh_seed_{seed}.jpg"
            render_three_views(
                verts, faces,
                title=f"PanoHead seed {seed} trunc_psi={trunc}  ({len(verts)} verts)",
                out_path=out_path,
            )


if __name__ == "__main__":
    main()
