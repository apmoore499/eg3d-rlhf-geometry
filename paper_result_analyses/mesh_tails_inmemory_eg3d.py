"""High-resolution top-N / bot-N σ-mesh tail renderer for EG3D
(pretrained or RLHF-tuned). Mirrors PanoHead/HyPlane/Sphere's
mesh_tails_inmemory.py but loads the generator through core_modules
gen_utils (the canonical EG3D conditioning path used everywhere else in
this codebase) instead of dnnlib + legacy directly. σ is sampled at
--mesh-shape-res (default 512) in memory per seed, never written to
disk.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyrender
import torch
import trimesh
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from skimage import measure
from tqdm.auto import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_result_analyses.path_defaults import EG3D_ROOT, RLHF_CORE_ROOT  # noqa: E402

for _p in (REPO_ROOT, EG3D_ROOT, RLHF_CORE_ROOT.parent):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)
try:
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
except Exception:
    pass
from core_modules.data.create_train_data import generation_utils as gen_utils  # noqa: E402

TILE_RES = 512
COLS = 5
SEP_H = 32
LABEL_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MARCHING_CUBES_LEVEL = 10.0
PAD_RATIO = 30 / 256
MESH_CANONICAL_ANGLE_DEG = -90.0


def create_samples(N: int, voxel_origin=(0, 0, 0), cube_length: float = 1.0):
    voxel_origin = np.array(voxel_origin) - cube_length / 2
    voxel_size = cube_length / (N - 1)
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    return samples.unsqueeze(0), voxel_origin, voxel_size


def _font(size: int):
    try:
        return ImageFont.truetype(LABEL_FONT, size)
    except Exception:
        return ImageFont.load_default()


def sample_sigma_inmemory(
    G, seed: int, tdca, shape_res: int, truncation_psi: float,
    truncation_cutoff: int, box_warp: float, device: torch.device,
    max_batch: int = 1_000_000,
) -> np.ndarray:
    samples, _, _ = create_samples(N=shape_res, voxel_origin=[0, 0, 0],
                                   cube_length=box_warp * 1.0)
    samples = samples.to(device)
    n_voxels = samples.shape[1]
    rays = torch.zeros((samples.shape[0], max_batch, 3), device=device)
    rays[..., -1] = -1.0
    z = torch.from_numpy(
        np.random.RandomState(int(seed)).randn(1, G.z_dim).astype(np.float32)
    ).to(device)
    sigmas = torch.zeros((1, n_voxels, 1), device=device, dtype=torch.float32)
    head = 0
    torch.manual_seed(0)
    while head < n_voxels:
        end = min(head + max_batch, n_voxels)
        out = G.sample(
            samples[:, head:end], rays[:, : end - head],
            z, tdca.conditioning_params,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            noise_mode="const",
        )["sigma"]
        sigmas[:, head:end] = out
        head = end
    cube = sigmas.reshape(shape_res, shape_res, shape_res).cpu().numpy()
    del sigmas, samples, rays, z
    torch.cuda.empty_cache()
    return cube


def render_mesh_tile(sigma_cube: np.ndarray, seed: int,
                     reward: float) -> np.ndarray:
    sigmas = sigma_cube.copy().astype(np.float32)
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
    mesh_pr = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene = pyrender.Scene(bg_color=[20, 20, 20, 255],
                           ambient_light=[80, 80, 80])
    scene.add(mesh_pr)
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4, aspectRatio=1.0)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = [0.0, 0.0, 2.4]
    scene.add(cam, pose=cam_pose)
    light_pose = np.eye(4)
    light_pose[:3, 3] = [0.5, 0.5, 2.4]
    scene.add(pyrender.PointLight(color=[255, 255, 255], intensity=12.0),
              pose=light_pose)
    r = pyrender.OffscreenRenderer(TILE_RES, TILE_RES)
    color, _ = r.render(scene)
    r.delete()
    return _draw_labels(color, seed, reward)


def _draw_labels(rgb: np.ndarray, seed: int, reward: float) -> np.ndarray:
    img = Image.fromarray(rgb).convert("RGBA")
    band = Image.new("RGBA", (TILE_RES, 56), (0, 0, 0, 180))
    img.paste(band, (0, TILE_RES - 56), band)
    draw = ImageDraw.Draw(img)
    draw.text((10, TILE_RES - 52), f"r = {reward:+.3f}",
              fill=(255, 255, 0), font=_font(26))
    draw.text((10, TILE_RES - 22), f"seed {seed}",
              fill=(220, 220, 220), font=_font(18))
    return np.asarray(img.convert("RGB"))


def _blank_tile(seed: int, reward: float, note: str = "") -> np.ndarray:
    canvas = np.full((TILE_RES, TILE_RES, 3), 80, dtype=np.uint8)
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), note, fill=(255, 200, 200), font=_font(18))
    return _draw_labels(np.asarray(img), seed, reward)


def _row_of_tiles(
    G, tdca, rows: pd.DataFrame, header: str, mesh_shape_res: int,
    truncation_psi: float, truncation_cutoff: int,
    box_warp: float, device: torch.device,
) -> np.ndarray:
    strip_h = SEP_H + TILE_RES
    strip_w = TILE_RES * COLS
    canvas = np.full((strip_h, strip_w, 3), 12, dtype=np.uint8)
    band = Image.fromarray(canvas[:SEP_H].copy())
    draw = ImageDraw.Draw(band)
    draw.text((12, 5), header, fill=(255, 255, 255), font=_font(22))
    canvas[:SEP_H] = np.asarray(band)
    for i, (_, r) in enumerate(tqdm(list(rows.iterrows()), desc=header)):
        if i >= COLS:
            break
        seed = int(r["seed"])
        reward = float(r["reward"])
        cube = sample_sigma_inmemory(
            G, seed, tdca, shape_res=mesh_shape_res,
            truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
            box_warp=box_warp, device=device,
        )
        tile = render_mesh_tile(cube, seed, reward)
        del cube
        x0 = i * TILE_RES
        canvas[SEP_H:SEP_H + TILE_RES, x0:x0 + TILE_RES] = tile
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--reward-csv", required=True)
    ap.add_argument("--reward-col", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-each", type=int, default=5)
    ap.add_argument("--mesh-shape-res", type=int, default=512)
    ap.add_argument("--truncation-psi", type=float, default=0.7)
    ap.add_argument("--truncation-cutoff", type=int, default=14)
    args = ap.parse_args()

    df = pd.read_csv(args.reward_csv).dropna(subset=[args.reward_col]).copy()
    df = df.rename(columns={args.reward_col: "reward"}).sort_values(
        "reward", ascending=False).reset_index(drop=True)
    n = args.n_each
    top = df.head(n).reset_index(drop=True)
    bot = df.tail(n).iloc[::-1].reset_index(drop=True)
    print(f"[mesh-tails-eg3d:{args.label}] selected from {len(df)} seeds")
    print(f"  top-{n} reward [{top['reward'].min():+.3f}, "
          f"{top['reward'].max():+.3f}]")
    print(f"  bot-{n} reward [{bot['reward'].min():+.3f}, "
          f"{bot['reward'].max():+.3f}]")

    device = torch.device("cuda")
    print(f"[mesh-tails-eg3d:{args.label}] loading G via core_modules gen_utils "
          f"from {args.pkl}")
    da = gen_utils.load_generator(
        model_path=Path(args.pkl), truncation_psi=args.truncation_psi,
        truncation_cutoff=args.truncation_cutoff, shape_res=256,
    )
    G = da.G
    tdca = gen_utils.get_single_dmap_cam(da)
    box_warp = float(G.rendering_kwargs.get("box_warp", 1.0))
    print(f"[mesh-tails-eg3d:{args.label}] σ in-memory at "
          f"{args.mesh_shape_res}³, box_warp={box_warp}")

    top_hdr = (f"Top {n}  σ mesh (σ@{args.mesh_shape_res}³, "
               f"highest reward, {args.label})")
    bot_hdr = (f"Bottom {n}  σ mesh (σ@{args.mesh_shape_res}³, "
               f"lowest reward, {args.label})")
    top_strip = _row_of_tiles(
        G, tdca, top, top_hdr, args.mesh_shape_res,
        args.truncation_psi, args.truncation_cutoff, box_warp, device,
    )
    bot_strip = _row_of_tiles(
        G, tdca, bot, bot_hdr, args.mesh_shape_res,
        args.truncation_psi, args.truncation_cutoff, box_warp, device,
    )
    sep = np.full((24, top_strip.shape[1], 3), 0, dtype=np.uint8)
    final = np.concatenate([top_strip, sep, bot_strip], axis=0)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final).save(out_path, quality=92)
    print(f"[mesh-tails-eg3d:{args.label}] saved {out_path}  "
          f"shape={final.shape}")


if __name__ == "__main__":
    main()
