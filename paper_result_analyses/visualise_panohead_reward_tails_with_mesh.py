"""Top-N / bottom-N PanoHead samples by σ_XYZ reward, with the canonical
RGB stacked above its canonical-frontal marching-cubes mesh.

σ is sampled in-memory at --mesh-shape-res (default 512) for each of the
2N highlighted seeds; nothing is written to disk. The reward-model σ
cubes at trunc{psi}/sigma_seed_*.pt stay at 256³ (they came from
extract_sigmas_for_reward_transfer.py).

Reads:
  reward_embedding_analysis/panohead_reward_transfer/
      panohead_trunc{psi}/per_seed_rewards.csv
  PanoHead/panohead_sigma_cubes_for_reward/trunc{psi}/rgb_canonical/
      rgb_seed_{seed}.jpg

Writes:
  PanoHead/panohead_sigma_cubes_for_reward/trunc{psi}/
      reward_tails_top_vs_bottom_with_mesh.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyrender
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont
from skimage import measure

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_result_analyses.path_defaults import (  # noqa: E402
    panohead_root,
    reward_embedding_analysis_dir,
)

PANOHEAD_REPO = panohead_root()
if str(PANOHEAD_REPO) not in sys.path:
    sys.path.insert(0, str(PANOHEAD_REPO))

import dnnlib  # type: ignore  # noqa: E402
import legacy  # type: ignore  # noqa: E402
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics  # type: ignore  # noqa: E402

DEFAULT_PKL = PANOHEAD_REPO / "models" / "easy-khair-180-gpc0.8-trans10-025000.pkl"
TILE_RES = 320
COLS = 5
SEP_H = 26
LABEL_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MARCHING_CUBES_LEVEL = 10.0
PAD_RATIO = 30 / 256
MESH_CANONICAL_ANGLE_DEG = -90.0  # frontal-facing per render_panohead_meshes.py


def _font(size: int):
    try:
        return ImageFont.truetype(LABEL_FONT, size)
    except Exception:
        return ImageFont.load_default()


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


def _load_generator(pkl: Path, device: torch.device):
    with dnnlib.util.open_url(str(pkl)) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)
    return G


def sample_sigma_inmemory(
    G, seed: int, conditioning_params: torch.Tensor,
    shape_res: int, truncation_psi: float, truncation_cutoff: int,
    box_warp: float, device: torch.device, max_batch: int = 1_000_000,
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
            z, conditioning_params,
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


def render_canonical_mesh(sigma_cube: np.ndarray, tile_res: int) -> np.ndarray:
    """Marching-cubes + single frontal view at MESH_CANONICAL_ANGLE_DEG."""
    sigmas = sigma_cube.copy().astype(np.float32)
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
    scene = pyrender.Scene(bg_color=[20, 20, 20, 255],
                           ambient_light=[60, 60, 60])
    scene.add(mesh_pr)
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4, aspectRatio=1.0)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = [0.0, 0.0, 2.4]
    scene.add(cam, pose=cam_pose)
    light_pose = np.eye(4)
    light_pose[:3, 3] = [0.5, 0.5, 2.4]
    scene.add(pyrender.PointLight(color=[255, 255, 255], intensity=12.0),
              pose=light_pose)
    r = pyrender.OffscreenRenderer(tile_res, tile_res)
    color, _ = r.render(scene)
    r.delete()
    return color


def _stack_tile(rgb_path: Path, mesh_arr: np.ndarray,
                seed: int, reward: float) -> np.ndarray:
    """One tile: RGB on top (TILE_RES x TILE_RES), mesh below (same), with a
    semi-transparent label band at the bottom of the RGB sub-tile."""
    rgb = Image.open(rgb_path).resize((TILE_RES, TILE_RES), Image.LANCZOS).convert("RGB")
    band = Image.new("RGBA", (TILE_RES, 44), (0, 0, 0, 180))
    rgba = rgb.convert("RGBA")
    rgba.paste(band, (0, TILE_RES - 44), band)
    draw = ImageDraw.Draw(rgba)
    draw.text((8, TILE_RES - 40), f"r = {reward:+.3f}",
              fill=(255, 255, 0), font=_font(24))
    draw.text((8, TILE_RES - 16), f"seed {seed}",
              fill=(220, 220, 220), font=_font(14))
    rgb_arr = np.asarray(rgba.convert("RGB"))
    return np.concatenate([rgb_arr, mesh_arr], axis=0)


def _group_strip(
    rows: pd.DataFrame, rgb_dir: Path, header: str,
    G, cond_params: torch.Tensor, mesh_shape_res: int,
    truncation_psi: float, truncation_cutoff: int,
    box_warp: float, device: torch.device,
) -> np.ndarray:
    n = len(rows)
    n_grid_rows = max(1, (n + COLS - 1) // COLS)
    tile_h = TILE_RES * 2
    strip_h = SEP_H + tile_h * n_grid_rows
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
        reward = float(r["reward_panohead"])
        rgb_path = rgb_dir / f"rgb_seed_{seed}.jpg"
        if not rgb_path.exists():
            print(f"  missing {rgb_path}, blank tile")
            tile = np.full((tile_h, TILE_RES, 3), 80, dtype=np.uint8)
        else:
            print(f"  [tile {i+1}/{n}] seed {seed}: sampling σ at "
                  f"{mesh_shape_res}³ + marching cubes")
            cube = sample_sigma_inmemory(
                G, seed, cond_params,
                shape_res=mesh_shape_res,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
                box_warp=box_warp, device=device,
            )
            mesh_img = render_canonical_mesh(cube, TILE_RES)
            del cube
            tile = _stack_tile(rgb_path, mesh_img, seed, reward)
        y0 = SEP_H + row * tile_h
        x0 = col * TILE_RES
        canvas[y0:y0 + tile_h, x0:x0 + TILE_RES] = tile
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truncation-psi", type=float, default=0.7)
    ap.add_argument("--n-each", type=int, default=10)
    ap.add_argument("--mesh-shape-res", type=int, default=512)
    ap.add_argument("--truncation-cutoff", type=int, default=14)
    ap.add_argument("--fov-deg", type=float, default=18.837)
    ap.add_argument("--pkl", default=str(DEFAULT_PKL))
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
        PANOHEAD_REPO
        / "panohead_sigma_cubes_for_reward"
        / f"trunc{trunc_str}"
        / "rgb_canonical"
    )
    out_path = (
        PANOHEAD_REPO
        / "panohead_sigma_cubes_for_reward"
        / f"trunc{trunc_str}"
        / "reward_tails_top_vs_bottom_with_mesh.jpg"
    )
    if not csv_path.exists():
        raise SystemExit(f"missing rewards CSV: {csv_path}")
    df = pd.read_csv(csv_path).dropna(subset=["reward_panohead"]).copy()
    df = df.sort_values("reward_panohead", ascending=False).reset_index(drop=True)
    n = args.n_each
    top = df.head(n)
    bot = df.tail(n).iloc[::-1]
    print(f"[tails+mesh] trunc={trunc_str}: top-{n} reward "
          f"[{top['reward_panohead'].min():+.3f}, {top['reward_panohead'].max():+.3f}], "
          f"bot-{n} [{bot['reward_panohead'].min():+.3f}, {bot['reward_panohead'].max():+.3f}]")

    device = torch.device("cuda")
    print(f"[tails+mesh] loading PanoHead from {args.pkl}")
    G = _load_generator(Path(args.pkl), device)
    cam_pivot = torch.tensor(
        G.rendering_kwargs.get("avg_camera_pivot", [0, 0, 0]),
        device=device, dtype=torch.float32,
    )
    cam_radius = float(G.rendering_kwargs.get("avg_camera_radius", 2.7))
    box_warp = float(G.rendering_kwargs.get("box_warp", 1.0))
    intrinsics = FOV_to_intrinsics(args.fov_deg, device=device)
    cond_pose = LookAtPoseSampler.sample(
        np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=device,
    )
    cond_params = torch.cat(
        [cond_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1,
    )

    print(f"[tails+mesh] rendering top-{n} (this can take a couple of minutes)")
    top_hdr = (f"Top {n}   (highest σ_XYZ reward in PanoHead, "
               f"trunc_psi={psi}, mesh_res={args.mesh_shape_res})")
    bot_hdr = (f"Bottom {n}   (lowest σ_XYZ reward in PanoHead, "
               f"trunc_psi={psi}, mesh_res={args.mesh_shape_res})")
    top_strip = _group_strip(
        top, rgb_dir, top_hdr, G, cond_params, args.mesh_shape_res,
        psi, args.truncation_cutoff, box_warp, device,
    )
    print(f"[tails+mesh] rendering bottom-{n}")
    bot_strip = _group_strip(
        bot, rgb_dir, bot_hdr, G, cond_params, args.mesh_shape_res,
        psi, args.truncation_cutoff, box_warp, device,
    )
    sep = np.full((20, top_strip.shape[1], 3), 0, dtype=np.uint8)
    final = np.concatenate([top_strip, sep, bot_strip], axis=0)
    Image.fromarray(final).save(out_path, quality=92)
    print(f"[tails+mesh] saved {out_path}  shape={final.shape}")


if __name__ == "__main__":
    main()
