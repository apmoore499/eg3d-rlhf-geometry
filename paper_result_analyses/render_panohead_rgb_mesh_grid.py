"""Produce a 2x3 grid for one or more PanoHead seeds:

  row 1: RGB renders at three yaws — left, canonical, right
  row 2: marching-cubes mesh renders at the same three angles

Run from anywhere in the env hf_geom_eg3d_py39. Output JPG per seed is
saved to:
  PanoHead/panohead_sigma_cubes_for_reward/trunc{psi}/rgb_mesh_grid/
    grid_seed_{seed}.jpg

σ is sampled twice per seed: in-memory only for the mesh row at
`--mesh-shape-res` (default 512, for sharper iso-surface detail), and
fully decoupled from the reward-model σ cubes on disk which stay at
shape_res=256. The in-memory mesh σ tensors are not persisted to disk.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, Tuple

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

PANOHEAD_REPO = panohead_root()
if str(PANOHEAD_REPO) not in sys.path:
    sys.path.insert(0, str(PANOHEAD_REPO))

import dnnlib  # type: ignore  # noqa: E402
import legacy  # type: ignore  # noqa: E402
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics  # type: ignore  # noqa: E402

DEFAULT_PKL = PANOHEAD_REPO / "models" / "easy-khair-180-gpc0.8-trans10-025000.pkl"
SIGMA_ROOT = PANOHEAD_REPO / "panohead_sigma_cubes_for_reward"
NEURAL_RES = 128
TILE_RES = 512
LABEL_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Yaw offsets (radians) from canonical (π/2). Negative = camera moves left of
# the subject and we see the right side of the face; positive = camera moves
# right of subject (we see the left side of the face).
YAW_OFFSETS_RAD: Tuple[float, float, float] = (-0.4, 0.0, +0.4)
LABELS: Tuple[str, str, str] = ("left view", "canonical", "right view")

# Mesh rotation angles (degrees, around Y) chosen so the rendered mesh matches
# the RGB camera offset. Canonical (frontal-facing) is -90° per
# render_panohead_meshes.py; a positive (negative) yaw offset of the camera
# corresponds to the mesh appearing rotated in the opposite direction.
MESH_ANGLES_DEG: Tuple[float, float, float] = (
    -90.0 + 23.0,   # left view (camera moved left, mesh appears rotated +23° around Y)
    -90.0,          # canonical
    -90.0 - 23.0,   # right view
)
MARCHING_CUBES_LEVEL = 10.0
PAD_RATIO = 30 / 256  # gen_meshes.py: pad = int(30 * shape_res / 256)


def create_samples(N: int, voxel_origin=(0, 0, 0), cube_length: float = 1.0):
    """Verbatim copy of PanoHead/gen_samples.py:create_samples — produces the
    world-coord grid for σ sampling at N³ resolution."""
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


def sample_sigma_inmemory(
    G, seed: int, conditioning_params: torch.Tensor,
    shape_res: int, truncation_psi: float, truncation_cutoff: int,
    box_warp: float, device: torch.device, max_batch: int = 1_000_000,
) -> np.ndarray:
    """Sample PanoHead σ at shape_res³ in memory only — does NOT touch disk.
    Returned as a NumPy array of shape (shape_res, shape_res, shape_res)."""
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
            samples[:, head:end],
            rays[:, : end - head],
            z, conditioning_params,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            noise_mode="const",
        )["sigma"]
        sigmas[:, head:end] = out
        head = end
    cube = sigmas.reshape(shape_res, shape_res, shape_res).cpu().numpy()
    # Free GPU memory immediately — these are large at 512³.
    del sigmas, samples, rays, z
    torch.cuda.empty_cache()
    return cube


def _load_generator(pkl: Path, device: torch.device):
    with dnnlib.util.open_url(str(pkl)) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)
    return G


def render_rgb_grid(
    G, seed: int, yaw_offsets: Sequence[float], truncation_psi: float,
    truncation_cutoff: int, fov_deg: float, device: torch.device,
) -> Sequence[np.ndarray]:
    cam_pivot = torch.tensor(
        G.rendering_kwargs.get("avg_camera_pivot", [0, 0, 0]),
        device=device, dtype=torch.float32,
    )
    cam_radius = float(G.rendering_kwargs.get("avg_camera_radius", 2.7))
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cond_pose = LookAtPoseSampler.sample(
        np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=device,
    )
    conditioning_params = torch.cat(
        [cond_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1,
    )
    z = torch.from_numpy(
        np.random.RandomState(int(seed)).randn(1, G.z_dim).astype(np.float32),
    ).to(device)
    ws = G.mapping(
        z, conditioning_params,
        truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
    )
    rgbs = []
    for yaw_off in yaw_offsets:
        cam_pose = LookAtPoseSampler.sample(
            np.pi / 2 + yaw_off, np.pi / 2, cam_pivot,
            radius=cam_radius, device=device,
        )
        camera_params = torch.cat(
            [cam_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1,
        )
        with torch.no_grad():
            out = G.synthesis(
                ws, camera_params,
                neural_rendering_resolution=NEURAL_RES, noise_mode="const",
            )
        img = out["image"][0].permute(1, 2, 0).clamp(-1, 1)
        img = (img * 127.5 + 128).to(torch.uint8).cpu().numpy()
        # Resize from G.img_resolution to TILE_RES for consistent grid sizing.
        img = np.asarray(
            Image.fromarray(img).resize((TILE_RES, TILE_RES), Image.LANCZOS)
        )
        rgbs.append(img)
    return rgbs


def render_mesh_grid(sigma_cube: np.ndarray,
                     angles_deg: Sequence[float]) -> Sequence[np.ndarray]:
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

    tiles = []
    for angle_deg in angles_deg:
        m = mesh.copy()
        rot = trimesh.transformations.rotation_matrix(
            np.radians(angle_deg), [0, 1, 0],
        )
        m.apply_transform(rot)
        mesh_pr = pyrender.Mesh.from_trimesh(m, smooth=False)
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
        r = pyrender.OffscreenRenderer(TILE_RES, TILE_RES)
        color, _ = r.render(scene)
        r.delete()
        tiles.append(color)
    return tiles


def assemble_grid(
    rgbs: Sequence[np.ndarray],
    meshes: Sequence[np.ndarray],
    title: str,
    labels: Sequence[str],
) -> Image.Image:
    pad = 8
    label_h = 28
    row_h = TILE_RES
    row_w = TILE_RES * 3 + pad * 2
    total_h = label_h + row_h + pad + row_h
    canvas = np.full((total_h, row_w, 3), 18, dtype=np.uint8)
    # Place RGB row
    for i, im in enumerate(rgbs):
        x0 = i * (TILE_RES + pad)
        canvas[label_h:label_h + row_h, x0:x0 + TILE_RES] = im
    # Place mesh row
    for i, im in enumerate(meshes):
        x0 = i * (TILE_RES + pad)
        canvas[label_h + row_h + pad:, x0:x0 + TILE_RES] = im
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    try:
        font_big = ImageFont.truetype(LABEL_FONT, 20)
        font_small = ImageFont.truetype(LABEL_FONT, 16)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()
    draw.text((10, 4), title, fill=(255, 255, 255), font=font_big)
    for i, lbl in enumerate(labels):
        x0 = i * (TILE_RES + pad) + 10
        draw.text((x0, label_h + 4), lbl, fill=(255, 255, 0), font=font_small)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=str(DEFAULT_PKL))
    ap.add_argument("--seeds", default="200050",
                    help="comma-separated list, or single seed")
    ap.add_argument("--truncation-psi", type=float, default=0.7)
    ap.add_argument("--truncation-cutoff", type=int, default=14)
    ap.add_argument("--fov-deg", type=float, default=18.837)
    ap.add_argument("--mesh-shape-res", type=int, default=512,
                    help="σ-cube resolution used only for marching cubes. "
                         "Sampled in memory each invocation; never written to "
                         "disk. The reward-model σ cubes on disk stay at 256.")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    psi = args.truncation_psi
    out_dir = SIGMA_ROOT / f"trunc{psi:.2f}" / "rgb_mesh_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    print(f"[grid] loading PanoHead from {args.pkl}")
    G = _load_generator(Path(args.pkl), device)
    # Canonical conditioning for σ sampling (same convention as
    # extract_sigmas_for_reward_transfer.py and the reward-model evaluation).
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
    conditioning_params_mesh = torch.cat(
        [cond_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1,
    )
    print(f"[grid] mesh σ at shape_res={args.mesh_shape_res} (in-memory only); "
          f"reward-model σ on disk stays at 256.")

    for seed in seeds:
        rgbs = render_rgb_grid(
            G, seed, YAW_OFFSETS_RAD,
            truncation_psi=psi,
            truncation_cutoff=args.truncation_cutoff,
            fov_deg=args.fov_deg, device=device,
        )
        cube = sample_sigma_inmemory(
            G, seed, conditioning_params_mesh,
            shape_res=args.mesh_shape_res,
            truncation_psi=psi, truncation_cutoff=args.truncation_cutoff,
            box_warp=box_warp, device=device,
        )
        meshes = render_mesh_grid(cube, MESH_ANGLES_DEG)
        # Free the σ cube as soon as the mesh tiles are rendered.
        del cube
        img = assemble_grid(
            rgbs, meshes,
            title=f"PanoHead seed {seed}  trunc_psi={psi}  mesh_res={args.mesh_shape_res}",
            labels=LABELS,
        )
        out_path = out_dir / f"grid_seed_{seed}.jpg"
        img.save(out_path, quality=92)
        print(f"[grid] saved {out_path}")


if __name__ == "__main__":
    main()
