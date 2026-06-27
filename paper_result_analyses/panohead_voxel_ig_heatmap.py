"""Per-voxel IG heatmap on PanoHead marching-cubes mesh. For each selected
seed, compute IG (zero baseline → σ_cube), then for each mesh vertex sample
the IG at the nearest cropped-cube voxel (or 0 if outside the cropped
region). Colour the mesh by per-vertex IG and render a canonical-frontal
view per seed; stack into a 2×N strip (top row top-reward, bot row
bottom-reward).

σ for marching-cubes is at the full 256³ resolution from disk (sharper mesh)
and the IG slab is at the pads_vals_entire crop (matches what the reward
saw). For vertices inside the cropped slab, look up IG; outside, color
neutral grey.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyrender
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
from skimage import measure

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_result_analyses.path_defaults import (  # noqa: E402
    EG3D_ROOT,
    RLHF_CORE_ROOT,
    panohead_root,
    reward_embedding_analysis_dir,
)

RLHF_SRC_ROOT = RLHF_CORE_ROOT
for _p in (REPO_ROOT, EG3D_ROOT, RLHF_SRC_ROOT.parent):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

try:
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
except Exception:
    pass

import hydra  # noqa: E402

from core_modules.data.create_train_data import generation_utils as gen_utils  # noqa: E402
from core_modules.utils import finetuning_utils, reward_loading  # noqa: E402

PSI = 0.70
TRUNC_STR = f"{PSI:.2f}"
REWARD_ID = "7wnzkgie"
SHAPE_RES = 256
IG_STEPS = 8

PANO_SIGMA_ROOT = panohead_root() / "panohead_sigma_cubes_for_reward" / f"trunc{TRUNC_STR}"
PER_SEED_CSV = (
    reward_embedding_analysis_dir()
    / "panohead_reward_attribution"
    / f"trunc{TRUNC_STR}"
    / "per_seed_ig_by_region.csv"
)
OUT_DIR = (
    reward_embedding_analysis_dir()
    / "panohead_reward_attribution"
    / f"trunc{TRUNC_STR}"
    / "voxel_ig_heatmaps"
)
TILE_RES = 512
PAD_RATIO = 30 / 256
MARCHING_CUBES_LEVEL = 10.0
MESH_CANONICAL_ANGLE_DEG = -90.0


def _crop_full_to_pads(full_cube_xyz: torch.Tensor, tri_idx, full_res: int):
    rhs = int(tri_idx.ax1horiz.right)
    lhs = int(full_res - tri_idx.ax1horiz.left)
    bot = int(tri_idx.ax2vert.bottom)
    top = int(full_res - tri_idx.ax2vert.top)
    rear = int(tri_idx.ax3depth.rear)
    front = int(full_res - tri_idx.ax3depth.front)
    return (full_cube_xyz[rhs:full_res - lhs,
                          bot:full_res - top,
                          rear:full_res - front].clone(),
            (rhs, full_res - lhs, bot, full_res - top, rear, full_res - front))


def _reward_forward(reward_model, sigma_aug, vol_xyz: torch.Tensor):
    aug = sigma_aug(vol_xyz)
    inp = aug.permute(2, 1, 0).contiguous().unsqueeze(0)
    emb8192 = reward_model.Conv3DModule.forward_to_global_vec(inp, return_global_only=True)
    emb512 = reward_model.MLP(emb8192)
    return reward_model.forward_to_scalar_reward_from_single_global(emb512).reshape(())


def _integrated_gradients(reward_model, sigma_aug,
                          vol_xyz: torch.Tensor, device: torch.device,
                          steps: int = IG_STEPS) -> torch.Tensor:
    x1 = vol_xyz.to(device)
    x0 = torch.zeros_like(x1)
    delta = x1 - x0
    total_grad = torch.zeros_like(x0)
    for alpha in torch.linspace(1.0 / steps, 1.0, steps, device=device):
        x = (x0 + alpha * delta).clone().detach().requires_grad_(True)
        r = _reward_forward(reward_model, sigma_aug, x)
        grad = torch.autograd.grad(r, x, retain_graph=False, create_graph=False)[0]
        total_grad += grad.detach()
    return (delta * total_grad / float(steps)).detach().cpu()


def _build_ig_full_cube(ig_cropped: np.ndarray, full_res: int,
                       crop_bounds: Tuple[int, ...]) -> np.ndarray:
    """Place the cropped IG slab back into a 256³ full cube with zeros outside."""
    rhs, lhs_end, bot, top_end, rear, front_end = crop_bounds
    full = np.zeros((full_res, full_res, full_res), dtype=np.float32)
    full[rhs:lhs_end, bot:top_end, rear:front_end] = ig_cropped
    return full


def _make_colorbar(width: int, height: int, vmax: float) -> np.ndarray:
    """Horizontal colorbar matching the mesh colormap. Diverging blue→grey→red,
    with [-vmax, +vmax] labelled at ends and 0 in the middle."""
    bar_h = max(20, height // 3)
    pad_top = (height - bar_h) // 2
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    bar = np.zeros((bar_h, width, 3), dtype=np.uint8)
    xs = np.linspace(-1.0, 1.0, width)
    for i, z in enumerate(xs):
        if z > 0:
            r = int(round(255 * z))
            g = int(round(180 * (1.0 - z)))
            b = int(round(180 * (1.0 - z)))
        elif z < 0:
            r = int(round(180 * (1.0 + z)))
            g = int(round(180 * (1.0 + z)))
            b = int(round(255 * (-z)))
        else:
            r = g = b = 180
        bar[:, i, 0] = r
        bar[:, i, 1] = g
        bar[:, i, 2] = b
    canvas[pad_top:pad_top + bar_h] = bar
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
    except Exception:
        font = ImageFont.load_default()
    label_y = pad_top + bar_h + 4
    draw.text((6, label_y), f"-{vmax:.3f} (reward ↓)",
              fill=(180, 180, 255), font=font)
    draw.text((width // 2 - 8, label_y), "0", fill=(220, 220, 220), font=font)
    draw.text((width - 130, label_y), f"+{vmax:.3f} (reward ↑)",
              fill=(255, 180, 180), font=font)
    draw.text((6, 4), "per-voxel IG attribution (signed)",
              fill=(255, 255, 255), font=font)
    return np.asarray(img)


def _render_ig_mesh(verts: np.ndarray, faces: np.ndarray,
                    vert_ig: np.ndarray, title: str,
                    vmax: float,
                    angle_deg: float = MESH_CANONICAL_ANGLE_DEG) -> np.ndarray:
    mesh = trimesh.Trimesh(vertices=verts.copy(), faces=faces.copy())
    mesh.fix_normals()
    mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
    s = float(np.max(np.abs(mesh.vertices)))
    if s > 0:
        mesh.vertices = mesh.vertices / s

    # Normalise IG to [-1, +1] using a SHARED vmax across all rendered meshes,
    # so colour intensities are directly comparable seed-to-seed.
    if vmax <= 0:
        vmax = 1.0
    z = np.clip(vert_ig / vmax, -1.0, 1.0)
    # Diverging colormap: red = positive IG (reward up), blue = negative.
    colors = np.zeros((len(verts), 4), dtype=np.uint8)
    pos = z > 0
    neg = z < 0
    colors[pos, 0] = (255 * z[pos]).astype(np.uint8)              # R
    colors[pos, 1] = (180 * (1.0 - z[pos])).astype(np.uint8)      # G
    colors[pos, 2] = (180 * (1.0 - z[pos])).astype(np.uint8)      # B
    colors[neg, 0] = (180 * (1.0 + z[neg])).astype(np.uint8)
    colors[neg, 1] = (180 * (1.0 + z[neg])).astype(np.uint8)
    colors[neg, 2] = (255 * (-z[neg])).astype(np.uint8)
    # outside-of-attribution voxels (vert_ig == 0): grey
    zero = (vert_ig == 0)
    colors[zero, :3] = 180
    colors[:, 3] = 255
    mesh.visual.vertex_colors = colors

    rot = trimesh.transformations.rotation_matrix(np.radians(angle_deg), [0, 1, 0])
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
    scene.add(pyrender.PointLight(color=[255, 255, 255], intensity=8.0),
              pose=light_pose)
    r = pyrender.OffscreenRenderer(TILE_RES, TILE_RES)
    color, _ = r.render(scene)
    r.delete()
    # add title strip
    img = Image.fromarray(color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, TILE_RES, 32), fill=(0, 0, 0))
    draw.text((6, 6), title, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    print(f"[heatmap] loading reward {REWARD_ID}")
    reward_model = reward_loading.load_rwd_model_from_cfg(REWARD_ID).to(device).eval()
    run_cfg = OmegaConf.load(
        RLHF_SRC_ROOT / "RWD_MODELS_FOR_TUNING" / REWARD_ID / "run_config.yaml"
    )
    sigma_aug = hydra.utils.instantiate(run_cfg.data.augmentations.sigma_norm).eval()
    if hasattr(sigma_aug, "to"):
        sigma_aug = sigma_aug.to(device)

    print("[heatmap] loading EG3D-orig for pads_vals_entire crop indices")
    da = gen_utils.load_generator(
        model_path=REPO_ROOT / "pkl_pt" / "eg3d_1" / "ffhq512-128.pkl",
        truncation_psi=0.7, truncation_cutoff=14, shape_res=SHAPE_RES,
    )
    mudc = finetuning_utils.MeshUtilsDataClass()
    pads = OmegaConf.load(gen_utils.STATIC_CONFIGS_DIR / "pads_vals_entire.yaml")
    _samples, _shape, tri_idx = mudc.get_samples_coordinates_from_pads_vals_dict(
        pads_vals=pads, G=da.G, shape_res=SHAPE_RES,
    )

    # Select 3 top + 3 bottom by reward_panohead
    df = pd.read_csv(PER_SEED_CSV).sort_values("reward_panohead", ascending=False).reset_index(drop=True)
    top_seeds = df.head(3)["seed"].astype(int).tolist()
    bot_seeds = df.tail(3)["seed"].astype(int).tolist()
    print(f"[heatmap] top-3 seeds = {top_seeds}, bot-3 seeds = {bot_seeds}")

    # Pass 1: compute IG + vertex IG arrays for all 6 seeds, accumulate
    # |vert_ig| for shared-vmax computation.
    cache = []
    abs_pool: List[np.ndarray] = []
    for label, seeds in [("TOP", top_seeds), ("BOT", bot_seeds)]:
        for seed in seeds:
            print(f"[heatmap] pass-1 IG  {label} seed {seed}")
            full = torch.load(
                PANO_SIGMA_ROOT / f"sigma_seed_{seed}.pt", map_location="cpu",
            ).float()
            cropped, bounds = _crop_full_to_pads(full, tri_idx, SHAPE_RES)
            vol_dev = cropped.to(device)
            ig = _integrated_gradients(reward_model, sigma_aug, vol_dev, device)
            with torch.no_grad():
                r = float(_reward_forward(reward_model, sigma_aug, vol_dev).cpu())
            ig_full = _build_ig_full_cube(ig.numpy(), SHAPE_RES, bounds)
            sigmas = full.numpy().copy().astype(np.float32)
            pad = max(1, int(round(PAD_RATIO * SHAPE_RES)))
            sigmas[:pad] = sigmas[-pad:] = -1000
            sigmas[:, :pad] = sigmas[:, -pad:] = -1000
            sigmas[:, :, :pad] = sigmas[:, :, -pad:] = -1000
            verts_idx, faces_idx, _, _ = measure.marching_cubes(
                np.transpose(sigmas, (2, 1, 0)),
                level=MARCHING_CUBES_LEVEL, spacing=[1, 1, 1],
            )
            verts_vox = verts_idx[:, [2, 1, 0]]
            vi = np.clip(np.round(verts_vox).astype(np.int64), 0, SHAPE_RES - 1)
            vert_ig = ig_full[vi[:, 0], vi[:, 1], vi[:, 2]]
            cache.append({
                "label": label, "seed": int(seed), "reward": r,
                "verts": verts_idx, "faces": faces_idx, "vert_ig": vert_ig,
            })
            abs_pool.append(np.abs(vert_ig))
            # free large tensors
            del full, cropped, vol_dev, ig, ig_full, sigmas

    # Shared vmax = 99th percentile of |vert_ig| pooled across all 6 meshes,
    # ignoring exact zeros so the percentile reflects the meaningful tail.
    pooled = np.concatenate([a[a != 0.0] for a in abs_pool]) if any((a != 0).any() for a in abs_pool) else np.concatenate(abs_pool)
    vmax_shared = float(np.percentile(pooled, 99)) if len(pooled) else 1.0
    print(f"[heatmap] shared vmax (99th pct of pooled |vert_ig|, nonzeros) = {vmax_shared:.4f}")
    # Also compute the actual extrema for the legend label
    abs_all = np.concatenate(abs_pool) if abs_pool else np.array([1.0])
    print(f"[heatmap] pooled |vert_ig| stats: max={abs_all.max():.4f}, "
          f"median(nonzero)={np.median(pooled) if len(pooled) else 0.0:.4f}")

    # Pass 2: render meshes with shared vmax
    tiles_top: List[np.ndarray] = []
    tiles_bot: List[np.ndarray] = []
    for entry in cache:
        tile = _render_ig_mesh(
            entry["verts"], entry["faces"], entry["vert_ig"],
            title=f"{entry['label']} seed {entry['seed']}  r={entry['reward']:+.3f}",
            vmax=vmax_shared,
        )
        (tiles_top if entry["label"] == "TOP" else tiles_bot).append(tile)
        Image.fromarray(tile).save(
            OUT_DIR / f"ig_mesh_{entry['label'].lower()}_seed_{entry['seed']}.jpg",
            quality=92,
        )

    strip_top = np.concatenate(tiles_top, axis=1)
    strip_bot = np.concatenate(tiles_bot, axis=1)
    sep = np.full((20, strip_top.shape[1], 3), 0, dtype=np.uint8)
    grid_only = np.concatenate([strip_top, sep, strip_bot], axis=0)

    # Build a colorbar legend band beneath the grid.
    cbar_h = 80
    cbar = _make_colorbar(width=grid_only.shape[1], height=cbar_h, vmax=vmax_shared)
    final = np.concatenate([grid_only, cbar], axis=0)
    out_path = OUT_DIR / "ig_mesh_top3_vs_bot3.jpg"
    Image.fromarray(final).save(out_path, quality=92)
    print(f"[heatmap] saved {out_path}")


if __name__ == "__main__":
    main()
