"""Build a PanoHead-aligned AW98 region partition of the σ-cube, mirroring
aw98_template_partition.py (which is EG3D-aligned).

PanoHead's σ field has its face centred at cam_pivot=[0,0,0.2] (vs EG3D's
[0,0,0]), so the AW98 masks have to be re-derived per PanoHead's camera
convention. Otherwise the masks would be misaligned by +0.2 in z.

Stages (run with --stage A then B then C):

  A  AW98 detection on the 100 PanoHead canonical RGBs already on disk
     (PanoHead/panohead_sigma_cubes_for_reward/trunc{psi}/rgb_canonical/).
     Saves 2D landmark coords + depth-at-landmark per seed.
  B  Back-project per-seed 2D landmarks → world coords via PanoHead's
     cam2world + pinhole intrinsics (cam_pivot=[0,0,0.2], radius=2.7).
     Average across 100 seeds → mean (98, 3) world template.
  C  Build region masks in σ-cube voxel space using the same WFLW
     semantic groups + AABB extents as aw98_template_partition.py,
     but with PanoHead's pads_vals_entire crop indices.

Output saved to:
  reward_embedding_analysis/panohead_aw98_template_masks/region_masks.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from tqdm.auto import tqdm

# EG3D + core_modules root for the σ-cube sampling pipeline (we re-use the same
# pads_vals_entire crop and MeshUtilsDataClass for voxel-coord world map).
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

# Add PanoHead repo to sys.path (it has its own training/dnnlib/torch_utils
# folders; once core_modules imports happen the EG3D versions win — for this
# script we only need PanoHead's camera_utils which is pure-Python no-collide).
PANOHEAD_REPO = panohead_root()
if str(PANOHEAD_REPO) not in sys.path:
    sys.path.append(str(PANOHEAD_REPO))

from core_modules.data.create_train_data import generation_utils as gen_utils  # noqa: E402
from core_modules.utils import finetuning_utils  # noqa: E402
from core_modules.utils.awloss_utils_AM import AW98Helper  # noqa: E402

# WFLW 98-pt semantic groups (must match aw98_template_partition.py).
WFLW_GROUPS: Dict[str, List[int]] = {
    "jaw_contour":   list(range(0, 33)),
    "brow_left":     list(range(33, 42)),
    "brow_right":    list(range(42, 51)),
    "nose_bridge":   list(range(51, 55)),
    "nose_bottom":   list(range(55, 60)),
    "eye_left":      list(range(60, 68)) + [96],
    "eye_right":     list(range(68, 76)) + [97],
    "mouth_outer":   list(range(76, 88)),
    "mouth_inner":   list(range(88, 96)),
}

REGION_PRIORITY: Tuple[str, ...] = (
    "nose", "mouth",
    "left_eye_orbit", "right_eye_orbit",
    "brow",
    "left_cheek", "right_cheek",
    "chin", "forehead", "jaw_periphery_ears",
    "front_of_camera", "background_rear", "other",
)

ANALYSIS_ROOT = reward_embedding_analysis_dir()
SEEDS: List[int] = list(range(200000, 200100))

WORKDIR = ANALYSIS_ROOT / "panohead_aw98_template_workdir"
MASKS_PT = ANALYSIS_ROOT / "panohead_aw98_template_masks" / "region_masks.pt"
MASKS_JSON = ANALYSIS_ROOT / "panohead_aw98_template_masks" / "region_metadata.json"

# PanoHead camera convention (matches single_dmap_conditioning for EG3D
# except that avg_camera_pivot is [0, 0, 0.2]).
PANO_CAM_PIVOT = (0.0, 0.0, 0.2)
PANO_CAM_RADIUS = 2.7
PANO_FOV_DEG = 18.837
PANO_RGB_HW = (512, 512)
SHAPE_RES = 256


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _aw98_detect_on_rgb_path(aw98: AW98Helper, rgb_path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(rgb_path).convert("RGB")
    arr = torch.as_tensor(np.asarray(img), device=device, dtype=torch.float32) / 255.0
    chw = arr.permute(2, 0, 1).unsqueeze(0)
    lmks = aw98.predict_landmarks_from_rgb_on_gpu(chw, detach=True).squeeze(0)
    return lmks  # (98, 2) pixel coords


def _build_panohead_cam2world_intrinsics(device: torch.device):
    """Build PanoHead canonical cam2world + normalised intrinsics matching
    its `LookAtPoseSampler.sample(π/2, π/2, cam_pivot=[0,0,0.2], radius=2.7)`
    convention and `FOV_to_intrinsics(18.837)`. This avoids importing
    PanoHead's training/dnnlib (which would collide with EG3D's)."""
    # FOV_to_intrinsics: focal = 1 / (2 * tan(deg2rad(fov_deg) / 2)). Center
    # at (0.5, 0.5). Returns 3x3 normalised K.
    focal = 1.0 / (2.0 * np.tan(np.deg2rad(PANO_FOV_DEG) / 2.0))
    K = torch.tensor(
        [[focal, 0.0, 0.5],
         [0.0, focal, 0.5],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32, device=device,
    )
    # LookAtPoseSampler.sample(π/2, π/2, pivot, radius) → camera at
    # pivot + radius * (cos(π/2)·sin(π/2), -sin(π/2)·sin(π/2), -cos(π/2))
    #  ... but the EG3D reference single_dmap_conditioning has cam at
    # (0, 0, 2.7) for pivot=[0,0,0]; for pivot=[0,0,0.2] it shifts to
    # (0, 0, 0.2 + 2.7) = (0, 0, 2.9) keeping the same orientation.
    pivot = torch.tensor(PANO_CAM_PIVOT, dtype=torch.float32, device=device)
    cam2world = torch.tensor(
        [[1.0, 0.0, 0.0, float(pivot[0])],
         [0.0, -1.0, 0.0, float(pivot[1])],
         [0.0, 0.0, -1.0, float(pivot[2]) + PANO_CAM_RADIUS],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float32, device=device,
    )
    return cam2world, K


def _load_depth_map_from_disk_or_none(seed: int, trunc_str: str):
    """PanoHead's extract_sigmas_for_reward_transfer.py does not save the
    depth map; we'd have to re-render it. For now return None and lift via
    ray-marching the cube to find the surface, OR (simpler) approximate
    depth=2.7 (cam-to-origin distance). We use the σ-cube to find the
    actual surface depth per landmark — that gives a per-landmark world
    z without requiring a separate depth render."""
    return None


def _backproject_pixel_to_world(
    lmks_pixel: torch.Tensor,   # (N, 2) in image pixel coords
    depths: torch.Tensor,       # (N,) world depth along ray
    cam2world: torch.Tensor, K: torch.Tensor, rgb_hw: Tuple[int, int],
) -> torch.Tensor:
    """Match aw98_template_partition._backproject_pixels_to_world."""
    h_rgb, w_rgb = rgb_hw
    u = lmks_pixel[:, 0].float() / float(w_rgb)
    v = lmks_pixel[:, 1].float() / float(h_rgb)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    z_cam = torch.ones_like(x_cam)
    ray_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)
    R = cam2world[:3, :3]
    origin = cam2world[:3, 3]
    ray_world = ray_cam @ R.T
    ray_world = ray_world / ray_world.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return origin.unsqueeze(0) + depths.unsqueeze(1) * ray_world


def _depths_from_sigma_cube(
    cube_xyz: np.ndarray,    # (256, 256, 256) — sigma_seed_{s}.pt content
    lmks_pixel: torch.Tensor, rgb_hw: Tuple[int, int],
    cam2world: torch.Tensor, K: torch.Tensor,
    box_warp: float = 1.0, sigma_threshold: float = 10.0,
) -> torch.Tensor:
    """Ray-march each landmark's pixel ray through the σ cube and return the
    distance along the ray at which σ first crosses `sigma_threshold`. Done
    in NumPy because per-landmark ray marching is small."""
    h_rgb, w_rgb = rgb_hw
    u = (lmks_pixel[:, 0].cpu().numpy() / float(w_rgb)).astype(np.float32)
    v = (lmks_pixel[:, 1].cpu().numpy() / float(h_rgb)).astype(np.float32)
    K_np = K.cpu().numpy()
    cam2world_np = cam2world.cpu().numpy()
    fx, fy = float(K_np[0, 0]), float(K_np[1, 1])
    cx, cy = float(K_np[0, 2]), float(K_np[1, 2])
    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    z_cam = np.ones_like(x_cam)
    ray_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    R = cam2world_np[:3, :3]
    origin = cam2world_np[:3, 3]
    ray_world = ray_cam @ R.T
    ray_world = ray_world / (np.linalg.norm(ray_world, axis=1, keepdims=True) + 1e-8)
    # Cube grid: 256 voxels span world [-box_warp/2, +box_warp/2] in each axis.
    half = box_warp / 2.0
    N = cube_xyz.shape[0]
    voxel_size = box_warp / (N - 1)
    n_lm = len(u)
    depths = np.zeros(n_lm, dtype=np.float32)
    # Distance from camera to face is roughly cam_radius (2.7) for orig EG3D
    # and cam_radius + pivot.z ≈ 2.5 to 2.9 for PanoHead. Search ±0.6 around 2.7.
    t_min = PANO_CAM_RADIUS - 0.6
    t_max = PANO_CAM_RADIUS + 0.6
    n_steps = 192
    ts = np.linspace(t_min, t_max, n_steps)
    for i in range(n_lm):
        best_t = float("nan")
        for t in ts:
            p = origin + t * ray_world[i]
            # Map world → voxel index
            ix = int(round((p[0] + half) / voxel_size))
            iy = int(round((p[1] + half) / voxel_size))
            iz = int(round((p[2] + half) / voxel_size))
            if 0 <= ix < N and 0 <= iy < N and 0 <= iz < N:
                # Note: cube_xyz axes are (X, Y, Z) per create_samples
                # reshape. PanoHead's extract reshape is the same.
                if cube_xyz[ix, iy, iz] >= sigma_threshold:
                    best_t = float(t)
                    break
        if not np.isnan(best_t):
            depths[i] = best_t
        else:
            depths[i] = PANO_CAM_RADIUS  # fallback: distance to origin
    return torch.from_numpy(depths)


def stage_a(device: torch.device, trunc_str: str) -> None:
    rgb_dir = PANOHEAD_REPO / "panohead_sigma_cubes_for_reward" / f"trunc{trunc_str}" / "rgb_canonical"
    if not rgb_dir.exists():
        raise SystemExit(f"missing PanoHead RGB dir at {rgb_dir}")
    work = _ensure_dir(WORKDIR / f"trunc{trunc_str}")
    print(f"[A] loading AW98")
    aw98 = AW98Helper()
    aw98.M_aw98 = aw98.M_aw98.to(device).eval()
    all_lmks = torch.zeros(len(SEEDS), 98, 2)
    for i, seed in enumerate(tqdm(SEEDS, desc="AW98 on PanoHead RGBs")):
        out_pt = work / f"lmks_2d_seed_{seed}.pt"
        rgb_path = rgb_dir / f"rgb_seed_{seed}.jpg"
        if not rgb_path.exists():
            raise SystemExit(f"missing PanoHead RGB at {rgb_path}")
        if out_pt.exists():
            lmks = torch.load(out_pt, map_location="cpu")
        else:
            lmks = _aw98_detect_on_rgb_path(aw98, rgb_path, device)
            torch.save(lmks.detach().cpu(), out_pt)
        all_lmks[i] = lmks.detach().cpu()
    torch.save(all_lmks, work / "all_lmks_2d.pt")
    torch.save(all_lmks.mean(dim=0), work / "mean_lmks_2d.pt")
    print(f"[A] saved AW98 landmarks for {len(SEEDS)} PanoHead seeds at trunc={trunc_str}")


def stage_b(device: torch.device, trunc_str: str) -> None:
    work = WORKDIR / f"trunc{trunc_str}"
    all_lmks = torch.load(work / "all_lmks_2d.pt", map_location="cpu")  # (N, 98, 2)
    sigma_dir = PANOHEAD_REPO / "panohead_sigma_cubes_for_reward" / f"trunc{trunc_str}"
    cam2world, K = _build_panohead_cam2world_intrinsics(device)
    print(f"[B] PanoHead cam2world:")
    print(cam2world.cpu().numpy())
    print(f"[B] intrinsics:")
    print(K.cpu().numpy())

    per_seed_world = []
    for i, seed in enumerate(tqdm(SEEDS, desc="back-project (ray-march σ)")):
        cube_pt = sigma_dir / f"sigma_seed_{seed}.pt"
        cube = torch.load(cube_pt, map_location="cpu").numpy()
        lmks = all_lmks[i].to(device)
        depths = _depths_from_sigma_cube(cube, lmks, PANO_RGB_HW, cam2world, K)
        world = _backproject_pixel_to_world(lmks, depths.to(device), cam2world, K, PANO_RGB_HW)
        per_seed_world.append(world.detach().cpu())
    all_world = torch.stack(per_seed_world)
    mean_world = all_world.mean(dim=0)
    torch.save(all_world, work / "all_world_lmks_3d.pt")
    torch.save(mean_world, work / "mean_world_lmks_3d.pt")

    def fmt(v):
        return "(" + ", ".join(f"{x:+.4f}" for x in v.tolist()) + ")"
    nose_tip = mean_world[54]
    brow_inner = (mean_world[41] + mean_world[42]) / 2.0
    chin_pt = mean_world[16]
    pupil_l = mean_world[96]
    pupil_r = mean_world[97]
    mouth_centre = (mean_world[76] + mean_world[82]) / 2.0
    print("[B] Mean 3D landmark world coords (sanity check):")
    print(f"  nose_tip      = {fmt(nose_tip)}")
    print(f"  brow_inner    = {fmt(brow_inner)}")
    print(f"  chin_pt       = {fmt(chin_pt)}")
    print(f"  pupil_left    = {fmt(pupil_l)}")
    print(f"  pupil_right   = {fmt(pupil_r)}")
    print(f"  mouth_centre  = {fmt(mouth_centre)}")
    print(f"  face x range  = [{mean_world[:33, 0].min().item():+.4f}, "
          f"{mean_world[:33, 0].max().item():+.4f}]")
    print(f"  face y range  = [{mean_world[:, 1].min().item():+.4f}, "
          f"{mean_world[:, 1].max().item():+.4f}]")
    print(f"  face z range  = [{mean_world[:, 2].min().item():+.4f}, "
          f"{mean_world[:, 2].max().item():+.4f}]")
    print(f"[B] saved mean_world_lmks_3d.pt to {work}")


def _aabb_world(coords: torch.Tensor):
    return (
        float(coords[:, 0].min()), float(coords[:, 0].max()),
        float(coords[:, 1].min()), float(coords[:, 1].max()),
        float(coords[:, 2].min()), float(coords[:, 2].max()),
    )


def _build_region_aabbs(mean_world: torch.Tensor) -> Dict[str, Dict[str, float]]:
    g = {name: mean_world[idxs] for name, idxs in WFLW_GROUPS.items()}
    margin_xy = 0.015
    z_back = 0.12
    z_front = 0.03

    def aabb_with_margin(pts: torch.Tensor) -> Dict[str, float]:
        x0, x1, y0, y1, z0, z1 = _aabb_world(pts)
        return {"x_min": x0 - margin_xy, "x_max": x1 + margin_xy,
                "y_min": y0 - margin_xy, "y_max": y1 + margin_xy,
                "z_min": z0 - z_back,    "z_max": z1 + z_front}

    aabbs: Dict[str, Dict[str, float]] = {}
    aabbs["nose"] = aabb_with_margin(torch.cat([g["nose_bridge"], g["nose_bottom"]]))
    aabbs["mouth"] = aabb_with_margin(torch.cat([g["mouth_outer"], g["mouth_inner"]]))
    aabbs["brow"] = aabb_with_margin(torch.cat([g["brow_left"], g["brow_right"]]))
    aabbs["left_eye_orbit"] = aabb_with_margin(g["eye_left"])
    aabbs["right_eye_orbit"] = aabb_with_margin(g["eye_right"])

    chin_pts = mean_world[list(range(12, 21))]
    aabbs["chin"] = aabb_with_margin(chin_pts)

    nose_pts = torch.cat([g["nose_bridge"], g["nose_bottom"]])
    chin_y_max = chin_pts[:, 1].max().item()
    eye_left_y_min = g["eye_left"][:, 1].min().item()
    eye_right_y_min = g["eye_right"][:, 1].min().item()
    nose_x_min = nose_pts[:, 0].min().item()
    nose_x_max = nose_pts[:, 0].max().item()
    jaw_left_pts = mean_world[list(range(2, 12))]
    jaw_right_pts = mean_world[list(range(21, 31))]
    face_z_min = float(torch.cat([nose_pts, chin_pts, g["eye_left"], g["eye_right"]])[:, 2].min())
    face_z_max = float(torch.cat([nose_pts, chin_pts, g["eye_left"], g["eye_right"]])[:, 2].max())

    aabbs["left_cheek"] = {
        "x_min": float(jaw_left_pts[:, 0].min()) + 0.01,
        "x_max": nose_x_min - 0.02,
        "y_min": chin_y_max + 0.02,
        "y_max": eye_left_y_min - 0.01,
        "z_min": face_z_min - 0.05,
        "z_max": face_z_max + 0.02,
    }
    aabbs["right_cheek"] = {
        "x_min": nose_x_max + 0.02,
        "x_max": float(jaw_right_pts[:, 0].max()) - 0.01,
        "y_min": chin_y_max + 0.02,
        "y_max": eye_right_y_min - 0.01,
        "z_min": face_z_min - 0.05,
        "z_max": face_z_max + 0.02,
    }

    brow_combined = torch.cat([g["brow_left"], g["brow_right"]])
    brow_y_top = brow_combined[:, 1].max().item()
    nose_tip_y = float(mean_world[54, 1])
    forehead_height = brow_y_top - nose_tip_y
    aabbs["forehead"] = {
        "x_min": brow_combined[:, 0].min().item() - margin_xy,
        "x_max": brow_combined[:, 0].max().item() + margin_xy,
        "y_min": brow_y_top + 0.005,
        "y_max": brow_y_top + max(forehead_height, 0.10),
        "z_min": brow_combined[:, 2].min().item() - z_back,
        "z_max": brow_combined[:, 2].max().item() + z_front,
    }
    jaw_left_outer = mean_world[list(range(0, 4))]
    jaw_right_outer = mean_world[list(range(29, 33))]
    aabbs["jaw_periphery_ears"] = {
        "x_min": float(torch.cat([jaw_left_outer[:, 0], jaw_right_outer[:, 0]]).min()) - 0.03,
        "x_max": float(torch.cat([jaw_left_outer[:, 0], jaw_right_outer[:, 0]]).max()) + 0.03,
        "y_min": chin_y_max - 0.03,
        "y_max": brow_y_top + 0.05,
        "z_min": face_z_min - 0.08,
        "z_max": face_z_max + 0.02,
    }
    return aabbs


def stage_c(device: torch.device, trunc_str: str) -> None:
    """Build region masks in σ-cube voxel space for PanoHead."""
    work = WORKDIR / f"trunc{trunc_str}"
    mean_world = torch.load(work / "mean_world_lmks_3d.pt", map_location="cpu")

    # Get voxel→world mapping via the same pads_vals_entire + shape_res=256
    # convention the reward model expects. Load EG3D-orig just to get the
    # mudc.get_samples_coordinates_from_pads_vals_dict outputs (the
    # voxel-coord grid is shared between EG3D and PanoHead — same box_warp).
    print(f"[C] loading EG3D-orig to determine pads_vals_entire crop coords "
          "(shared with PanoHead because box_warp=1, voxel_origin=[0,0,0])")
    da = gen_utils.load_generator(
        model_path=REPO_ROOT / "pkl_pt" / "eg3d_1" / "ffhq512-128.pkl",
        truncation_psi=0.7, truncation_cutoff=14, shape_res=SHAPE_RES,
    )
    G = da.G
    box_warp = float(G.rendering_kwargs.get("box_warp", 1.0))
    mudc = finetuning_utils.MeshUtilsDataClass()
    pads = OmegaConf.load(gen_utils.STATIC_CONFIGS_DIR / "pads_vals_entire.yaml")
    samples, shape, tri_idx = mudc.get_samples_coordinates_from_pads_vals_dict(
        pads_vals=pads, G=G, shape_res=SHAPE_RES,
    )
    cube_shape = tuple(int(x) for x in shape[1:4])
    voxel_world = samples.detach().cpu().reshape(*cube_shape, 3)
    vx, vy, vz = voxel_world[..., 0], voxel_world[..., 1], voxel_world[..., 2]
    print(f"[C] cropped cube shape (shared) = {cube_shape}; box_warp = {box_warp}")
    print(f"[C] voxel world bounds: "
          f"x=[{vx.min():.3f}, {vx.max():.3f}], "
          f"y=[{vy.min():.3f}, {vy.max():.3f}], "
          f"z=[{vz.min():.3f}, {vz.max():.3f}]")

    aabbs = _build_region_aabbs(mean_world)
    for name, b in aabbs.items():
        print(f"[C] AABB {name:>20s}: "
              f"x[{b['x_min']:+.3f},{b['x_max']:+.3f}] "
              f"y[{b['y_min']:+.3f},{b['y_max']:+.3f}] "
              f"z[{b['z_min']:+.3f},{b['z_max']:+.3f}]")

    # Front / background bands restricted to face x,y rectangle
    face_z_max_world = float(mean_world[:, 2].max())
    face_z_min_world = float(mean_world[:, 2].min())
    face_x_min_world = float(mean_world[:33, 0].min()) - 0.04
    face_x_max_world = float(mean_world[:33, 0].max()) + 0.04
    face_y_min_world = float(mean_world[:, 1].min()) - 0.03
    face_y_max_world = float(mean_world[:, 1].max()) + 0.08
    in_face_xy = (
        (vx >= face_x_min_world) & (vx <= face_x_max_world) &
        (vy >= face_y_min_world) & (vy <= face_y_max_world)
    )

    assigned = torch.zeros(cube_shape, dtype=torch.bool)
    masks: Dict[str, torch.Tensor] = {}
    for region in REGION_PRIORITY[:-1]:
        if region in aabbs:
            b = aabbs[region]
            raw = (
                (vx >= b["x_min"]) & (vx <= b["x_max"]) &
                (vy >= b["y_min"]) & (vy <= b["y_max"]) &
                (vz >= b["z_min"]) & (vz <= b["z_max"])
            )
        elif region == "front_of_camera":
            raw = in_face_xy & (vz > face_z_max_world + 0.005)
        elif region == "background_rear":
            raw = in_face_xy & (vz < face_z_min_world - 0.005)
        else:
            raise RuntimeError(f"no rule for {region}")
        mask = raw & ~assigned
        masks[region] = mask
        assigned |= mask
    masks["other"] = ~assigned

    out_dir = _ensure_dir(MASKS_PT.parent)
    torch.save(
        {"masks": {k: v.bool().contiguous() for k, v in masks.items()},
         "region_priority": REGION_PRIORITY,
         "aabbs_world": aabbs,
         "cube_shape": cube_shape,
         "box_warp": box_warp,
         "mean_world_lmks": mean_world,
         "panohead_cam_pivot": list(PANO_CAM_PIVOT),
         "trunc": trunc_str,
         },
        MASKS_PT,
    )
    counts = {k: int(v.sum().item()) for k, v in masks.items()}
    total = sum(counts.values())
    named = sum(c for k, c in counts.items() if k not in ("other", "front_of_camera", "background_rear"))
    print(f"[C] Voxel counts per region:")
    for k, v in counts.items():
        print(f"   {k:>20s}: {v:8d}  ({100.0*v/total:5.2f}%)")
    print(f"[C] Named-face regions cover {100.0*named/total:.2f}% of cube.")
    with open(MASKS_JSON, "w") as f:
        json.dump({"voxel_counts": counts, "named_face_fraction": named / total,
                   "aabbs_world": aabbs, "cube_shape": list(cube_shape),
                   "panohead_cam_pivot": list(PANO_CAM_PIVOT)}, f, indent=2)
    print(f"[C] saved masks to {MASKS_PT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B", "C"], required=True)
    ap.add_argument("--truncation-psi", type=float, default=0.7)
    args = ap.parse_args()
    trunc_str = f"{args.truncation_psi:.2f}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "A":
        stage_a(device, trunc_str)
    elif args.stage == "B":
        stage_b(device, trunc_str)
    elif args.stage == "C":
        stage_c(device, trunc_str)


if __name__ == "__main__":
    main()
