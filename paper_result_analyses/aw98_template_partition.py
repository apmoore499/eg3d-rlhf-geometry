"""Build a σ-cube region partition anchored to WFLW-98 (AW98) facial landmarks.

Stage A: render canonical view + run AW98 for each of the 100 exp3 seeds, save
2D landmarks + depth lookup, and write a few overlay JPGs for visual sanity.

Stage B (--stage=B): back-project per-seed landmarks to world coords using the
EG3D canonical-view cam2world + intrinsics, then average across seeds.

Stage C (--stage=C): group landmarks by WFLW semantic IDs, compute per-region
AABB in σ-cube coords, extrapolate forehead from brow, add diagnostic z-bands
(front_of_camera / background_rear), save region masks + 3D mesh overlay.

Stage D is performed by reward_geometry_explainability.py with
`partition_version="aw98_template"`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_result_analyses.path_defaults import (  # noqa: E402
    EG3D_ROOT,
    RLHF_CORE_ROOT,
    reward_embedding_analysis_dir,
)

for _path in (REPO_ROOT, EG3D_ROOT, RLHF_CORE_ROOT.parent):
    _str = str(_path)
    if _str not in sys.path:
        sys.path.insert(0, _str)

try:
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
except Exception:
    pass

from core_modules.data.create_train_data import generation_utils as gen_utils  # noqa: E402
from core_modules.utils import finetuning_utils  # noqa: E402
from core_modules.utils.awloss_utils_AM import AW98Helper  # noqa: E402


SEEDS: List[int] = list(range(200000, 200100))

TRUNCATION_PSI = 0.7
TRUNCATION_CUTOFF = 14
SHAPE_RES = 256
NEURAL_RES = 128

ORIG_PKL = REPO_ROOT / "pkl_pt" / "eg3d_1" / "ffhq512-128.pkl"

ANALYSIS_ROOT = reward_embedding_analysis_dir()
WORKDIR = ANALYSIS_ROOT / "aw98_template_workdir"
MASKS_OUT_PT = ANALYSIS_ROOT / "aw98_template_masks" / "region_masks.pt"
MASKS_OUT_JSON = ANALYSIS_ROOT / "aw98_template_masks" / "region_metadata.json"

OVERLAY_SEEDS = (200000, 200050, 200060)


# WFLW 98-point semantic groups (standard layout)
WFLW_GROUPS: Dict[str, List[int]] = {
    "jaw_contour": list(range(0, 33)),
    "brow_left": list(range(33, 42)),
    "brow_right": list(range(42, 51)),
    "nose_bridge": list(range(51, 55)),
    "nose_bottom": list(range(55, 60)),
    "eye_left": list(range(60, 68)) + [96],
    "eye_right": list(range(68, 76)) + [97],
    "mouth_outer": list(range(76, 88)),
    "mouth_inner": list(range(88, 96)),
}


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _seed_to_z(seed: int, z_dim: int, device: torch.device) -> torch.Tensor:
    z_np = np.random.RandomState(int(seed)).randn(z_dim).astype(np.float32)
    return torch.from_numpy(z_np).unsqueeze(0).to(device)


def _render_canonical_rgb_and_depth(da, tdca, z: torch.Tensor):
    """Render canonical RGB + depth at NEURAL_RES neural rendering resolution."""
    G = da.G
    ws = G.mapping(
        z,
        tdca.conditioning_params,
        truncation_psi=TRUNCATION_PSI,
        truncation_cutoff=TRUNCATION_CUTOFF,
    )
    with torch.no_grad():
        out = G.synthesis(
            ws,
            tdca.camera_params,
            neural_rendering_resolution=NEURAL_RES,
            noise_mode="const",
        )
    return out["image"], out["image_depth"]


def _rgb_neg1_to_01(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb + 1.0) * 0.5


def _aw98_detect(aw98: AW98Helper, rgb_01: torch.Tensor) -> torch.Tensor:
    """Run AW98 on an (1, 3, H, W) RGB tensor in [0,1]. Returns (98, 2) pixel coords."""
    lmks = aw98.predict_landmarks_from_rgb_on_gpu(rgb_01, detach=True)
    return lmks.squeeze(0)


def _sample_depth_at_pixels(
    depth: torch.Tensor,
    lmks_pixel: torch.Tensor,
    rgb_hw: Tuple[int, int],
) -> torch.Tensor:
    """Sample depth at pixel coords (in RGB resolution); returns (N,)."""
    h_rgb, w_rgb = rgb_hw
    h_d, w_d = depth.shape[-2:]
    px = (lmks_pixel[:, 0] / float(w_rgb)) * float(w_d)
    py = (lmks_pixel[:, 1] / float(h_rgb)) * float(h_d)
    px = px.clamp(0, w_d - 1)
    py = py.clamp(0, h_d - 1)
    ix = px.round().long()
    iy = py.round().long()
    return depth[0, 0, iy, ix]


def _draw_overlay(rgb_01: torch.Tensor, lmks_pixel: torch.Tensor, title: str) -> Image.Image:
    rgb_np = (rgb_01.clamp(0, 1).cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    img = Image.fromarray(rgb_np)
    draw = ImageDraw.Draw(img)
    coords = lmks_pixel.cpu().numpy()
    radius = max(2, img.size[0] // 256)
    palette = {
        "jaw_contour": (255, 0, 0),
        "brow_left": (255, 165, 0),
        "brow_right": (255, 200, 0),
        "nose_bridge": (0, 255, 0),
        "nose_bottom": (0, 200, 0),
        "eye_left": (0, 200, 255),
        "eye_right": (0, 150, 255),
        "mouth_outer": (255, 0, 255),
        "mouth_inner": (200, 0, 200),
    }
    for group_name, idxs in WFLW_GROUPS.items():
        color = palette[group_name]
        for i in idxs:
            x, y = float(coords[i, 0]), float(coords[i, 1])
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, 6), title, fill=(255, 255, 255), font=font)
    return img


def stage_a(device: torch.device) -> None:
    _ensure_dir(WORKDIR)
    print(f"[stage A] Loading untuned generator: {ORIG_PKL}")
    da = gen_utils.load_generator(
        model_path=ORIG_PKL,
        truncation_psi=TRUNCATION_PSI,
        truncation_cutoff=TRUNCATION_CUTOFF,
        shape_res=SHAPE_RES,
    )
    tdca = gen_utils.get_single_dmap_cam(da)
    G = da.G

    print("[stage A] Loading AW98 model")
    aw98 = AW98Helper()
    # AW98Helper.get_device() is broken (assumes conv1.conv path); move the
    # underlying model to CUDA ourselves and discover the device from a param.
    aw98.M_aw98 = aw98.M_aw98.to(device).eval()
    aw98_device = next(aw98.M_aw98.parameters()).device

    rgb_resolution: Tuple[int, int] = (G.img_resolution, G.img_resolution)

    overlays_done = set()
    for seed in tqdm(SEEDS, desc="AW98 per-seed detection"):
        out_pt = WORKDIR / f"lmks_2d_seed_{seed}.pt"
        depth_pt = WORKDIR / f"depth_at_lmks_seed_{seed}.pt"
        if out_pt.exists() and depth_pt.exists():
            continue
        z = _seed_to_z(seed, G.z_dim, device)
        rgb, depth = _render_canonical_rgb_and_depth(da, tdca, z)
        rgb_01 = _rgb_neg1_to_01(rgb)
        lmks = _aw98_detect(aw98, rgb_01.to(aw98_device))
        depth_at_lmks = _sample_depth_at_pixels(depth, lmks.to(device), rgb_resolution)
        torch.save(lmks.detach().cpu(), out_pt)
        torch.save(depth_at_lmks.detach().cpu(), depth_pt)

        if seed in OVERLAY_SEEDS and seed not in overlays_done:
            ov = _draw_overlay(rgb_01.detach().cpu(), lmks.detach().cpu(),
                               f"seed {seed} (canonical, psi={TRUNCATION_PSI})")
            ov.save(WORKDIR / f"overlay_seed_{seed}.jpg", quality=92)
            overlays_done.add(seed)

    # Compute averaged 2D landmarks across all seeds (saves time in Stage B).
    all_lmks = torch.stack(
        [torch.load(WORKDIR / f"lmks_2d_seed_{s}.pt", map_location="cpu") for s in SEEDS]
    )  # (N, 98, 2)
    mean_lmks = all_lmks.mean(dim=0)  # (98, 2)
    torch.save(mean_lmks, WORKDIR / "mean_lmks_2d.pt")
    torch.save(all_lmks, WORKDIR / "all_lmks_2d.pt")

    print(f"[stage A] mean_lmks_2d saved to {WORKDIR/'mean_lmks_2d.pt'} "
          f"(shape={tuple(mean_lmks.shape)})")
    print(f"[stage A] Overlay JPGs written for seeds {OVERLAY_SEEDS}.")
    print(f"[stage A] PAUSE: eyeball the overlays at {WORKDIR}/ before stage B.")


def _load_canonical_camera(device: torch.device):
    """Load the EG3D canonical-view cam2world + normalised intrinsics."""
    cam_path = gen_utils.STATIC_CONFIGS_DIR / "single_dmap_cameras.pt"
    cam_params = torch.load(cam_path, map_location=device).float()  # (1, 25)
    flat = cam_params.reshape(-1)
    cam2world = flat[:16].reshape(4, 4)
    intrinsics = flat[16:25].reshape(3, 3)
    return cam2world, intrinsics


def _backproject_pixels_to_world(
    lmks_pixel: torch.Tensor,
    depth_at_lmks: torch.Tensor,
    cam2world: torch.Tensor,
    intrinsics: torch.Tensor,
    rgb_hw: Tuple[int, int],
) -> torch.Tensor:
    """(N, 2) pixel + (N,) ray depth -> (N, 3) world coords.

    EG3D uses normalised image coords in [0, 1] with the OpenCV-style pinhole
    intrinsics in single_dmap_conditioning.pt (fx=fy=4.263, cx=cy=0.5). The
    camera looks down its own +Z axis; depth from G.synthesis() is the
    ray-marched distance along the *normalised* ray direction.
    """
    h_rgb, w_rgb = rgb_hw
    u = lmks_pixel[:, 0].float() / float(w_rgb)  # [0, 1]
    v = lmks_pixel[:, 1].float() / float(h_rgb)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    z_cam = torch.ones_like(x_cam)
    ray_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)  # (N, 3)
    R = cam2world[:3, :3]
    cam_origin = cam2world[:3, 3]
    ray_world = ray_cam @ R.T  # (N, 3)
    ray_world = ray_world / ray_world.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return cam_origin.unsqueeze(0) + depth_at_lmks.unsqueeze(1) * ray_world


def stage_b(device: torch.device) -> None:
    print(f"[stage B] Loading per-seed 2D landmarks + depth from {WORKDIR}")
    all_lmks = torch.load(WORKDIR / "all_lmks_2d.pt", map_location="cpu")  # (N_seeds, 98, 2)
    n_seeds = all_lmks.shape[0]
    if n_seeds != len(SEEDS):
        raise RuntimeError(
            f"Stage A produced {n_seeds} seeds but {len(SEEDS)} expected. Re-run --stage A."
        )

    # We need the RGB resolution that AW98 was run on. Reload the generator to get it.
    da = gen_utils.load_generator(
        model_path=ORIG_PKL,
        truncation_psi=TRUNCATION_PSI,
        truncation_cutoff=TRUNCATION_CUTOFF,
        shape_res=SHAPE_RES,
    )
    rgb_hw = (da.G.img_resolution, da.G.img_resolution)
    print(f"[stage B] RGB resolution = {rgb_hw}; cube box_warp = "
          f"{da.G.rendering_kwargs.get('box_warp', 'unknown')}")

    cam2world, intrinsics = _load_canonical_camera(device)
    print("[stage B] cam2world:")
    print(cam2world.cpu().numpy())
    print("[stage B] intrinsics (normalised image coords):")
    print(intrinsics.cpu().numpy())

    per_seed_world: List[torch.Tensor] = []
    for i, seed in enumerate(SEEDS):
        lmks = torch.load(WORKDIR / f"lmks_2d_seed_{seed}.pt", map_location=device).float()
        depth = torch.load(WORKDIR / f"depth_at_lmks_seed_{seed}.pt", map_location=device).float()
        world = _backproject_pixels_to_world(lmks, depth, cam2world, intrinsics, rgb_hw)
        per_seed_world.append(world.detach().cpu())
    all_world = torch.stack(per_seed_world)  # (N_seeds, 98, 3)
    mean_world = all_world.mean(dim=0)  # (98, 3)

    torch.save(all_world, WORKDIR / "all_world_lmks_3d.pt")
    torch.save(mean_world, WORKDIR / "mean_world_lmks_3d.pt")

    # Spot-check anatomical positions.
    # WFLW indices: nose_tip ~ 54 (last nose-bridge point) or center of nose_bottom (55-59);
    # use 54 as the most-forward nose-bridge point. Inner-brow midpoint ~ mean(41, 42).
    # Chin point = jaw_contour midpoint (index 16). Inner-eye centers = pupils 96, 97.
    def fmt(v):
        return "(" + ", ".join(f"{x:+.4f}" for x in v.tolist()) + ")"
    nose_tip = mean_world[54]
    brow_inner = (mean_world[41] + mean_world[42]) / 2.0
    chin_pt = mean_world[16]
    pupil_l = mean_world[96]
    pupil_r = mean_world[97]
    mouth_centre = (mean_world[76] + mean_world[82]) / 2.0  # outer mouth left + right corner
    print("[stage B] Mean 3D landmark world coords (sanity check):")
    print(f"  nose_tip      (idx 54)       = {fmt(nose_tip)}")
    print(f"  brow_inner    (mean 41+42)/2 = {fmt(brow_inner)}")
    print(f"  chin_pt       (idx 16)       = {fmt(chin_pt)}")
    print(f"  pupil_left    (idx 96)       = {fmt(pupil_l)}")
    print(f"  pupil_right   (idx 97)       = {fmt(pupil_r)}")
    print(f"  mouth_centre  (mean 76+82)/2 = {fmt(mouth_centre)}")
    print("  expect: nose_tip.z > brow_inner.z (nose protrudes forward)")
    print("         brow_inner.y > pupil.y > nose_tip.y > mouth.y > chin.y (face vertical order)")
    print("         pupil_left.x < pupil_right.x  AND  pupil_l.x + pupil_r.x ≈ 0 (face centered)")
    print("  face span (jaw 0..32) world x range: "
          f"[{mean_world[:33, 0].min().item():+.4f}, {mean_world[:33, 0].max().item():+.4f}]")
    print("  face span (all lmks)  world y range: "
          f"[{mean_world[:, 1].min().item():+.4f}, {mean_world[:, 1].max().item():+.4f}]")
    print("  face span (all lmks)  world z range: "
          f"[{mean_world[:, 2].min().item():+.4f}, {mean_world[:, 2].max().item():+.4f}]")
    print(f"[stage B] Saved mean_world_lmks_3d.pt + all_world_lmks_3d.pt to {WORKDIR}")


REGION_PRIORITY: Tuple[str, ...] = (
    "nose",
    "mouth",
    "left_eye_orbit",
    "right_eye_orbit",
    "brow",
    "left_cheek",
    "right_cheek",
    "chin",
    "forehead",
    "jaw_periphery_ears",
    "front_of_camera",
    "background_rear",
    "other",
)


def _aabb_world(coords: torch.Tensor) -> Tuple[float, float, float, float, float, float]:
    return (
        float(coords[:, 0].min()), float(coords[:, 0].max()),
        float(coords[:, 1].min()), float(coords[:, 1].max()),
        float(coords[:, 2].min()), float(coords[:, 2].max()),
    )


def _build_region_aabbs(mean_world: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """Build per-region AABBs in WORLD coords from grouped landmarks + extras.

    Margins:
      MARGIN_XY = 0.015  (~3% of box_warp=1)
      Z_BACK    = 0.12   (extend the region back into the cube to cover
                         subsurface voxels; the face surface is on the
                         camera-facing side of each region)
      Z_FRONT   = 0.03   (modest forward extension)
    """
    g = {name: mean_world[idxs] for name, idxs in WFLW_GROUPS.items()}
    margin_xy = 0.015
    z_back = 0.12
    z_front = 0.03

    def aabb_with_margin(pts: torch.Tensor) -> Dict[str, float]:
        x0, x1, y0, y1, z0, z1 = _aabb_world(pts)
        return {
            "x_min": x0 - margin_xy, "x_max": x1 + margin_xy,
            "y_min": y0 - margin_xy, "y_max": y1 + margin_xy,
            "z_min": z0 - z_back,    "z_max": z1 + z_front,
        }

    aabbs: Dict[str, Dict[str, float]] = {}

    # Tier 1: directly from WFLW groups.
    aabbs["nose"] = aabb_with_margin(torch.cat([g["nose_bridge"], g["nose_bottom"]], dim=0))
    aabbs["mouth"] = aabb_with_margin(torch.cat([g["mouth_outer"], g["mouth_inner"]], dim=0))
    aabbs["brow"] = aabb_with_margin(torch.cat([g["brow_left"], g["brow_right"]], dim=0))
    aabbs["left_eye_orbit"] = aabb_with_margin(g["eye_left"])
    aabbs["right_eye_orbit"] = aabb_with_margin(g["eye_right"])

    # Tier 2: derived.
    # Chin: central lower jaw_contour (indices 12-20 cover the chin curve).
    chin_pts = mean_world[list(range(12, 21))]
    aabbs["chin"] = aabb_with_margin(chin_pts)

    # Left cheek: x in [jaw_left_outer.x + 0.01, nose_left.x - 0.02];
    #             y in [chin.y + 0.02, eye_left_bottom.y - 0.01].
    nose_pts = torch.cat([g["nose_bridge"], g["nose_bottom"]], dim=0)
    chin_y_max = chin_pts[:, 1].max().item()
    eye_left_y_min = g["eye_left"][:, 1].min().item()
    nose_x_min = nose_pts[:, 0].min().item()
    nose_x_max = nose_pts[:, 0].max().item()
    jaw_left_pts = mean_world[list(range(2, 12))]  # left jaw side
    jaw_right_pts = mean_world[list(range(21, 31))]  # right jaw side
    face_z_min = float(torch.cat([nose_pts, chin_pts, g["eye_left"], g["eye_right"]],
                                 dim=0)[:, 2].min())
    face_z_max = float(torch.cat([nose_pts, chin_pts, g["eye_left"], g["eye_right"]],
                                 dim=0)[:, 2].max())

    aabbs["left_cheek"] = {
        "x_min": float(jaw_left_pts[:, 0].min()) + 0.01,
        "x_max": nose_x_min - 0.02,
        "y_min": chin_y_max + 0.02,
        "y_max": eye_left_y_min - 0.01,
        "z_min": face_z_min - 0.05,
        "z_max": face_z_max + 0.02,
    }
    eye_right_y_min = g["eye_right"][:, 1].min().item()
    aabbs["right_cheek"] = {
        "x_min": nose_x_max + 0.02,
        "x_max": float(jaw_right_pts[:, 0].max()) - 0.01,
        "y_min": chin_y_max + 0.02,
        "y_max": eye_right_y_min - 0.01,
        "z_min": face_z_min - 0.05,
        "z_max": face_z_max + 0.02,
    }

    # Forehead: extrapolate above brow by (brow.y_top - nose_tip.y) anatomical span.
    brow_combined = torch.cat([g["brow_left"], g["brow_right"]], dim=0)
    brow_y_top = brow_combined[:, 1].max().item()
    nose_tip_y = float(mean_world[54, 1])  # last nose-bridge point ~ tip-of-nose-bridge
    forehead_height = brow_y_top - nose_tip_y  # ~equals brow-to-nose-tip-vertical-distance
    aabbs["forehead"] = {
        "x_min": brow_combined[:, 0].min().item() - margin_xy,
        "x_max": brow_combined[:, 0].max().item() + margin_xy,
        "y_min": brow_y_top + 0.005,
        "y_max": brow_y_top + max(forehead_height, 0.10),
        "z_min": brow_combined[:, 2].min().item() - z_back,
        "z_max": brow_combined[:, 2].max().item() + z_front,
    }

    # Jaw periphery / ears: outer jaw_contour points (0-3 and 29-32) + outside the
    # face x-extent at face y-range.
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


def _world_to_cube_norm(coords_world: torch.Tensor, box_warp: float) -> torch.Tensor:
    """Map world coords [-box_warp/2, +box_warp/2] -> cube normalised [0, 1]."""
    return coords_world / box_warp + 0.5


def stage_c(device: torch.device) -> None:
    print(f"[stage C] Loading mean landmarks from {WORKDIR}")
    mean_world = torch.load(WORKDIR / "mean_world_lmks_3d.pt", map_location="cpu")  # (98, 3)

    print("[stage C] Loading generator + cube partition (untuned, psi=0.7)")
    da = gen_utils.load_generator(
        model_path=ORIG_PKL,
        truncation_psi=TRUNCATION_PSI,
        truncation_cutoff=TRUNCATION_CUTOFF,
        shape_res=SHAPE_RES,
    )
    G = da.G
    box_warp = float(G.rendering_kwargs.get("box_warp", 1.0))
    print(f"[stage C] box_warp = {box_warp}")

    mudc = finetuning_utils.MeshUtilsDataClass()
    pads = OmegaConf.load(gen_utils.STATIC_CONFIGS_DIR / "pads_vals_entire.yaml")
    samples, shape, tri_idx = mudc.get_samples_coordinates_from_pads_vals_dict(
        pads_vals=pads, G=G, shape_res=SHAPE_RES,
    )
    # samples: (N_total, 3) world coords for each cropped-cube voxel
    cube_shape = tuple(int(x) for x in shape[1:4])
    print(f"[stage C] Cropped cube shape: {cube_shape}")
    voxel_world = samples.detach().cpu().reshape(*cube_shape, 3)
    vx = voxel_world[..., 0]
    vy = voxel_world[..., 1]
    vz = voxel_world[..., 2]
    print(f"[stage C] Voxel world bounds: "
          f"x=[{vx.min():.3f}, {vx.max():.3f}], "
          f"y=[{vy.min():.3f}, {vy.max():.3f}], "
          f"z=[{vz.min():.3f}, {vz.max():.3f}]")

    aabbs = _build_region_aabbs(mean_world)
    for name, b in aabbs.items():
        print(f"[stage C] AABB {name:>20s}: "
              f"x[{b['x_min']:+.3f},{b['x_max']:+.3f}] "
              f"y[{b['y_min']:+.3f},{b['y_max']:+.3f}] "
              f"z[{b['z_min']:+.3f},{b['z_max']:+.3f}]")

    # Front-of-camera = forward of any landmark depth, anywhere in face x,y range.
    # Background_rear  = behind the rear-most landmark depth, anywhere in face x,y range.
    face_z_max_world = float(mean_world[:, 2].max())  # ~ nose tip
    face_z_min_world = float(mean_world[:, 2].min())  # ~ rear-most ear/jaw
    face_x_min_world = float(mean_world[:33, 0].min()) - 0.04
    face_x_max_world = float(mean_world[:33, 0].max()) + 0.04
    face_y_min_world = float(mean_world[:, 1].min()) - 0.03
    face_y_max_world = float(mean_world[:, 1].max()) + 0.08  # extend up to include forehead area
    in_face_xy = (
        (vx >= face_x_min_world) & (vx <= face_x_max_world) &
        (vy >= face_y_min_world) & (vy <= face_y_max_world)
    )

    # Build masks with priority claim — earlier regions win contested voxels.
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
            # Voxels in front of the front-most face surface landmark, restricted
            # to the face x,y rectangle (so we measure floater-style failures).
            raw = in_face_xy & (vz > face_z_max_world + 0.005)
        elif region == "background_rear":
            # Voxels behind the rearmost face landmark (the rear of the head /
            # cube background), restricted to face x,y so we can attribute it.
            raw = in_face_xy & (vz < face_z_min_world - 0.005)
        else:
            raise RuntimeError(f"No rule for region {region!r}")
        mask = raw & ~assigned
        masks[region] = mask
        assigned |= mask
    masks["other"] = ~assigned

    out_dir = _ensure_dir(MASKS_OUT_PT.parent)
    torch.save(
        {
            "masks": {k: v.bool().contiguous() for k, v in masks.items()},
            "region_priority": REGION_PRIORITY,
            "aabbs_world": aabbs,
            "cube_shape": cube_shape,
            "box_warp": box_warp,
            "mean_world_lmks": mean_world,
        },
        MASKS_OUT_PT,
    )

    # Metadata + voxel counts for sanity.
    counts = {k: int(v.sum().item()) for k, v in masks.items()}
    total = sum(counts.values())
    named = sum(c for k, c in counts.items() if k not in ("other", "front_of_camera", "background_rear"))
    metadata = {
        "cube_shape": list(cube_shape),
        "box_warp": box_warp,
        "voxel_counts": counts,
        "voxel_count_total": total,
        "named_face_fraction": named / total if total else 0.0,
        "aabbs_world": aabbs,
    }
    import json as _json
    with open(MASKS_OUT_JSON, "w") as f:
        _json.dump(metadata, f, indent=2)

    print(f"[stage C] Voxel counts per region:")
    for k, v in counts.items():
        pct = 100.0 * v / total if total else 0.0
        print(f"           {k:>20s}: {v:8d}  ({pct:5.2f}%)")
    print(f"[stage C] Named-face regions cover {metadata['named_face_fraction']*100:.2f}% "
          f"of the cube; diagnostic + other absorb the rest.")

    # Visualisation: three mid-slices of the cube coloured by region.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    region_to_id = {k: i + 1 for i, k in enumerate(REGION_PRIORITY)}
    label_volume = np.zeros(cube_shape, dtype=np.int32)
    for k, m in masks.items():
        label_volume[m.cpu().numpy()] = region_to_id[k]

    cmap = plt.get_cmap("tab20", len(REGION_PRIORITY) + 1)
    mid_x = cube_shape[0] // 2
    mid_y = cube_shape[1] // 2
    mid_z = cube_shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # With label_volume[mid_x, :, :].T → row=Z, col=Y, imshow gives image_x=Y
    # (col) and image_y=Z (row). So xlabel is the cube's second-axis name and
    # ylabel is the cube's third-axis name.
    panels = [
        (label_volume[mid_x, :, :], "mid-X slice (Y horizontal, Z vertical)", "y (face vertical: chin → forehead)", "z (depth: rear → camera)"),
        (label_volume[:, mid_y, :], "mid-Y slice (X horizontal, Z vertical)", "x (face horizontal: left → right)", "z (depth: rear → camera)"),
        (label_volume[:, :, mid_z], "mid-Z slice (X horizontal, Y vertical)", "x (face horizontal: left → right)", "y (face vertical: chin → forehead)"),
    ]
    for ax, (arr, title, xlabel, ylabel) in zip(axes, panels):
        im = ax.imshow(arr.T, origin="lower", cmap=cmap, vmin=0, vmax=len(REGION_PRIORITY))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color=cmap(region_to_id[k]), label=k) for k in REGION_PRIORITY]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.04))
    plt.suptitle(f"AW98 template partition (cube_shape={cube_shape}, "
                 f"named-face cover={metadata['named_face_fraction']*100:.1f}%)")
    plt.tight_layout()
    panel_path = out_dir / "region_slices.jpg"
    plt.savefig(panel_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage C] Saved region_slices.jpg to {panel_path}")
    print(f"[stage C] Saved masks to {MASKS_OUT_PT} and metadata to {MASKS_OUT_JSON}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["A", "B", "C"], default="A")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "A":
        stage_a(device)
    elif args.stage == "B":
        stage_b(device)
    elif args.stage == "C":
        stage_c(device)
    else:
        raise NotImplementedError(f"Stage {args.stage} not yet implemented")


if __name__ == "__main__":
    main()
