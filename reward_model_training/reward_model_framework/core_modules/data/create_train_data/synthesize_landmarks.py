"""
Extract 2D and depth-augmented 3D keypoints from synthesized RGB/depth pairs.
Outputs per-view AW98 (98 pt) landmarks and their depth-projected 3D counterparts.
Mediapipe/Dlib support was removed in favor of the AW98 pipeline currently used
by the loaders (triple_rgb_lmks_98 2d, and 3D depth-projected).

Usage:
  RWD_DATA_DIR=/path/to/eg3dredo_data \\
  python synthesize_landmarks.py --seeds_csv rankedseedsall.csv --views 0 1 2
"""

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple, Union

import autoroot  # noqa: F401

try:
    import cv2
except Exception:
    cv2 = None
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image, ImageDraw

from core_modules.data.create_train_data import generation_utils as gen_utils
from core_modules.data.misc_small_utils import assemble_triple_dmap, ddir_func


CANONICAL_VIEW_IDX = 1


def load_depth(seed: int, view: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load depth map for a given view and return it plus its resolution."""
    depth = assemble_triple_dmap(seed, ddir=ddir_func(seed))

    # Select view-specific depth when available, otherwise fall back to canonical.
    if isinstance(depth, (list, tuple)) and len(depth) > view:
        depth_tensor = depth[view]
    elif hasattr(depth, "ndim") and depth.ndim >= 3 and depth.shape[0] > view:
        depth_tensor = depth[view]
    elif isinstance(depth, (list, tuple)) and len(depth) > CANONICAL_VIEW_IDX:
        depth_tensor = depth[CANONICAL_VIEW_IDX]
    elif hasattr(depth, "ndim") and depth.ndim >= 3 and depth.shape[0] > CANONICAL_VIEW_IDX:
        depth_tensor = depth[CANONICAL_VIEW_IDX]
    else:
        depth_tensor = depth

    if hasattr(depth_tensor, "ndim") and depth_tensor.ndim == 4:
        depth_tensor = depth_tensor.squeeze(0)
    depth_tensor = depth_tensor.squeeze()
    return depth_tensor, depth_tensor.shape[-2:]


def _prepare_img_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """Convert PIL image to CHW float tensor on the requested device."""
    arr = torch.as_tensor(np.array(img), device=device, dtype=torch.float32)
    return arr.permute(2, 0, 1)  # C,H,W


def _pipnet_detect_torch(model, img_chw: torch.Tensor, orig_hw: Tuple[int, int]) -> torch.Tensor:
    """Run torchlm PIPNet decoding fully in torch to avoid CPU/GPU hops."""
    model.eval()
    device = img_chw.device
    height, width = orig_hw
    target = getattr(model, "input_size", 256)
    img = img_chw.unsqueeze(0)  # 1,C,H,W
    img = F.interpolate(img, size=(target, target), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img = (img / 255.0 - mean) / std

    with torch.no_grad():
        outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = model(img)

    tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()
    outputs_cls = outputs_cls.view(tmp_batch * tmp_channel, -1)
    max_ids = torch.argmax(outputs_cls, 1, keepdim=True)
    max_ids_nb = max_ids.repeat(1, model.num_nb).view(-1, 1)

    outputs_x = outputs_x.view(tmp_batch * tmp_channel, -1)
    outputs_y = outputs_y.view(tmp_batch * tmp_channel, -1)
    outputs_x_select = torch.gather(outputs_x, 1, max_ids).squeeze(1)
    outputs_y_select = torch.gather(outputs_y, 1, max_ids).squeeze(1)

    outputs_nb_x = outputs_nb_x.view(tmp_batch * model.num_nb * tmp_channel, -1)
    outputs_nb_y = outputs_nb_y.view(tmp_batch * model.num_nb * tmp_channel, -1)
    outputs_nb_x_select = torch.gather(outputs_nb_x, 1, max_ids_nb).squeeze(1).view(-1, model.num_nb)
    outputs_nb_y_select = torch.gather(outputs_nb_y, 1, max_ids_nb).squeeze(1).view(-1, model.num_nb)

    lms_pred_x = (max_ids % tmp_width).view(-1, 1).float() + outputs_x_select.view(-1, 1)
    lms_pred_y = torch.floor(max_ids / tmp_width).view(-1, 1).float() + outputs_y_select.view(-1, 1)
    stride_ratio = float(model.input_size) / model.net_stride
    lms_pred_x /= stride_ratio
    lms_pred_y /= stride_ratio

    lms_pred_nb_x = (max_ids % tmp_width).view(-1, 1).float() + outputs_nb_x_select
    lms_pred_nb_y = torch.floor(max_ids / tmp_width).view(-1, 1).float() + outputs_nb_y_select
    lms_pred_nb_x = lms_pred_nb_x.view(-1, model.num_nb) / stride_ratio
    lms_pred_nb_y = lms_pred_nb_y.view(-1, model.num_nb) / stride_ratio

    tmp_nb_x = lms_pred_nb_x[model.reverse_index1, model.reverse_index2].view(model.num_lms, model.max_len)
    tmp_nb_y = lms_pred_nb_y[model.reverse_index1, model.reverse_index2].view(model.num_lms, model.max_len)
    tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
    tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
    lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1)

    lms_pred_merge[:, 0] *= float(width)
    lms_pred_merge[:, 1] *= float(height)
    return lms_pred_merge


def sample_depth_at_points(depth: torch.Tensor, coords: torch.Tensor, rgb_size: Tuple[int, int]) -> torch.Tensor:
    """Append depth value from depth map to 2D coords (expects coords in RGB pixel space)."""
    h_rgb, w_rgb = rgb_size
    h_d, w_d = depth.shape[-2:]
    x = (coords[:, 0] / w_rgb) * w_d
    y = (coords[:, 1] / h_rgb) * h_d
    x = torch.clamp(x, 0, w_d - 1)
    y = torch.clamp(y, 0, h_d - 1)
    idx_x = torch.round(x).long()
    idx_y = torch.round(y).long()
    z = depth[idx_y, idx_x]
    return torch.stack([coords[:, 0], coords[:, 1], z], dim=1)


def get_aw98_landmarks(img_chw: torch.Tensor) -> torch.Tensor:
    """Best-effort AW98 using torchlm; returns empty if unavailable."""
    try:
        from core_modules.utils.awloss_utils_AM import return_awloss_model_98  # resnet50

        model = return_awloss_model_98().to(img_chw.device)
        h_img, w_img = img_chw.shape[1:]
        coords = _pipnet_detect_torch(model, img_chw, (h_img, w_img))
        if coords is None or coords.numel() == 0:
            return torch.empty((0, 3), device=img_chw.device)
        z = torch.zeros((coords.shape[0], 1), device=img_chw.device, dtype=coords.dtype)
        return torch.cat([coords, z], dim=1)
    except Exception as exc:
        print(f"[WARN] AW98 landmark extraction failed: {exc}")
        return torch.empty((0, 3), device=img_chw.device)


def predict_98_landmarks(img: Union[Image.Image, torch.Tensor], device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Convenience wrapper to predict AW98 landmarks from a PIL image or CHW tensor.
    Returns (coords_xyz, (h_rgb, w_rgb)) to make downstream depth sampling easier.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(img, Image.Image):
        img_chw = _prepare_img_tensor(img, device)
    elif torch.is_tensor(img):
        img_chw = img.to(device)
        if img_chw.ndim == 4 and img_chw.shape[0] == 1:
            img_chw = img_chw.squeeze(0)
        if img_chw.ndim != 3 or img_chw.shape[0] != 3:
            raise ValueError(f"Expected CHW image tensor, got shape {tuple(img_chw.shape)}")
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    _, h_rgb, w_rgb = img_chw.shape
    return get_aw98_landmarks(img_chw), (h_rgb, w_rgb)


def build_landmark_dicts(aw98_xyz: torch.Tensor, depth: Optional[torch.Tensor], rgb_size: Tuple[int, int], view: int) -> Tuple[dict, dict]:
    """Assemble the 2D/3D landmark dictionaries used by downstream loaders."""
    lmks2d = {}
    if aw98_xyz.numel():
        coords2d = aw98_xyz[:, :2].detach()
        lmks2d["triple_rgb_lmks_98"] = coords2d

    lmks3d = {}
    if depth is not None and lmks2d:
        for name, coords2d in lmks2d.items():
            depth_on_device = depth.to(coords2d.device, non_blocking=True) if hasattr(depth, "to") else depth
            coords3d = sample_depth_at_points(depth_on_device, coords2d[:, :2], rgb_size)
            lmks3d[name + "_3d"] = coords3d

    return lmks2d, lmks3d


def save_landmarks(seed: int, view: int, lmks: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, v in lmks.items():
        is_empty = v.numel() == 0 if isinstance(v, torch.Tensor) else v.size == 0
        if is_empty:
            continue

        tensor = v.detach().cpu() if isinstance(v, torch.Tensor) else torch.from_numpy(v)
        torch.save(tensor, out_dir / f"{k}_s_{seed}_{view}.pt")


def _parse_seed_from_rgb_path(path: Path) -> Optional[int]:
    match = re.search(r"triple_rgb_s_(\d+)_\d+\.jpg$", path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _parse_seed_from_dmap_path(path: Path) -> Optional[int]:
    match = re.search(r"triple_dmap_s_(\d+)\.pt$", path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def find_candidate_seeds(rgb_dir: Path) -> Set[int]:
    """Find seeds that have both RGB JPGs (any view) and depth maps."""
    rgb_dir = Path(rgb_dir)
    rgb_seeds: Set[int] = set()
    for jpg in rgb_dir.glob("triple_rgb_s_*_*.jpg"):
        seed = _parse_seed_from_rgb_path(jpg)
        if seed is not None:
            rgb_seeds.add(seed)

    dmap_seeds: Set[int] = set()
    for pt in rgb_dir.glob("triple_dmap_s_*.pt"):
        seed = _parse_seed_from_dmap_path(pt)
        if seed is not None:
            dmap_seeds.add(seed)

    return rgb_seeds.intersection(dmap_seeds)


def has_all_landmarks(seed: int, out_dir: Path, views: Iterable[int]) -> bool:
    """Check if all requested 2D landmark files already exist for this seed."""
    out_dir = Path(out_dir)
    return all((out_dir / f"triple_rgb_lmks_98_s_{seed}_{view}.pt").exists() for view in views)


def process_seed(seed: int, views: Iterable[int], rgb_dir: Path, out_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for view in views:
        rgb_path = rgb_dir / f"triple_rgb_s_{seed}_{view}.jpg"
        if not rgb_path.exists():
            continue
        img = Image.open(rgb_path).convert("RGB")
        depth, _ = load_depth(seed, view)
        depth = depth.to(device)

        aw98_xyz, rgb_size = predict_98_landmarks(img, device=device)

        lmks2d, lmks3d = build_landmark_dicts(aw98_xyz, depth, rgb_size, view)

        save_landmarks(seed, view, {**lmks2d, **lmks3d}, out_dir)
        # breakpoint()
        # print(f"[OK] seed={seed} view={view} saved {list(lmks2d.keys()) + list(lmks3d.keys())}")


def _overlay_landmarks_on_image(img: Image.Image, coords: Union[np.ndarray, torch.Tensor], n: int = 98) -> Image.Image:
    """Draw small red circles on the image for quick landmark sanity checks."""
    if torch.is_tensor(coords):
        coords = coords.detach().cpu().numpy()
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Expected Nx2 landmark array, got shape {coords.shape}")

    coords = coords[: min(n, len(coords))]
    marker_size = 2 if img.size[1] > 128 else 1
    if cv2 is not None and hasattr(cv2, "circle"):
        img_np = np.array(img)
        for x, y in coords:
            cv2.circle(img_np, (int(x), int(y)), marker_size, (0, 0, 255), -1)
        return Image.fromarray(img_np)

    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    for x, y in coords:
        draw.ellipse(
            (x - marker_size, y - marker_size, x + marker_size, y + marker_size),
            fill=(255, 0, 0),
        )
    return img_copy


def save_landmark_overlay_for_seed(
    seed: int,
    n: int = 98,
    rgb_dir: Optional[Path] = None,
    landmarks_dir: Optional[Path] = None,
    output_dir: Path = gen_utils.DEFAULT_CHECK_DIR,
) -> Path:
    """
    Load canonical view RGB + 2D landmarks for a seed and save an overlay for inspection.
    """
    rgb_dir = Path(rgb_dir) if rgb_dir is not None else Path(ddir_func(seed))
    landmarks_dir = Path(landmarks_dir) if landmarks_dir is not None else rgb_dir
    rgb_path = rgb_dir / f"triple_rgb_s_{seed}_{CANONICAL_VIEW_IDX}.jpg"
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB for seed {seed} not found at {rgb_path}")

    landmark_candidates = [
        landmarks_dir / f"triple_rgb_lmks_98_s_{seed}_{CANONICAL_VIEW_IDX}.pt",
    ]
    lmk_path = next((p for p in landmark_candidates if p.exists()), None)
    if lmk_path is None:
        raise FileNotFoundError(f"No landmarks found for seed {seed}. Tried: {', '.join(str(p) for p in landmark_candidates)}")

    coords = torch.load(lmk_path, map_location="cpu")
    img = Image.open(rgb_path).convert("RGB")
    overlaid = _overlay_landmarks_on_image(img, coords, n=n)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"seed_{seed}_view{CANONICAL_VIEW_IDX}_lmks_{min(n, len(coords))}.jpg"
    overlaid.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds_csv", type=Path, default=None, help="Optional CSV filter for seeds. If omitted, seeds are discovered from available data.")
    parser.add_argument("--views", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rgb_dir", type=Path, default=None, help="Directory containing triple_rgb_s_* images. Defaults to RWD_DATA_DIR.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory for landmark pt files. Defaults to rgb_dir.")
    parser.add_argument(
        "--include_hiq",
        action="store_true",
        help="Also process the high-quality seed range (default 100000-101000) used by triple_rgb/dmap generation scripts.",
    )
    parser.add_argument(
        "--hiq_range",
        type=int,
        nargs=2,
        default=[100000, 101000],
        help="Start/end (exclusive) for the high-quality seeds when --include_hiq is set.",
    )
    args = parser.parse_args()

    rgb_dir = args.rgb_dir or Path(ddir_func(0))
    out_dir = args.out_dir or rgb_dir

    # Discover seeds from existing RGB/DMap data.
    candidate_seeds = find_candidate_seeds(Path(rgb_dir))

    # Optional: filter via CSV if provided.
    if args.seeds_csv is not None:
        try:
            df = pd.read_csv(args.seeds_csv, index_col=0)
            seeds_from_csv = []
            for col in df.columns:
                seeds_from_csv.extend(df[col].dropna().astype(int).tolist())
        except Exception:
            seeds_from_csv = torch.tensor(np.loadtxt(args.seeds_csv, delimiter=",", dtype=int)).flatten().tolist()
        candidate_seeds = {s for s in candidate_seeds if s in set(seeds_from_csv)}

    if args.include_hiq:
        hiq_start, hiq_end = args.hiq_range
        hiq_candidates = set(s for s in range(hiq_start, hiq_end) if (Path(rgb_dir) / f"triple_rgb_s_{s}_{CANONICAL_VIEW_IDX}.jpg").exists() and (Path(rgb_dir) / f"triple_dmap_s_{s}.pt").exists())
        candidate_seeds = candidate_seeds.union(hiq_candidates)

    seeds = sorted(candidate_seeds)

    for seed in tqdm.tqdm(seeds):
        if has_all_landmarks(seed, out_dir, args.views):
            continue
        process_seed(seed, args.views, rgb_dir=Path(rgb_dir), out_dir=Path(out_dir))
        # print(f"[PROGRESS] completed seed {seed}")


if __name__ == "__main__":
    main()
