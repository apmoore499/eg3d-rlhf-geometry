"""
Shared helpers for dataset I/O, cameras, depth-to-point-cloud conversion, and landmark masks.
Existing helpers in misc_small_utils, small_data_tools, and rwd_model_utils import from here to avoid duplication.
"""

import os
from pathlib import Path
from typing import Tuple

import cv2
import imageio.v3 as iio
import numpy as np
import torch


# ---------- Path builders ----------
def _ensure_landmark_tensor(tensor: torch.Tensor, coord_dim: int) -> torch.Tensor:
    """
    Normalise saved landmark tensors into an expected (N, coord_dim) shape.
    Some legacy saves include an extra leading dim; we strip/reshape them here.
    """
    if tensor.ndim == 1:
        tensor = tensor.view(-1, coord_dim)
    elif tensor.ndim == 3 and tensor.shape[0] == 1 and tensor.shape[-1] == coord_dim:
        tensor = tensor.squeeze(0)

    if tensor.ndim != 2 or tensor.shape[1] != coord_dim:
        raise ValueError(f"Unexpected landmark tensor shape {tuple(tensor.shape)} (wanted (*, {coord_dim}))")

    return tensor


def create_pt_fn(ddir: str, ot: str, seed: int) -> str:
    return os.path.join(ddir, f"{ot}_s_{seed}.pt")


def seed_func_default(s):
    return s


def ddir_func(query_val=None):
    # Priority: env var -> hydra cfg (paths.rwd_data_dir) -> legacy default.
    env_val = os.environ.get("RWD_DATA_DIR")
    if env_val:
        return env_val

    try:
        from core_modules.utils.config_store import ConfigStore

        cfg_store = ConfigStore.instance()
        if cfg_store and getattr(cfg_store, "cfg", None):
            paths = getattr(cfg_store.cfg, "paths", None)
            if paths and getattr(paths, "rwd_data_dir", None):
                return paths.rwd_data_dir
    except Exception:
        pass

    return str(Path.home() / "Documents" / "eg3dredo_data")


# ---------- Camera utilities ----------
def get_canonical_dmap_cams():
    from core_modules.utils import camera_utils

    return camera_utils.get_canonical_dmap_cams_for_rlhf()


def load_vertex_sampling_weights_dmap_128():
    from core_modules.utils import camera_utils

    cfg_dir = camera_utils.get_static_configs_dir()
    prob_sampling_faces = torch.load(cfg_dir / "weight_sampling_for_canon_pcd_modules_depthmap_128.pt", map_location="cpu")
    return prob_sampling_faces


# ---------- Depth / point-cloud ----------
def imd_to_xyz_with_radius_cutoff(
    image_depth: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    neural_rendering_resolution: int,
    radius_cutoff=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    final_dim = neural_rendering_resolution * neural_rendering_resolution
    imd_list = image_depth.reshape(final_dim)
    imd_greater = torch.where(imd_list <= torch.max(imd_list))
    if radius_cutoff is not None:
        imd_greater = torch.where(imd_list <= radius_cutoff)

    device = torch.device("cpu") if ray_origins.get_device() == -1 else torch.device(f"cuda:{ray_origins.get_device()}")
    imd = image_depth.reshape(1, final_dim).unsqueeze(2).expand(1, final_dim, 3).to(device)
    retval = ray_origins + imd * ray_directions
    return (retval, imd_greater)


def rescale_im_dmp_for_lmk(dmap: np.ndarray) -> np.ndarray:
    rmin = 2.25
    rmax = 3.3
    dm_min = -1.0
    dmap = (((dmap - rmin) / (rmax - rmin)) * 2) - 1
    dmap[dmap < dm_min] = dm_min
    dmap[dmap > 1.0] = 1.0
    return dmap


# ---------- Landmark + RGB assemblers ----------
def _load_triple_rgb_pt(seed: int, ddir: str) -> torch.Tensor:
    """
    Load triple RGB tensor from pt (preferred) and normalise to match legacy jpg path.
    Returns shape (3,1,3,H,W) in [-1,1].
    """
    candidates = [
        os.path.join(ddir, f"triple_rgb_s_{seed}.pt"),
        os.path.join(ddir, f"triple_rgb_s_{seed}_tensor.pt"),
    ]
    fn = next((p for p in candidates if os.path.exists(p)), None)
    if fn is None:
        raise FileNotFoundError(f"No triple RGB pt found for seed {seed}; tried {candidates}")

    rgb = torch.load(fn, map_location=torch.device("cpu"))
    if isinstance(rgb, (list, tuple)):
        rgb = torch.stack([torch.as_tensor(r) for r in rgb])
    else:
        rgb = torch.as_tensor(rgb)

    # Squeeze leading singleton dims
    while rgb.ndim > 4 and rgb.shape[0] == 1:
        rgb = rgb.squeeze(0)

    # Expect (3, C, H, W) or (3, H, W, C)
    if rgb.ndim == 4 and rgb.shape[0] == 3 and rgb.shape[1] != 3 and rgb.shape[-1] == 3:
        rgb = rgb.permute(0, 3, 1, 2)
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    if rgb.shape[0] != 3 or rgb.shape[1] != 3:
        raise ValueError(f"Unexpected triple_rgb tensor shape {tuple(rgb.shape)} from {fn}")

    rgb = rgb.to(torch.float32)
    if rgb.max() > 1.5 or rgb.min() < -1.5:
        rgb = (rgb / 127.5) - 1.0  # assume 0..255

    return rgb.unsqueeze(1)


def assemble_triple_lmks(seed: int, ddir: str):
    fns = [os.path.join(ddir, f"triple_rgb_lmks_98_s_{seed}_{k}.pt") for k in range(3)]
    missing = [f for f in fns if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing triple_rgb_lmks_98 landmark files: {missing}")

    lmks = [torch.load(fn, map_location=torch.device("cpu")) for fn in fns]
    lmks = [_ensure_landmark_tensor(t, coord_dim=2) for t in lmks]
    return torch.stack(lmks, dim=0)


def assemble_single_lmks(seed: int, ddir: str, view: int = 1):
    candidates = [
        os.path.join(ddir, f"canonical_rgb_lmks_98_s_{seed}_{view}.pt"),
        os.path.join(ddir, f"triple_rgb_lmks_98_s_{seed}_{view}.pt"),
    ]
    fn = next((p for p in candidates if os.path.exists(p)), None)
    if fn is None:
        raise FileNotFoundError(f"No canonical landmark file found for seed {seed}; tried {candidates}")

    tensor = torch.load(fn, map_location=torch.device("cpu"))
    tensor = _ensure_landmark_tensor(tensor, coord_dim=2)
    return tensor.unsqueeze(0)


def assemble_triple_dmap(seed: int, ddir: str):
    fn = os.path.join(ddir, f"triple_dmap_s_{seed}.pt")
    return torch.load(fn, map_location=torch.device("cpu"))


def assemble_single_dmap(seed: int, ddir: str):
    fn = os.path.join(ddir, f"triple_dmap_s_{seed}.pt")
    return torch.load(fn, map_location=torch.device("cpu"))[1]


def assemble_mediapipe_468_lmks(seed: int, ddir: str, dmap_res: int = 256):
    fn = os.path.join(ddir, f"mediapipe_468_lmk_s_{seed}_1.pt")
    lmks = torch.load(fn, map_location=torch.device("cpu"))
    lmks_xy = (lmks[:, :2] * dmap_res).to(torch.int32)
    return lmks_xy


def assemble_single_rgb(seed: int, ddir: str):
    triple = _load_triple_rgb_pt(seed, ddir)
    rgb = triple[1].unsqueeze(0)  # canonical view
    return rgb


def assemble_triple_rgb(seed: int, ddir: str):
    return _load_triple_rgb_pt(seed, ddir)


# ---------- Landmark masks ----------
def return_lmks_mask_aw98_no_edit(s):
    # Returns the AW98 2D landmark pixel coords (98, 2); despite the name it is
    # not a mask and does not use the depth map.
    lmks = assemble_single_lmks(ddir=ddir_func(s), seed=s)
    return lmks.squeeze(0)


def return_lmks_mask_aw98(s, radius=9, return_im=False, randomize_sel=False):
    fn_depth = create_pt_fn(ddir=ddir_func(s), ot="triple_dmap", seed=s)
    tdm = torch.load(fn_depth, map_location=torch.device("cpu"))[1].unsqueeze(0)
    tdm = torch.nn.functional.interpolate(tdm, size=(256, 256)).squeeze()
    lmks = assemble_single_lmks(ddir=ddir_func(s), seed=s)
    lmks = lmks.cpu().numpy().squeeze().astype(np.int32)
    dmp = tdm.squeeze(0)[:, :, None].expand(256, 256, 3).cpu().numpy()

    dmp = rescale_im_dmp_for_lmk(dmp)
    dmp = dmp / 2 + 0.5
    dmp = (dmp * 255).astype(np.uint8)
    ocvim = cv2.cvtColor(dmp, cv2.COLOR_RGB2BGR)

    if randomize_sel:
        iterable = lmks[randomize_sel]
    else:
        iterable = lmks

    for l in iterable:
        x, y = l
        ocvim = cv2.circle(ocvim, (x, y), radius=radius, color=(0, 0, 255), thickness=-1)

    mask = torch.from_numpy(cv2.cvtColor(ocvim, cv2.COLOR_BGR2RGB))
    mask = torch.nn.functional.interpolate(mask.permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode="nearest").squeeze().permute(1, 2, 0).numpy()

    retmask = np.empty_like(mask).astype(np.bool_)
    retmask.fill(False)
    retmask[mask == [0, 0, 255]] = True

    if return_im:
        return (retmask, mask)
    return retmask


def return_lmks_mask_mediapipe_468(s, radius=9, return_im=False):
    fn_depth = create_pt_fn(ddir=ddir_func(s), ot="triple_dmap", seed=s)
    tdm = torch.load(fn_depth, map_location=torch.device("cpu"))[1].unsqueeze(0)
    tdm = torch.nn.functional.interpolate(tdm, size=(256, 256)).squeeze()
    lmks = assemble_mediapipe_468_lmks(seed=s, ddir=ddir_func(s))
    lmks = lmks.cpu().numpy().squeeze().astype(np.int32)
    dmp = tdm.squeeze(0)[:, :, None].expand(256, 256, 3).cpu().numpy()

    dmp = rescale_im_dmp_for_lmk(dmp)
    dmp = dmp / 2 + 0.5
    dmp = (dmp * 255).astype(np.uint8)
    ocvim = cv2.cvtColor(dmp, cv2.COLOR_RGB2BGR)

    for l in lmks:
        x, y = l
        ocvim = cv2.circle(ocvim, (x, y), radius=radius, color=(0, 0, 255), thickness=-1)
    mask = cv2.cvtColor(ocvim, cv2.COLOR_BGR2RGB)

    retmask = np.empty_like(dmp).astype(np.bool_)
    retmask.fill(False)
    retmask[mask == [0, 0, 255]] = True
    if return_im:
        return (retmask, mask)
    return retmask


def return_only_five_lmks_mask(s, radius=9, return_im=False):
    fn_depth = create_pt_fn(ddir=ddir_func(s), ot="triple_dmap", seed=s)
    tdm = torch.load(fn_depth, map_location=torch.device("cpu"))[1].unsqueeze(0)
    tdm = torch.nn.functional.interpolate(tdm, size=(256, 256)).squeeze()
    lmks = assemble_single_lmks(ddir=ddir_func(s), seed=s)
    lmks = lmks.cpu().numpy().squeeze().astype(np.int32)
    dmp = tdm.squeeze(0)[:, :, None].expand(256, 256, 3).cpu().numpy()

    dmp = rescale_im_dmp_for_lmk(dmp)
    dmp = dmp / 2 + 0.5
    dmp = (dmp * 255).astype(np.uint8)
    ocvim = cv2.cvtColor(dmp, cv2.COLOR_RGB2BGR)

    lmks = lmks[[96, 97, 54, 76, 82], :]
    for l in lmks:
        x, y = l
        ocvim = cv2.circle(ocvim, (x, y), radius=radius, color=(0, 0, 255), thickness=-1)
    mask = cv2.cvtColor(ocvim, cv2.COLOR_BGR2RGB)

    retmask = np.empty_like(dmp).astype(np.bool_)
    retmask.fill(False)
    retmask[mask == [0, 0, 255]] = True
    if return_im:
        return (retmask, mask)
    return retmask
