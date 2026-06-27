"""Shared helpers for the dataset generation scripts."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import autoroot  # noqa: F401
import numpy as np
import pandas as pd
import torch
from PIL import Image

import dnnlib
from eg3d import legacy
from eg3d.camera_utils import FOV_to_intrinsics, LookAtPoseSampler
from eg3d.training.triplane import TriPlaneGenerator
from eg3d.torch_utils import misc

# Project locations

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[5]))
DEFAULT_MODEL_PATH = PROJECT_ROOT / "pkl_pt/eg3d_1/ffhq512-128.pkl"
DEFAULT_SAVE_DIR = Path(os.environ.get("E3D_RLHF_SAVE_DIR", str(PROJECT_ROOT / "generated_data")))
DEFAULT_CHECK_DIR = Path(os.environ.get("E3D_RLHF_CHECK_DIR", str(DEFAULT_SAVE_DIR / "checking")))


def _default_static_configs_dir() -> Path:
    if PROJECT_ROOT.name == "reward_model_framework":
        return PROJECT_ROOT.parent / "static_configs"
    return PROJECT_ROOT / "reward_model_training" / "static_configs"


STATIC_CONFIGS_DIR = Path(os.environ.get("STATIC_CONFIGS_DIR", _default_static_configs_dir()))


class DArgs:
    """Lightweight holder for generator/config attributes with dot syntax."""

    def __init__(self):
        pass

    def set_network_pkl(self, pkl_str: str):
        self.network_pkl = pkl_str
        return self

    def set_tail(self, use_fat_tail: bool):
        self.use_fat_tail = use_fat_tail
        return self

    def set_nrs_single_dmap(self, nrs: int):
        self.nrs_single_dmap = nrs
        return self

    def set_nrs_triple_dmap(self, nrs: int):
        self.nrs_triple_dmap = nrs
        return self


def set_defaults(da: DArgs) -> DArgs:
    da.shape_res = 512
    da.fovdeg = 18.837
    da.reload_modules = False
    da.nsamps = 70
    da.use_fat_tail = False
    da.dmap_cam_radius = 2.7
    da.compose_images = True
    da.synthesise_im_and_dmap_and_pcd = True
    da.visualise_chimerax = True
    da.compose_mesh_and_im = True
    da.export_style_code = True
    da.export_mrc_and_ply = True
    da.delete_mrc_file = True
    da.export_obj = False
    da.delete_ply_file = False
    da.delete_images_after_compose = False
    da.export_draco = False
    da.nrs_single_dmap = 128
    da.nrs_triple_dmap = 128
    da.device = torch.device("cuda")
    dmap_rot = 1.0
    da.dmap_angles = [dmap_rot, 0, -dmap_rot]
    da.seed = 1
    da.truncation_psi = 0.7
    da.truncation_cutoff = 14
    da.level = 10
    da.noise_mode = "const"
    da.script_abs_path_mesh = str(PROJECT_ROOT / "eg3d" / "visualise_sdf_chimerax.py")
    da.angle_p_fixed_rgb = -0.2
    da.angles_y_rgb = [-0.4, 0, 0.4]
    da.single_angle_y_fixed_rgb = 0.0
    da.get_imrgb_single_256 = False
    da.get_imrgb_single_512 = True
    return da


def load_pkl_G(da: DArgs):
    """Load EG3D generator from pkl path on da.network_pkl."""
    print(f'Loading networks from "{da.network_pkl}"...')
    try:
        with dnnlib.util.open_url(da.network_pkl) as f:
            G = legacy.load_network_pkl(f)["G_ema"].to(da.device).eval()  # type: ignore
    except Exception:
        G = torch.load(da.network_pkl).to(da.device).eval()

    if da.reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(da.device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new.eval()

    return G


@dataclass
class TripleDmapCams:
    camera_params: torch.Tensor
    conditioning_params: torch.Tensor


def get_triple_dmap_cams(da: DArgs) -> TripleDmapCams:
    """3-view depth map camera + conditioning params."""
    y0, y1, y2 = [1.0, 0.0, -1.0]
    angle_p = 0
    list_camera_params = []
    list_conditioning_params = []
    device = torch.device("cuda")
    G = da.G

    cam_radius = G.rendering_kwargs.get("avg_camera_radius", 2.7)
    intrinsics = FOV_to_intrinsics(da.fovdeg, device=device)

    for angle_y, angle_p in [(y0, angle_p), (y1, angle_p), (y2, angle_p)]:
        cam_pivot = torch.tensor(G.rendering_kwargs.get("avg_camera_pivot", [0, 0, 0]), device=device)
        cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        list_camera_params.append(camera_params)
        list_conditioning_params.append(conditioning_params)

    camera_params = torch.cat(list_camera_params, dim=0)
    conditioning_params = torch.cat(list_conditioning_params, dim=0)
    return TripleDmapCams(camera_params=camera_params, conditioning_params=conditioning_params)


@dataclass
class SingleDmapCam:
    camera_params: torch.Tensor
    conditioning_params: torch.Tensor


def get_single_dmap_cam(da: DArgs) -> SingleDmapCam:
    """Canonical depth cam (2nd element of triple)."""
    angle_y = 0.0
    angle_p = 0
    device = torch.device("cuda")
    G = da.G

    cam_radius = G.rendering_kwargs.get("avg_camera_radius", 2.7)
    intrinsics = FOV_to_intrinsics(da.fovdeg, device=device)

    cam_pivot = torch.tensor(G.rendering_kwargs.get("avg_camera_pivot", [0, 0, 0]), device=device)
    cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    return SingleDmapCam(camera_params=camera_params, conditioning_params=conditioning_params)


def get_triple_img_cams(da: DArgs):
    """3-view RGB camera + conditioning params."""
    y0, y1, y2 = [-0.4, 0.0, 0.4]
    angle_p = -0.2
    list_camera_params = []
    list_conditioning_params = []
    device = torch.device("cuda")
    G = da.G

    cam_radius = G.rendering_kwargs.get("avg_camera_radius", 2.7)
    intrinsics = FOV_to_intrinsics(da.fovdeg, device=device)

    for angle_y, angle_p in [(y0, angle_p), (y1, angle_p), (y2, angle_p)]:
        cam_pivot = torch.tensor(G.rendering_kwargs.get("avg_camera_pivot", [0, 0, 0]), device=device)
        cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        list_camera_params.append(camera_params)
        list_conditioning_params.append(conditioning_params)

    camera_params = torch.cat(list_camera_params, dim=0)
    conditioning_params = torch.cat(list_conditioning_params, dim=0)
    return (camera_params, conditioning_params)


def load_generator(model_path: Path = DEFAULT_MODEL_PATH, truncation_psi: float = 1.0, truncation_cutoff: int = 14, shape_res: Optional[int] = None):
    """Return a configured EG3D generator."""
    da = DArgs()
    da = set_defaults(da)
    da.set_network_pkl(str(model_path))
    da.G = load_pkl_G(da).cuda()
    da.truncation_psi = truncation_psi
    da.truncation_cutoff = truncation_cutoff
    if shape_res is not None:
        da.shape_res = shape_res
    ensure_static_configs(da)
    return da


def ensure_static_configs(da: DArgs):
    """Create camera static config files if they are missing."""
    STATIC_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    triple_dmap_cams_fn = STATIC_CONFIGS_DIR / "triple_dmap_cameras.pt"
    triple_dmap_conds_fn = STATIC_CONFIGS_DIR / "triple_dmap_conditioning.pt"
    single_dmap_cams_fn = STATIC_CONFIGS_DIR / "single_dmap_cameras.pt"
    single_dmap_conds_fn = STATIC_CONFIGS_DIR / "single_dmap_conditioning.pt"

    triple_img_cams_fn = STATIC_CONFIGS_DIR / "triple_img_cameras.pt"
    triple_img_conds_fn = STATIC_CONFIGS_DIR / "triple_img_conditioning.pt"
    single_img_cams_fn = STATIC_CONFIGS_DIR / "single_img_cameras.pt"
    single_img_conds_fn = STATIC_CONFIGS_DIR / "single_img_conditioning.pt"

    # Depth cams
    if not triple_dmap_cams_fn.exists() or not triple_dmap_conds_fn.exists():
        tdca = get_triple_dmap_cams(da)
        torch.save(tdca.camera_params, triple_dmap_cams_fn)
        torch.save(tdca.conditioning_params, triple_dmap_conds_fn)

    if not single_dmap_cams_fn.exists() or not single_dmap_conds_fn.exists():
        tdca = get_triple_dmap_cams(da)
        torch.save(tdca.camera_params[1].unsqueeze(0), single_dmap_cams_fn)
        torch.save(tdca.conditioning_params[1].unsqueeze(0), single_dmap_conds_fn)

    # RGB cams
    if not triple_img_cams_fn.exists() or not triple_img_conds_fn.exists():
        cams, conds = get_triple_img_cams(da)
        torch.save(cams, triple_img_cams_fn)
        torch.save(conds, triple_img_conds_fn)

    if not single_img_cams_fn.exists() or not single_img_conds_fn.exists():
        cams, conds = get_triple_img_cams(da)
        torch.save(cams[1].unsqueeze(0), single_img_cams_fn)
        torch.save(conds[1].unsqueeze(0), single_img_conds_fn)


def get_existing_seeds(save_dir: Path, prefix: str, suffix: str = ".pt", extra_strips: Optional[Sequence[str]] = None) -> List[int]:
    """Return processed seed ids by scanning the save directory."""
    seeds: list[int] = []
    for f in Path(save_dir).glob(f"{prefix}*{suffix}"):
        seed_part = f.stem
        if seed_part.startswith(prefix):
            seed_part = seed_part[len(prefix) :]
        for extra in extra_strips or []:
            seed_part = seed_part.replace(extra, "")
        try:
            seeds.append(int(seed_part))
        except ValueError:
            continue
    return seeds


def load_ranked_seeds(csv_path: Path, columns: Sequence[str] = ("rank1", "rank2")) -> List[int]:
    """Load ranked seeds (rank1, rank2 columns) from a CSV."""
    df = pd.read_csv(csv_path, index_col=0)
    seeds: list[int] = []
    for col in columns:
        if col not in df:
            continue
        seeds.extend(df[col].dropna().astype(int).tolist())
    return seeds


def load_seed_csvs(base_dir: Path, filenames: Iterable[str], skip_first_col: bool = True) -> List[int]:
    """Flatten seeds from a list of CSV files."""
    seeds: list[int] = []
    for filename in filenames:
        df = pd.read_csv(Path(base_dir) / filename)
        cols = df.columns[1:] if skip_first_col else df.columns
        for col in cols:
            seeds.extend(df[col].dropna().astype(int).tolist())
    return seeds


def convert_stylegan_to_rgb_images(tensor: torch.Tensor) -> List["Image.Image"]:
    """Convert StyleGAN tensor to list of PIL RGB images."""

    tensor = tensor.detach().cpu()
    tensor = (tensor + 1) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    tensor = (tensor * 255).to(torch.uint8)

    images: list[Image.Image] = []
    for i in range(tensor.shape[0]):
        img_array = tensor[i].permute(1, 2, 0).numpy()
        images.append(Image.fromarray(img_array, mode="RGB"))
    return images
