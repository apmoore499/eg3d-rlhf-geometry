from typing import Callable, Optional, Tuple, Union

import torch

from core_modules.utils import camera_utils
from core_modules.data import io_geometry_utils as io_utils


def build_lim_pts() -> torch.Tensor:
    lims = [-0.5, 0.5]
    lim_pts = []
    for xl in lims:
        for yl in lims:
            for zl in lims:
                lim_pts.append([xl, yl, zl])
    return torch.tensor(lim_pts).float()


def center_points(ttl: torch.Tensor) -> torch.Tensor:
    return ttl - ttl.mean(dim=0, keepdim=True)


def mean_scale_pts(ttl: torch.Tensor) -> torch.Tensor:
    ttl_c = ttl - ttl.mean(dim=0, keepdim=True)
    max_abs = ttl_c.abs().max()
    if max_abs == 0:
        return ttl_c
    scale = (1 / max_abs) * 0.999999
    ttl_c = ttl_c * scale
    return ttl_c


def downsample_pcd_points(ttl: torch.Tensor, n_points: int, perm: Optional[torch.Tensor] = None) -> torch.Tensor:
    ptc = ttl
    if ptc.shape[0] < n_points:
        pad = torch.zeros_like(ptc[0, :][None, :]).expand(n_points - ptc.shape[0], -1)
        ttl = torch.cat([ptc, pad], dim=0)

    if perm is None:
        perm = torch.randperm(ttl.size(0))
    idx = perm[: min(ttl.size(0), n_points)]
    samples = ttl[idx]
    return samples


def modules_depthmap_to_pcd_from_image(
    modules_depthmap_image: torch.Tensor,
    ray_sampler: Callable[[torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]],
    n_point_samples_per_pcd_batch: int = 2048,
    return_im: bool = False,
    downsample: bool = False,
    gen_c: Optional[torch.Tensor] = None,
    nrs: Optional[int] = 128,
    radius_cutoff: Optional[float] = None,
    return_inverted: bool = False,
    center_mean: bool = False,
    lim_pts: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Convert depth map tensor to point cloud using provided RaySampler and cameras."""
    if not isinstance(modules_depthmap_image, torch.Tensor):
        raise TypeError("modules_depthmap_image must be a torch.Tensor")
    if not callable(ray_sampler):
        raise TypeError("ray_sampler must be callable")
    if modules_depthmap_image.ndim < 2:
        raise ValueError("modules_depthmap_image expected shape [..., H, W]")

    canon_cam = camera_utils.get_canonical_dmap_cams_for_rlhf()
    cam2world_matrix = canon_cam["cam2world_matrix"]
    intrinsics = canon_cam["intrinsics"]

    if modules_depthmap_image.get_device() == -1:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{modules_depthmap_image.get_device()}")

    height, width = modules_depthmap_image.shape[-2:]
    if height != width:
        raise ValueError(f"modules_depthmap_image must be square; got {height}x{width}")
    resolved_nrs = width if nrs is None else nrs
    if resolved_nrs != width:
        raise ValueError(f"nrs ({resolved_nrs}) must match depth map resolution ({width})")

    if gen_c is None:
        cam2world_matrix = canon_cam["cam2world_matrix"].to(device)
        intrinsics = canon_cam["intrinsics"].to(device)
    else:
        c = gen_c
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

    ray_origins, ray_directions = ray_sampler(cam2world_matrix, intrinsics, resolved_nrs)
    dd, retmask = io_utils.imd_to_xyz_with_radius_cutoff(
        image_depth=modules_depthmap_image,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        neural_rendering_resolution=resolved_nrs,
        radius_cutoff=radius_cutoff,
    )

    dd = dd[:, :, :].reshape(-1, 3)

    if center_mean:
        pcd = center_points(dd)
        pcd = mean_scale_pts(pcd)
    else:
        pcd = dd

    ptc = pcd[retmask[0]]
    ptc_inv = pcd[~retmask[0]]

    if downsample:
        raise AssertionError("downsample not supported in modules_depthmap_to_pcd_from_image helper")

    if lim_pts is not None:
        ptc[-8:, :] = lim_pts.type_as(ptc)

    if return_im:
        return (ptc, modules_depthmap_image)

    if return_inverted:
        return (ptc, modules_depthmap_image, ptc_inv)
    return ptc
