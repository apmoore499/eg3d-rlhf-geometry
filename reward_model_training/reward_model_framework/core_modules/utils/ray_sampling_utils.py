from typing import Optional

import torch
from tqdm import tqdm

from core_modules.utils import meshing_utils as mutils


def sample_sigma_rays_from_z_as_tensor(
    G: torch.nn.Module,
    z: torch.Tensor,
    c: torch.Tensor,
    shape_res: int = 128,
    device: Optional[torch.device] = torch.device("cuda"),
    truncation_psi: float = 1.0,
    truncation_cutoff: int = 14,
    noise_mode: str = "const",
    cl_frac: float = 1.0,
    border: int = 30,
    export_type: str = "ply",
    bordermain: int = 30,
    bordersides: int = 60,
    borderback: int = 80,
    update_emas: bool = False,
) -> torch.Tensor:
    if shape_res <= 0:
        raise ValueError("shape_res must be positive")
    if cl_frac <= 0:
        raise ValueError("cl_frac must be positive")
    if not hasattr(G, "rendering_kwargs") or "box_warp" not in G.rendering_kwargs:
        raise ValueError("G must expose rendering_kwargs['box_warp']")
    N = int(shape_res * cl_frac)
    c_length = float(G.rendering_kwargs["box_warp"] * 1 * cl_frac)
    max_batch = 1000000
    samples, voxel_origin, voxel_size = mutils.create_samples(N=N, voxel_origin=[0, 0, 0], cube_length=c_length)

    resolved_device = device or z.device
    samples = samples.to(resolved_device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=resolved_device)
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=resolved_device)
    transformed_ray_directions_expanded[..., -1] = -1
    head = 0
    with tqdm(total=samples.shape[1]) as pbar:
        while head < samples.shape[1]:
            sigma = G.sample(
                coordinates=samples[:, head : head + max_batch],
                directions=transformed_ray_directions_expanded[:, : samples.shape[1] - head],
                z=z,
                c=c,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
                noise_mode=noise_mode,
                update_emas=update_emas,
            )["sigma"]
            sigmas[:, head : head + max_batch] = sigma
            head += max_batch
            pbar.update(max_batch)
    sigmas = sigmas.reshape((N, N, N))

    sigmas = torch.flip(sigmas, [0])
    pad = int(10 * shape_res / 256)
    pad_value = -1000
    sigmas[:pad] = pad_value
    sigmas[-pad:] = pad_value
    sigmas[:, :pad] = pad_value
    sigmas[:, -pad:] = pad_value
    sigmas[:, :, :pad] = pad_value
    sigmas[:, :, -pad:] = pad_value

    return sigmas


def sample_sigma_rays_from_ws_as_tensor(
    G: torch.nn.Module,
    ws: torch.Tensor,
    conditioning_params=None,
    shape_res: int = 128,
    device: Optional[torch.device] = torch.device("cuda"),
    truncation_psi: float = 1.0,
    truncation_cutoff: int = 14,
    noise_mode: str = "const",
    cl_frac: float = 1.0,
    border: int = 30,
    export_type: str = "ply",
) -> torch.Tensor:
    if shape_res <= 0:
        raise ValueError("shape_res must be positive")
    if cl_frac <= 0:
        raise ValueError("cl_frac must be positive")
    if not hasattr(G, "rendering_kwargs") or "box_warp" not in G.rendering_kwargs:
        raise ValueError("G must expose rendering_kwargs['box_warp']")
    N = int(shape_res * cl_frac)
    c_length = float(G.rendering_kwargs["box_warp"] * 1 * cl_frac)
    max_batch = 1000000
    samples, voxel_origin, voxel_size = mutils.create_samples(N=N, voxel_origin=[0, 0, 0], cube_length=c_length)

    resolved_device = device or ws.device
    samples = samples.to(resolved_device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=resolved_device)
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=resolved_device)
    transformed_ray_directions_expanded[..., -1] = -1
    head = 0
    with tqdm(total=samples.shape[1]) as pbar:
        with torch.no_grad():
            while head < samples.shape[1]:
                torch.manual_seed(0)
                sigma = G.sample_mixed(
                    coordinates=samples[:, head : head + max_batch],
                    directions=transformed_ray_directions_expanded[:, : samples.shape[1] - head],
                    ws=ws,
                    truncation_psi=truncation_psi,
                    truncation_cutoff=truncation_cutoff,
                    noise_mode=noise_mode,
                )["sigma"]
                sigmas[:, head : head + max_batch] = sigma
                head += max_batch
                pbar.update(max_batch)
    sigmas = sigmas.reshape((N, N, N))

    pad = int(10 * shape_res / 256)
    pad_value = -1000
    sigmas[:pad] = pad_value
    sigmas[-pad:] = pad_value
    sigmas[:, :pad] = pad_value
    sigmas[:, -pad:] = pad_value
    sigmas[:, :, :pad] = pad_value
    sigmas[:, :, -pad:] = pad_value

    return sigmas
