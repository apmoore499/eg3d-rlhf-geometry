from typing import Optional

import torch

from core_modules.utils import meshing_utils as mutils


def sample_radiance_field_sigma_rgb_from_z(
    G: torch.nn.Module,
    z: torch.Tensor,
    c: torch.Tensor,
    shape_res: int = 128,
    device: Optional[torch.device] = torch.device("cuda"),
    truncation_psi: float = 0.7,
    truncation_cutoff: int = 14,
    noise_mode: str = "const",
    cl_frac: float = 1.0,
    rgbs_thru_sigmoid: bool = True,
    with_grad: bool = False,
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
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=resolved_device)
    transformed_ray_directions_expanded[..., -1] = -1
    head = 0
    rgbs = []
    sigmas = []
    if with_grad:
        while head < samples.shape[1]:
            rfield = G.sample(
                coordinates=samples[:, head : head + max_batch],
                directions=transformed_ray_directions_expanded[:, : samples.shape[1] - head],
                z=z,
                c=c,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
                noise_mode=noise_mode,
            )
            rgbs.append(rfield["rgb"][:3])
            sigmas.append(rfield["sigma"])
            del rfield
            head += max_batch
    else:
        with torch.no_grad():
            while head < samples.shape[1]:
                rfield = G.sample(
                    coordinates=samples[:, head : head + max_batch],
                    directions=transformed_ray_directions_expanded[:, : samples.shape[1] - head],
                    z=z,
                    c=c,
                    truncation_psi=truncation_psi,
                    truncation_cutoff=truncation_cutoff,
                    noise_mode=noise_mode,
                )
                rgbs.append(rfield["rgb"][:3])
                sigmas.append(rfield["sigma"])
                del rfield
                head += max_batch
    rgbs = torch.cat(rgbs, -2)[..., :3]
    sigmas = torch.cat(sigmas, -2)
    rgb_cat = rgbs.reshape(1, shape_res, shape_res, shape_res, 3)
    sigma_cat = sigmas.reshape(1, shape_res, shape_res, shape_res, 1)
    rgb_sigma = torch.cat((rgb_cat, sigma_cat), -1)
    return rgb_sigma
