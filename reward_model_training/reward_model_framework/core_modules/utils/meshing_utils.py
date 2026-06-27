from typing import Optional, Sequence, Tuple

import numpy as np
import skimage.measure
import torch
import trimesh
from tqdm import tqdm

from core_modules.utils import camera_utils


def create_samples(
    N: int = 256,
    voxel_origin: Optional[Sequence[float]] = None,
    cube_length: float = 2.0,
) -> Tuple[torch.Tensor, np.ndarray, float]:
    """Create grid samples for volumetric queries."""
    if N <= 1:
        raise ValueError("N must be greater than 1")
    if voxel_origin is None:
        voxel_origin = [0, 0, 0]
    if len(voxel_origin) != 3:
        raise ValueError("voxel_origin must have three coordinates")
    voxel_origin = np.array(voxel_origin) - cube_length / 2
    voxel_size = cube_length / (N - 1)
    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 3)
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    num_samples = N**3
    return samples.unsqueeze(0), voxel_origin, voxel_size


def convert_sdf_samples_to_ply(
    numpy_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    offset=None,
    scale=None,
    level=0.0,
    process=False,
) -> trimesh.Trimesh:
    """Marching cubes + trimesh conversion."""
    if isinstance(voxel_size, list):
        spacing = voxel_size
    else:
        spacing = [voxel_size] * 3

    verts, faces, normals, values = skimage.measure.marching_cubes(numpy_3d_sdf_tensor, level=level, spacing=spacing)

    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    mesh = trimesh.Trimesh(vertices=mesh_points, faces=faces, process=process)
    return mesh


def convert_hollow_sigmas_to_solid(sigmas: np.ndarray, shape_res: int) -> np.ndarray:
    """Solidify hollow volumetric sigmas by cumulative max along depth."""
    if sigmas.ndim != 3 or sigmas.shape[0] != sigmas.shape[1] or sigmas.shape[1] != sigmas.shape[2]:
        raise ValueError("sigmas must be cubic")
    cube_size = sigmas.shape[0]
    shape_res = cube_size
    sigs = sigmas
    sigs_rev = sigs[:, :, ::-1]
    ser = sigs_rev.reshape(-1, shape_res)
    max_ofe = np.max(ser, 1)
    max_ltr = np.maximum.accumulate(sigs_rev.reshape(-1, shape_res), axis=1)
    srv = sigs_rev.reshape(-1, shape_res)
    st_zeros = np.zeros_like(srv)
    moi = max_ofe.repeat(shape_res).reshape(shape_res * shape_res, shape_res)
    bl = ser == moi
    ble = bl.reshape(-1, shape_res)
    st_zeros[ble] = srv[ble]
    st_zeros[np.invert(ble)] = max_ltr[np.invert(ble)]
    st_orig_order = st_zeros.reshape(shape_res, shape_res, shape_res)[:, :, ::-1]
    sigmas = st_orig_order
    return sigmas


def _apply_face_padding(sigmas: np.ndarray, shape_res: int, bordermain: int, bordersides: int, borderback: int) -> np.ndarray:
    if sigmas.ndim != 3 or sigmas.shape[0] != sigmas.shape[1] or sigmas.shape[1] != sigmas.shape[2]:
        raise ValueError("sigmas must be cubic")
    shape_res = sigmas.shape[0]
    pad_main = int(bordermain * shape_res / 256)
    pad_sides = int(bordersides * shape_res / 256)
    pad_back = int(borderback * shape_res / 256)
    pad_value = -1000
    sigmas[:pad_sides] = pad_value
    sigmas[-pad_sides:] = pad_value
    sigmas[:, :pad_main] = pad_value
    sigmas[:, -pad_main:] = pad_value
    sigmas[:, :, :pad_back] = pad_value
    sigmas[:, :, -pad_main:] = pad_value
    return sigmas


def sample_sigmas_to_trimesh_from_ws(
    G,
    ws,
    conditioning_params=None,
    shape_res: int = 256,
    device=torch.device("cuda"),
    truncation_psi: float = 0.7,
    truncation_cutoff: int = 14,
    noise_mode: str = "const",
    cl_frac: float = 1.0,
    bordermain: int = 30,
    bordersides: int = 60,
    borderback: int = 80,
    level: float = 10,
) -> trimesh.Trimesh:
    """Sample sigma field from ws and convert to trimesh."""
    if shape_res <= 0:
        raise ValueError("shape_res must be positive")
    if cl_frac <= 0:
        raise ValueError("cl_frac must be positive")
    if not hasattr(G, "rendering_kwargs") or "box_warp" not in G.rendering_kwargs:
        raise ValueError("G must expose rendering_kwargs['box_warp']")
    N = int(shape_res * cl_frac)
    c_length = float(G.rendering_kwargs["box_warp"] * 1 * cl_frac)
    max_batch = 1000000
    samples, voxel_origin, voxel_size = create_samples(N=N, voxel_origin=[0, 0, 0], cube_length=c_length)

    samples = samples.to(device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
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
    sigmas = sigmas.reshape((N, N, N)).cpu().numpy()
    sigmas = np.flip(sigmas, 0)

    sigmas = _apply_face_padding(sigmas, shape_res=shape_res, bordermain=bordermain, bordersides=bordersides, borderback=borderback)

    mesh = convert_sdf_samples_to_ply(
        numpy_3d_sdf_tensor=np.transpose(sigmas, (2, 1, 0)),
        voxel_grid_origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / shape_res,
        level=level,
    )
    return mesh


def sample_sigmas_to_trimesh_from_ws_and_solidify(
    G,
    ws,
    conditioning_params=None,
    shape_res: int = 256,
    device=torch.device("cuda"),
    truncation_psi: float = 0.7,
    truncation_cutoff: int = 14,
    noise_mode: str = "const",
    cl_frac: float = 1.0,
    bordermain: int = 30,
    bordersides: int = 60,
    borderback: int = 80,
    level: float = 10,
) -> trimesh.Trimesh:
    """Sample sigma field, solidify, and convert to trimesh."""
    if shape_res <= 0:
        raise ValueError("shape_res must be positive")
    if cl_frac <= 0:
        raise ValueError("cl_frac must be positive")
    if not hasattr(G, "rendering_kwargs") or "box_warp" not in G.rendering_kwargs:
        raise ValueError("G must expose rendering_kwargs['box_warp']")
    N = int(shape_res * cl_frac)
    c_length = float(G.rendering_kwargs["box_warp"] * 1 * cl_frac)
    max_batch = 1000000
    samples, voxel_origin, voxel_size = create_samples(N=N, voxel_origin=[0, 0, 0], cube_length=c_length)

    samples = samples.to(device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
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
    sigmas = sigmas.reshape((N, N, N)).cpu().numpy()
    sigmas = np.flip(sigmas, 0)

    sigmas = convert_hollow_sigmas_to_solid(sigmas, shape_res=shape_res)
    sigmas = _apply_face_padding(sigmas, shape_res=shape_res, bordermain=bordermain, bordersides=bordersides, borderback=borderback)

    mesh = convert_sdf_samples_to_ply(
        numpy_3d_sdf_tensor=np.transpose(sigmas, (2, 1, 0)),
        voxel_grid_origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / shape_res,
        level=level,
    )
    return mesh
