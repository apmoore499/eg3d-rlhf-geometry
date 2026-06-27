# load in our models
import os
import sys
from typing import List, Optional, Tuple, Union

#import pytorch3d
import torch
#from pytorch3d import renderer

# to synthesise the meshes 26_07_2023


sys.path.append("/path/to/eg3d-rlhf-geometry/eg3d")

# os.chdir()

# using pytorch3d to unproject points


# # NDC space camera
# fcl_ndc = (4.7,)
# prp_ndc = ((0.0, 0.0),)
# cameras_ndc = renderer.cameras.PerspectiveCameras(focal_length=fcl_ndc, principal_point=prp_ndc)
# cameras_ndc.in_ndc()
# xy = torch.tensor([[-1, 0]])
# depth = torch.tensor([[2.4]])
# xyd = torch.cat([xy, depth], dim=1)
# cameras_ndc.unproject_points(xyd, world_coordinates=False)


# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Generate images and shapes using pretrained network pickle."""
# ----------------------------------------------------------------------------
"""MODIFIED 25_04_2023 TO JUST RENDER THE MESH"""
# from pynput.keyboard import Key, Listener
import glob
import os
import re
import shutil
import subprocess
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import legacy
import matplotlib
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import open3d
import pandas as pd
import PIL
import PIL.Image
import torch
from camera_utils import FOV_to_intrinsics, LookAtPoseSampler
from IPython.display import clear_output

# from pynput.keyboard import Key, Listener
from PIL import Image
from torch_utils import misc
from tqdm import tqdm
from training.triplane import TriPlaneGenerator

plt.ion()
dpi = matplotlib.rcParams["figure.dpi"]  # Acquire default dots per inch value of matplotlib


# ----------------------------------------------------------------------------
def parse_range(s=Union[str, List]) -> List[int]:
    """Parse a comma separated list of numbers or ranges and return a list of ints.

    Example='1,2,5-10' returns [1, 2, 5, 6, 7]
    """
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------
def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    """Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    """
    if isinstance(s, tuple):
        return s
    parts = s.split(",")
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f"cannot parse 2-vector {s}")


# ----------------------------------------------------------------------------
def make_transform(translate: Tuple[float, float], angle: float):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


# ----------------------------------------------------------------------------
def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length / 2
    voxel_size = cube_length / (N - 1)
    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 3)
    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N
    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    num_samples = N**3
    return samples.unsqueeze(0), voxel_origin, voxel_size


# ----------------------------------------------------------------------------


# converts the image to a point cloud given some depth values
def imd_to_xyz(image_depth, ray_origins, ray_directions, neural_rendering_resolution):
    final_dim = neural_rendering_resolution * neural_rendering_resolution
    imd = image_depth.unsqueeze(2).expand(1, final_dim, 3)
    retval = ray_origins + imd * ray_directions
    return retval


# ----------------------------------------------------------------------------


# return depth map + seed combo ONLY (ie does not output the 3D mesh)
def generate_depth_map_128(
    G,
    seed: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shape_res: int,
    fov_deg: float,
    reload_modules: bool,
    use_fat_tail=None,
):
    """Generate images using pretrained network pickle.
    Examples:
    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """
    import numpy as np

    os.makedirs(outdir, exist_ok=True)
    cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    retvals = {}
    # Generate images.
    seed_idx = 0

    if use_fat_tail is not None:
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        z = -torch.log(torch.rand(z.numel(), device=z.device)) * z / torch.abs(z) / 1.414  # (Optional for heavier tails)
    else:
        # print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    retvals["z"] = z
    imgs = []
    angle_p = -0.2
    # for angle_y, angle_p in [(.4, angle_p), (0, angle_p), (-.4, angle_p)]:
    #    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    #    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    #    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
    #    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    #    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    #    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    #    ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
    #    img = G.synthesis(ws, camera_params)['image']
    #    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    #    imgs.append(img)
    # img = torch.cat(imgs, dim=2)
    # out_img=PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')#.save(f'{outdir}/seed{seed:04d}.png')
    # retvals['image']=out_img

    # get the depth map for it (canonical)
    angle_p = 0  # previously angle_p=-0.2
    for angle_y, angle_p in [(0, angle_p)]:
        cam_pivot = torch.tensor(G.rendering_kwargs.get("avg_camera_pivot", [0, 0, 0]), device=device)
        cam_radius = G.rendering_kwargs.get("avg_camera_radius", 2.7)
        cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        z.requires_grad = True
        ws = G.mapping(
            z,
            conditioning_params,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )

        neural_rendering_resolution = 128
        img = G.synthesis(ws, camera_params, neural_rendering_resolution=neural_rendering_resolution)

        # imd=img['image_depth']#.reshape(1,-1)

    return dict(image_depth=img["image_depth"], z=z)


# ----------------------------------------------------------------------------
def generate_single_image_and_mesh_and_pcd_256(
    G,
    seed: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shape_res: int,
    fov_deg: float,
    reload_modules: bool,
    use_fat_tail=None,
):
    """Generate images using pretrained network pickle.
    Examples:
    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """

    if torch.has_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    import numpy as np

    os.makedirs(outdir, exist_ok=True)
    cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    retvals = {}
    # Generate images.
    seed_idx = 0

    if use_fat_tail is not None:
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        z = -torch.log(torch.rand(z.numel(), device=z.device)) * z / torch.abs(z) / 1.414  # (Optional for heavier tails)
    else:
        # print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    retvals["z"] = z
    imgs = []
    angle_p = -0.2
    for angle_y, angle_p in [(0.4, angle_p), (0, angle_p), (-0.4, angle_p)]:
        cam_pivot = torch.tensor(G.rendering_kwargs.get("avg_camera_pivot", [0, 0, 0]), device=device)
        cam_radius = G.rendering_kwargs.get("avg_camera_radius", 2.7)
        cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        ws = G.mapping(
            z,
            conditioning_params,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )
        img = G.synthesis(ws, camera_params)["image"]
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        imgs.append(img)
    img = torch.cat(imgs, dim=2)
    out_img = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")  # .save(f'{outdir}/seed{seed:04d}.png')
    retvals["image"] = out_img

    # get the depth map for it
    angle_p = 0  # previously angle_p=-0.2
    for angle_y, angle_p in [(0, angle_p)]:
        cam_pivot = torch.tensor(G.rendering_kwargs.get("avg_camera_pivot", [0, 0, 0]), device=device)
        cam_radius = G.rendering_kwargs.get("avg_camera_radius", 2.7)
        cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        z.requires_grad = True
        ws = G.mapping(
            z,
            conditioning_params,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )

        neural_rendering_resolution = 256
        img = G.synthesis(ws, camera_params, neural_rendering_resolution=neural_rendering_resolution)

        imd = img["image_depth"].reshape(1, -1)
        import numpy as np
        import open3d as o3d
        import trimesh
        from training.volumetric_rendering.ray_sampler import RaySampler

        # neural_rendering_resolution=128

        rs = RaySampler()
        c = conditioning_params
        # neural_rendering_resolution=128
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)
        ray_origins, ray_directions = rs(cam2world_matrix, intrinsics, neural_rendering_resolution)
        dd = imd_to_xyz(
            image_depth=imd,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            neural_rendering_resolution=neural_rendering_resolution,
        )
        dd_c = dd.detach().cpu().numpy()
        dd_np = np.array(dd_c, dtype=np.float64)

        pcd = o3d.geometry.PointCloud()
        # np_points = np.random.rand(100, 3)

        # From numpy to Open3D
        pcd.points = o3d.utility.Vector3dVector(dd_np.reshape(-1, 3))

        pcd.estimate_normals()

        # estimate radius for rolling ball
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 1.1 * avg_dist

        mesh_ball = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))

        mesh_poisson = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False, n_threads=-1)[0]

        # create the triangular mesh with the vertices and faces from open3d
        tri_mesh = trimesh.Trimesh(
            np.asarray(mesh_ball.vertices),
            np.asarray(mesh_ball.triangles),
            vertex_normals=np.asarray(mesh_ball.vertex_normals),
        )

        trimesh.convex.is_convex(tri_mesh)

        tri_mesh.export("mesh_ball.obj")

        # create the triangular mesh with the vertices and faces from open3d
        tri_mesh = trimesh.Trimesh(
            np.asarray(mesh_poisson.vertices),
            np.asarray(mesh_poisson.triangles),
            vertex_normals=np.asarray(mesh_poisson.vertex_normals),
        )

        trimesh.convex.is_convex(tri_mesh)

        tri_mesh.export("mesh_poisson.obj")

    # np.linalg.norm(dd_n,ord=2,axis=2)
    # img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    # imgs.append(img)
    dd_n = dd_np.reshape(-1, 3)

    dd_n[:, 2] -= 0.4

    dd_sub = dd_n[np.linalg.norm(dd_n, ord=2, axis=1) <= 0.385]

    dd_sub[:, 2] += 0.4

    pcd.points = o3d.utility.Vector3dVector(dd_sub)

    # open3d.io.write_point_cloud(filename='pcd_256_sub.pcd', pointcloud=pcd)
    # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
    max_batch = 1000000
    samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs["box_warp"] * 1)  # .reshape(1, -1, 3)
    samples = samples.to(z.device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
    transformed_ray_directions_expanded[..., -1] = -1
    head = 0
    with tqdm(total=samples.shape[1]) as pbar:
        with torch.no_grad():
            while head < samples.shape[1]:
                torch.manual_seed(0)
                sigma = G.sample(
                    samples[:, head : head + max_batch],
                    transformed_ray_directions_expanded[:, : samples.shape[1] - head],
                    z,
                    conditioning_params,
                    truncation_psi=truncation_psi,
                    truncation_cutoff=truncation_cutoff,
                    noise_mode="const",
                )["sigma"]
                sigmas[:, head : head + max_batch] = sigma
                head += max_batch
                pbar.update(max_batch)
    sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
    sigmas = np.flip(sigmas, 0)
    # Trim the border of the extracted cube
    pad = int(30 * shape_res / 256)
    pad_value = -1000
    sigmas[:pad] = pad_value
    sigmas[-pad:] = pad_value
    sigmas[:, :pad] = pad_value
    sigmas[:, -pad:] = pad_value
    sigmas[:, :, :pad] = pad_value
    sigmas[:, :, -pad:] = pad_value
    # if shape_format == '.ply':
    from shape_utils import convert_sdf_samples_to_ply

    convert_sdf_samples_to_ply(
        np.transpose(sigmas, (2, 1, 0)),
        [0, 0, 0],
        1,
        os.path.join(outdir, f"seed{seed:04d}.ply"),
        level=10,
    )
    # elif shape_format == '.mrc': # output mrc
    with mrcfile.new_mmap(os.path.join(outdir, f"seed{seed:04d}.mrc"), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
        mrc.data[:] = sigmas
    retvals["geom_ply_fn"] = f"seed{seed:04d}.ply"
    retvals["geom_mrc_fn"] = f"seed{seed:04d}.mrc"
    retvals["pcd_256"] = pcd
    return retvals


# ----------------------------------------------------------------------------
def compose_vertical(mfile, ifile):
    im1 = Image.open(mfile)
    im2 = Image.open(ifile)

    # i=Image.open(i)
    im2 = im2.resize((im2.width * 2, im2.height * 2))
    # i.save('temp.png')

    im3 = Image.new("RGB", (im1.width, im1.height + im2.height))
    im3.paste(im1, (0, 0))

    im3.paste(im2, (2250, im1.height))
    return im3


# ----------------------------------------------------------------------------
def get_digits_from_string(s):
    return re.findall(r"\d+", s)


# ----------------------------------------------------------------------------
def get_seed_from_mfile(mfile):
    mfn = mfile.split("/").pop()

    digits = get_digits_from_string(mfn)

    if type(digits) == list and len(digits) == 1:
        digits = digits[0]
    return digits


# ----------------------------------------------------------------------------
def get_mfile_from_seed(seed, mfiles):
    relevant_seeds = [get_seed_from_mfile(m) for m in mfiles]

    for rs in relevant_seeds:
        if rs == seed:
            return mfiles[relevant_seeds.index(rs)]


# ----------------------------------------------------------------------------
def get_ifile_from_seed(seed, ifiles):
    relevant_seeds = [get_seed_from_mfile(m) for m in ifiles]

    for rs in relevant_seeds:
        if rs == seed:
            return ifiles[relevant_seeds.index(rs)]


# ----------------------------------------------------------------------------
def get_new_seeds(da):
    # get seeds in dir

    mfiles = glob.glob(os.path.join(da.outdir, "seed*_mesh.png"))
    mfiles.sort(key=os.path.getctime)

    # get the corresponding normal png

    ifiles = [r.replace("_mesh.png", ".png") for r in mfiles]

    # check all of ifiles are in the dir

    ifile_exists = [os.path.exists(r) for r in ifiles]

    mfiles = [mfiles[i] for i in range(len(mfiles)) if ifile_exists[i]]
    ifiles = [ifiles[i] for i in range(len(ifiles)) if ifile_exists[i]]

    # read in two images, and display them vertically

    all_found_file_seeds = [get_seed_from_mfile(mfile) for mfile in mfiles]

    new_seeds = all_found_file_seeds

    mfiles_remaining = [mfile for mfile in mfiles if get_seed_from_mfile(mfile) in new_seeds]
    ifiles_remaining = [ifile for ifile in ifiles if get_seed_from_mfile(ifile) in new_seeds]

    assert len(mfiles_remaining) == len(ifiles_remaining)
    assert all([get_seed_from_mfile(mfile) == get_seed_from_mfile(ifile) for mfile, ifile in zip(mfiles_remaining, ifiles_remaining)])

    retval = dict(new_seeds=new_seeds, mfiles=mfiles_remaining, ifiles=ifiles_remaining)

    return retval

    # return(new_seeds,mfiles_remaining,ifiles_remaining)


# ----------------------------------------------------------------------------
def compose_mesh_and_image(da, n=None):
    nsd = get_new_seeds(da)

    new_seeds = nsd["new_seeds"]
    mfiles = nsd["mfiles"]
    ifiles = nsd["ifiles"]

    if n is not None:
        new_seeds = new_seeds[:n]
    for seed in new_seeds:
        print(seed)
        m = get_mfile_from_seed(seed, mfiles)
        i = get_ifile_from_seed(seed, ifiles)
        composed = compose_vertical(m, i)
        composed.save(f"{da.outdir}/{seed}_composed.png")

    print(f"composed all for this {da.outdir}")


# ----------------------------------------------------------------------------
def create_pc_name_from_ply_fn(ply_fn):
    seed_signifier = ply_fn.split("/")[-1].split(".")[0]
    pc_sig = seed_signifier + "_pc.pcd"
    pc_fn = ply_fn.replace(seed_signifier + ".ply", pc_sig)
    return pc_fn


# ----------------------------------------------------------------------------
def convert_ply_to_pc_for_outdir(da, n=None):
    all_ply_fn = glob.glob(f"{da.outdir}/*.ply")

    all_ply_fn.sort(key=os.path.getctime)

    if n is not None:
        all_ply_fn = all_ply_fn[:n]

    for ply_fn in all_ply_fn:
        print(ply_fn)
        # open mesh
        # read ply file using open3d
        mesh = open3d.io.read_triangle_mesh(ply_fn)
        # sample from mesh for point cloud using open3d
        pcd = mesh.sample_points_uniformly(number_of_points=30000)
        # save point cloud using open3d
        pc_fn = create_pc_name_from_ply_fn(ply_fn)
        # write this one
        open3d.io.write_point_cloud(pc_fn, pcd, write_ascii=False, compressed=False, print_progress=False)


# ----------------------------------------------------------------------------
class dargs:
    def __init__(self):
        pass


# ----------------------------------------------------------------------------


def get_z_from_seed(seed, use_fat_tail=False):
    seed = int(seed)
    zdim = 512
    if use_fat_tail == True:
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, zdim))  # .to(device)
        z = -torch.log(torch.rand(z.numel(), device=z.device)) * z / torch.abs(z) / 1.414  # (Optional for heavier tails)
    else:
        # print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, zdim))  # .to(device)

    return z


import json

from PIL import Image


def stack_snapshot_images_fn(image_files):
    images = [Image.open(x) for x in image_files]
    widths, heights = zip(*(i.size for i in images))

    total_height = sum(heights)
    max_width = max(widths)

    new_im = Image.new("RGB", (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]

    return new_im


def stack_snapshot_images(rundir, cur_nimg):
    all_files = glob.glob(os.path.join(rundir, f"*mesh_network-snapshot-{cur_nimg//1000:06d}.png"))

    stacked_image = stack_snapshot_images_fn(all_files)

    stacked_im_fn = os.path.join(rundir, f"stacked_im_cur_nimg_{cur_nimg//1000:06d}.png")

    stacked_image.save(stacked_im_fn)

    return  # return(stacked_fn)


def stack_snapshot_images_from_file_names(rundir, cur_nimg, all_files):
    # all_files=glob.glob(os.path.join(rundir,f'*mesh_network-snapshot-{cur_nimg//1000:06d}.png'))
    stacked_image = stack_snapshot_images_fn(all_files)
    stacked_im_fn = os.path.join(rundir, f"stacked_im_cur_nimg_{cur_nimg//1000:06d}.png")
    stacked_image.save(stacked_im_fn)

    return  # return(stacked_fn)


def synthesise_mesh_from_G(G, seed, outdir, cur_nimg, shape_res=512, remove_mrc=True):
    st = time.time()
    da = dargs()

    # da.network_pkl=new_dict[k]['dict_of_pkl'][pkl_snapshot]

    da.outdir = outdir

    # da.network_pkl='/path/to/eg3d-rlhf-geometry/training-runs/00070-ffhq-eg3d_rebal_02_07_2023_10k_uniform_yaws-gpus1-batch2-gamma50_mse_pixelwise_sigma_reg/network-snapshot-000200.pkl'

    da.seed = seed
    da.truncation_psi = 0.7
    da.truncation_cutoff = 14
    da.level = 10  # for rendering....

    # od=da.network_pkl.split('/')[-1].replace('.pkl','')

    # da.outdir='/path/to/eg3d-rlhf-geometry/rlhf_meshes_ffhq512-128_RLMODEL'

    # os.makedirs(da.outdir,exist_ok=True)
    # seed=100

    da.shape_res = shape_res
    da.fovdeg = 18.837
    da.reload_modules = False
    da.nsamps = 70
    da.use_fat_tail = False

    G.eval()
    with torch.no_grad():
        rv = generate_single_image_and_mesh_and_pcd_256(
            G,
            seed=da.seed,
            truncation_psi=da.truncation_psi,
            truncation_cutoff=da.truncation_cutoff,
            outdir=da.outdir,
            shape_res=da.shape_res,
            fov_deg=da.fovdeg,
            reload_modules=da.reload_modules,
            use_fat_tail=da.use_fat_tail,
        )

    rv["image"].save(f"{da.outdir}/seed{seed:04d}.png")

    # get that chimera...

    # get that chimera...

    mrc_name = f"seed0{seed}.mrc"
    ply_name = f"seed0{seed}.ply"
    im_name = f"seed0{seed}.png"
    pixel_size = "2.5"

    level = str(da.level)  # level for isosurface

    script_abs_path_mesh = getattr(
        da,
        "script_abs_path_mesh",
        str(Path(__file__).resolve().parents[1] / "visualise_sdf_chimerax.py"),
    )

    mrc_name = f"seed0{seed}.mrc"
    im_name = f"seed0{seed}.png"

    supersample = 1
    yinit = 0

    n_angles = 6
    # [-90,-60,-30,0,30,60,90]
    ysteps = str([-30 for i in range(n_angles)])  #'[-15,-30,-15]'

    meshpath = da.outdir
    subprocess.call(
        [
            "chimerax",
            "--nogui",
            "--offscreen",
            script_abs_path_mesh,
            mrc_name,
            im_name,
            pixel_size,
            str(yinit),
            str(supersample),
            ysteps,
            meshpath,
            level,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    ims_dir = os.path.join(meshpath, "tmp")

    # open all im_fn
    all_ims = []
    all_im_fn = glob.glob(os.path.join(meshpath, "tmp", "*.png"))
    all_im_fn.sort(key=os.path.getctime)

    # reverse order
    all_im_fn.reverse()
    for im_fn in all_im_fn:
        current_im_fn = os.path.join(ims_dir, im_fn)
        all_ims.append(Image.open(current_im_fn))

    tiled_images = PIL.Image.new("RGB", (all_ims[0].width * len(all_ims), all_ims[0].height))
    for i, im in enumerate(all_ims):
        tiled_images.paste(im, (i * im.width, 0))

    out_fn = im_name.replace(".png", f"_mesh_network-snapshot-{cur_nimg//1000:06d}.png")
    tiled_images.save(os.path.join(meshpath, out_fn))

    if os.path.exists(ims_dir) and os.path.isdir(ims_dir):
        shutil.rmtree(ims_dir)
    et = time.time()

    mrc_path = filename = f"{da.outdir}/seed{seed:04d}.mrc"

    if remove_mrc:
        os.remove(mrc_path)

    open3d.io.write_point_cloud(filename=f"{da.outdir}/seed{seed:04d}.pcd", pointcloud=rv["pcd_256"])

    print(f"Time for seed {seed}: {et-st} seconds")
