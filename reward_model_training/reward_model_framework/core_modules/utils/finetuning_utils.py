import sys
from pathlib import Path

import autoroot  # noqa: F401

import core_modules
from core_modules.data import io_geometry_utils as io_utils
from core_modules.data import centroid_patches

import core_modules.utils.awloss_utils_AM as awl
from core_modules.utils import camera_utils as cam_utils
from core_modules.utils import depth_to_pcd as dpcd
from core_modules.utils import meshing_utils as mutils
from core_modules.utils import radiance_field_utils as rfutils
from core_modules.utils import ray_sampling_utils as rsutils
from core_modules.utils import reward_loading as rload


import hydra
import numpy as np

# ----------------------------------------------------------------------------
import omegaconf
import pandas as pd
import torch
from omegaconf import OmegaConf
from eg3d.training.volumetric_rendering.ray_sampler import RaySampler
import copy
import glob
import os
import time

import mrcfile
import PIL
import pyvista as pv
import skimage.measure
import torch.nn as nn
import torch.utils.data
import trimesh
from PIL import Image
from tqdm import tqdm

import torch.nn.functional as F

import pathlib

import DracoPy
import open3d as o3d
import pymeshlab
import torch_geometric

import torchvision.transforms.functional as TF

import math

try:
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
except Exception:
    pass

# Compatibility re-exports (moved to core_modules.utils.reward_loading)
get_datatype_from_model_id = rload.get_datatype_from_model_id
get_cfg_fn_from_id = rload.get_cfg_fn_from_id
load_cfg_from_rm_id = rload.load_cfg_from_rm_id
load_rwd_model_from_cfg_id = rload.load_rwd_model_from_cfg_id
get_mdir_from_cfg_id = rload.get_mdir_from_cfg_id
load_datamodules_from_cfg_id = rload.load_datamodules_from_cfg_id
load_rwd_model_from_cfg = rload.load_rwd_model_from_cfg
load_tune_augmentation_from_cfg = rload.load_tune_augmentation_from_cfg


# Compatibility alias after extraction to utils.awloss_utils_AM
aw98_helper = awl.AW98Helper


class MipRayMarcher2Depth(nn.Module):
    #   AM just for the depth map, no colour
    def __init__(self):
        super().__init__()

    def run_forward(self, densities, depths):
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        densities_mid = (densities[:, :, :-1] + densities[:, :, 1:]) / 2
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2

        # deltas=torch.ones_like(densities_mid)*delta_resolution

        densities_mid = F.softplus(densities_mid - 1)  # activation bias of -1 makes things initialize better
        density_delta = densities_mid * deltas
        alpha = 1 - torch.exp(-density_delta)
        alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1 - alpha + 1e-10], -1)

        weights = alpha * torch.cumprod(alpha_shifted, -1)[:, :, :-1]
        weight_total = weights.sum(2)
        composite_depth = torch.sum(weights * depths_mid, -1) / weight_total

        # clip the composite to min/max range of depths
        composite_depth = torch.nan_to_num(composite_depth, float("inf"))
        composite_depth = torch.clamp(composite_depth, torch.min(depths), torch.max(depths))

        return composite_depth, weights

    def forward(self, densities, depths, rendering_options=None):
        composite_depth, weights = self.run_forward(
            densities,
            depths,
        )

        return composite_depth, weights


class MeshUtilsDataClass:
    def __init__(self):
        self.ray_sampler_static = RaySampler()

        self.mipnerf_depth_marcher = MipRayMarcher2Depth()

        self.canonical_pose = self.get_canonical_dmap_cams_for_rlhf()["gen_c"].reshape((1, 25)).float().cuda()

    def get_w_opt(self, G):  # gets w_opt from middle of the distyribution
        retval = G.mapping(z=torch.randn(1, 512).cuda(), c=self.get_canonical_dmap_cams_for_rlhf()["gen_c"].cuda(), truncation_psi=0.0)
        return retval

    def get_canonical_dmap_cams_for_rlhf(self):
        return cam_utils.get_canonical_dmap_cams_for_rlhf()

    def depth2mesh(self, D, cam=None, max_cos=1.0, max_len=-1, mask=None, eps=1e-6):
        x, y = np.meshgrid(np.arange(D.shape[1]), np.arange(D.shape[0]))
        if cam is None:
            cam = np.identity(3, 3)
            persp = False
        elif np.linalg.det(cam) <= eps:
            cam[2, 2] = 1
            persp = False
        else:
            persp = True
        v = np.linalg.inv(cam).dot(np.vstack((x.reshape(-1), y.reshape(-1), np.ones(len(D.reshape(-1)))))).T
        if persp:
            v = v * np.tile(D.reshape((-1, 1)), (1, 3))
        else:
            v[:, 2] = v[:, 2] * D.reshape(-1)
        if max_cos > 0:
            quad = np.vstack(
                (
                    (x[:-1, :-1] + y[:-1, :-1] * D.shape[1]).reshape(-1),
                    (x[1:, :-1] + y[1:, :-1] * D.shape[1]).reshape(-1),
                    (x[1:, 1:] + y[1:, 1:] * D.shape[1]).reshape(-1),
                    (x[:-1, 1:] + y[:-1, 1:] * D.shape[1]).reshape(-1),
                )
            ).T
            roll = np.array([1, 2, 3, 0])
            e = v[quad[:, roll], :] - v[quad, :]
            d = v[quad[:, 2:], :] - v[quad[:, :2], :]
            d = np.concatenate((d, -d), 1)
            n = np.sqrt(np.concatenate(((e * e).sum(-1), (d * d).sum(-1)), 1))
            corner = -(e * e[:, roll, :]).sum(-1) / np.maximum(n[:, :4] * n[:, roll], eps)
            split = np.concatenate(
                (
                    (e * d).sum(-1) / np.maximum(n[:, :4] * n[:, 4:], eps),
                    (e * -d[:, roll, :]).sum(-1) / np.maximum(n[:, :4] * n[:, 4 + roll], eps),
                ),
                1,
            )
            tri_cos = np.concatenate(
                (
                    np.expand_dims(split[:, :4], -1),
                    np.expand_dims(corner, -1),
                    np.expand_dims(split[:, 4 + roll], -1),
                ),
                -1,
            ).max(-1)
            tri_len = np.concatenate(
                (
                    np.expand_dims(n[:, :4], -1),
                    np.expand_dims(n[:, 4:], -1),
                    np.expand_dims(n[:, roll], -1),
                ),
                -1,
            ).max(-1)
            v_valid = (v[:, 2] > eps).astype("uint8")
            quad_valid = v_valid[quad].sum(-1)
            if mask is not None:
                if len(mask.shape) == 3 and mask.shape[-1] == 3:
                    mask = mask.astype(np.int64)
                    mask = mask[:, :, 0] + 255 * (mask[:, :, 1] + 255 * mask[:, :, 2])
                elif len(mask.shape) == 3:
                    mask = mask[:, :, 0]
                if D.shape[:2] != mask.shape[:2]:
                    mask = cv2.resize(mask, (D.shape[1], D.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = mask.reshape(-1)
                mask_valid = [[]] * 4
                mask_valid[0] = np.logical_and(
                    np.logical_and(
                        mask[quad[:, 0]] != mask[quad[:, 1]],
                        mask[quad[:, 1]] == mask[quad[:, 2]],
                    ),
                    mask[quad[:, 2]] == mask[quad[:, 3]],
                )
                mask_valid[1] = np.logical_and(
                    np.logical_and(
                        mask[quad[:, 0]] != mask[quad[:, 1]],
                        mask[quad[:, 0]] == mask[quad[:, 2]],
                    ),
                    mask[quad[:, 2]] == mask[quad[:, 3]],
                )
                mask_valid[2] = np.logical_and(
                    np.logical_and(
                        mask[quad[:, 0]] == mask[quad[:, 1]],
                        mask[quad[:, 1]] != mask[quad[:, 2]],
                    ),
                    mask[quad[:, 1]] == mask[quad[:, 3]],
                )
                mask_valid[3] = np.logical_and(
                    np.logical_and(
                        mask[quad[:, 0]] == mask[quad[:, 1]],
                        mask[quad[:, 1]] == mask[quad[:, 2]],
                    ),
                    mask[quad[:, 2]] != mask[quad[:, 3]],
                )
                mask_valid1 = np.logical_and(
                    np.logical_and(
                        mask[quad[:, 0]] == mask[quad[:, 1]],
                        mask[quad[:, 1]] == mask[quad[:, 2]],
                    ),
                    mask[quad[:, 2]] == mask[quad[:, 3]],
                )
                quad_valid[np.logical_not(mask_valid1)] = 0
                tri1 = []
                for _ in range(4):
                    i = (_ + 1) % 4
                    j = (i + 1) % 4
                    k = (j + 1) % 4
                    tri1 += [quad[t, [i, j, k]] for t in np.where(mask_valid[_])[0] if v_valid[quad[t, i]] and v_valid[quad[t, j]] and v_valid[quad[t, k]] and tri_cos[t, i] < max_cos and (max_len <= 0 or tri_len[t, i] < max_len)]
            else:
                tri1 = []
            quad_type1 = np.where(quad_valid == 3)[0]
            quad_type2 = np.where(np.logical_and(quad_valid == 4, tri_cos[:, ::2].max(1) <= tri_cos[:, 1::2].max(1)))[0]
            quad_type3 = np.where(np.logical_and(quad_valid == 4, tri_cos[:, ::2].max(1) > tri_cos[:, 1::2].max(1)))[0]
            tri1 = np.array(tri1 + [quad[i, [(j + 1) % 4, (j + 2) % 4, (j + 3) % 4]] for i, j in zip(quad_type1, np.where(v_valid[quad[quad_type1, :]] == 0)[1]) if tri_cos[i, (j + 1) % 4] < max_cos and (max_len <= 0 or tri_len[i, (j + 1) % 4] < max_len)]).reshape((-1, 3))
            tri2 = np.array([quad[i, [0, 1, 2]] for i in quad_type2 if tri_cos[i, 0] < max_cos and (max_len <= 0 or tri_len[i, 0] < max_len)] + [quad[i, [2, 3, 0]] for i in quad_type2 if tri_cos[i, 2] < max_cos and (max_len <= 0 or tri_len[i, 2] < max_len)]).reshape((-1, 3))
            tri3 = np.array([quad[i, [1, 2, 3]] for i in quad_type3 if tri_cos[i, 1] < max_cos and (max_len <= 0 or tri_len[i, 1] < max_len)] + [quad[i, [3, 0, 1]] for i in quad_type3 if tri_cos[i, 3] < max_cos and (max_len <= 0 or tri_len[i, 3] < max_len)]).reshape((-1, 3))
            tri = np.concatenate((tri1, tri2, tri3), 0)
        else:
            tri = np.zeros((0, 3), x.dtype)
        return v, tri

    def project_mesh(self, xyz, K, RT):
        """

        xyz: [N, 3]

        K: [3, 3]

        RT: [3, 4]

        """
        xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
        xyz = np.dot(xyz, K.T)

        xy = xyz[:, :2] / xyz[:, 2:]

        return xy, xyz[:, 2:]

    def read_dmap_img_to_tensor(self, dmap_img_fn):
        depth_canonical = load_depth(dmap_img_fn) / 1000.0
        depth_canonical = cv2.resize(depth_canonical, (W, H))
        return depth_canonical

    def convert_modules_depthmap_to_mesh(self, dmap_tensor, camera_params, HW=128):
        H, W = HW, HW
        cpose = camera_params.reshape((4, 4))
        K = np.array([[4.2647 * W, 0, 0.5 * W], [0, 4.2647 * H, 0.5 * H], [0, 0, 1]])
        # do thresholding here
        # depth_canonical[depth_canonical > 2.7] = 10.0
        dmap_tensor = dmap_tensor.reshape((H, W))
        v, tri = self.depth2mesh(dmap_tensor, cam=K)
        v = np.matmul(v, cpose[:3, :3].T) + cpose[:3, 3:].T

        canonical_mesh = trimesh.Trimesh(vertices=v, faces=tri)

        return canonical_mesh

    def convert_modules_depthmap_to_mesh_canonical(self, dmap_tensor, HW=128):
        camera_params = self.get_canonical_dmap_cams_for_rlhf()["gen_c"][0, :16]
        canonical_mesh = self.convert_modules_depthmap_to_mesh(dmap_tensor, camera_params=camera_params, HW=HW)
        return canonical_mesh

    def convert_sdf_samples_to_ply(
        self,
        numpy_3d_sdf_tensor,
        voxel_grid_origin,
        voxel_size,
        offset=None,
        scale=None,
        level=0.0,
        process=False,
    ):
        return mutils.convert_sdf_samples_to_ply(
            numpy_3d_sdf_tensor=numpy_3d_sdf_tensor,
            voxel_grid_origin=voxel_grid_origin,
            voxel_size=voxel_size,
            offset=offset,
            scale=scale,
            level=level,
            process=process,
        )

        # el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
        # el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

        # ply_data = plyfile.PlyData([el_verts, el_faces])
        # ply_data.write(ply_filename_out)
        # print(f"wrote to {ply_filename_out}")

    def convert_mrc(self, input_filename, output_filename, isosurface_level=1):
        with mrcfile.open(input_filename) as mrc:
            output_mesh = self.convert_sdf_samples_to_ply(
                numpy_3d_sdf_tensor=np.transpose(mrc.data, (2, 1, 0)),
                voxel_grid_origin=[0, 0, 0],
                voxel_size=1,
                level=isosurface_level,
            )

        return output_mesh

    def create_samples(self, N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
        return mutils.create_samples(N=N, voxel_origin=voxel_origin, cube_length=cube_length)

    # converts the image to a point cloud given some depth values
    def imd_to_xyz(self, image_depth, ray_origins, ray_directions, neural_rendering_resolution):
        final_dim = neural_rendering_resolution * neural_rendering_resolution
        imd = image_depth.unsqueeze(2).expand(1, final_dim, 3)
        retval = ray_origins + imd * ray_directions
        return retval

    def get_triple_dmap_cams_for_rlhf(self):
        return cam_utils.get_triple_dmap_cams_for_rlhf()

    def convert_dmap_to_point_cloud(self, modules_depthmap_image):
        nrs = modules_depthmap_image.shape[-1]
        modules_depthmap_image = modules_depthmap_image.view((nrs, nrs)).reshape((1, nrs * nrs))
        ray_sampler_static = RaySampler()
        cams = self.get_canonical_dmap_cams_for_rlhf()

        cam2world_matrix = cams["cam2world_matrix"]
        intrinsics = cams["intrinsics"]

        ray_origins, ray_directions = ray_sampler_static(cam2world_matrix, intrinsics, nrs)
        final_dim = nrs * nrs
        imd = modules_depthmap_image.unsqueeze(2).expand(1, final_dim, 3).to(ray_origins.device)
        pcd = ray_origins + imd * ray_directions
        return pcd

    def sample_sigma_rays_from_ws(
        self,
        G,
        ws,
        shape_res=256,
        device=torch.device("cuda"),
        truncation_psi=0.7,
        truncation_cutoff=14,
        noise_mode="const",
        cl_frac=1.0,
        **kwargs,
    ):
        N = int(shape_res * cl_frac)

        c_length = float(G.rendering_kwargs["box_warp"] * 1 * cl_frac)

        max_batch = 1000000
        samples, voxel_origin, voxel_size = self.create_samples(N=N, voxel_origin=[0, 0, 0], cube_length=c_length)  # .reshape(1, -1, 3)

        samples = samples.to(device)
        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
        transformed_ray_directions_expanded[..., -1] = -1
        head = 0
        # with tqdm(total = samples.shape[1]) as pbar:
        #     with torch.no_grad():
        while head < samples.shape[1]:
            # torch.manual_seed(0)
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
            # pbar.update(max_batch)
        sigmas = sigmas.reshape((N, N, N))  # .cpu().numpy()

        # sigmas = np.flip(sigmas, 0)

        return sigmas

    def sample_sigma_rays_from_z(
        self,
        G,
        z,
        c,
        shape_res=256,
        device=torch.device("cuda"),
        truncation_psi=0.7,
        truncation_cutoff=14,
        noise_mode="const",
        cl_frac=1.0,
        **kwargs,
    ):
        N = int(shape_res * cl_frac)

        c_length = float(G.rendering_kwargs["box_warp"] * 1 * cl_frac)

        max_batch = 1000000
        samples, voxel_origin, voxel_size = self.create_samples(N=N, voxel_origin=[0, 0, 0], cube_length=c_length)  # .reshape(1, -1, 3)

        samples = samples.to(device)
        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
        transformed_ray_directions_expanded[..., -1] = -1
        head = 0
        # with tqdm(total = samples.shape[1]) as pbar:
        #     with torch.no_grad():
        while head < samples.shape[1]:
            # torch.manual_seed(0)
            sigma = G.sample(
                coordinates=samples[:, head : head + max_batch],
                directions=transformed_ray_directions_expanded[:, : samples.shape[1] - head],
                z=z,
                c=c,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
                noise_mode=noise_mode,
            )["sigma"]
            sigmas[:, head : head + max_batch] = sigma
            head += max_batch
            # pbar.update(max_batch)
        sigmas = sigmas.reshape((N, N, N))  # .cpu().numpy()

        # sigmas = np.flip(sigmas, 0)

        return sigmas

    def sample_rgb_rays_from_z(
        self,
        G,
        z,
        c,
        shape_res=256,
        device=torch.device("cuda"),
        truncation_psi=0.7,
        truncation_cutoff=14,
        noise_mode="const",
        cl_frac=1.0,
        **kwargs,
    ):
        N = int(shape_res * cl_frac)

        c_length = float(G.rendering_kwargs["box_warp"] * 1 * cl_frac)

        max_batch = 1000000
        samples, voxel_origin, voxel_size = self.create_samples(N=N, voxel_origin=[0, 0, 0], cube_length=c_length)  # .reshape(1, -1, 3)

        samples = samples.to(device)
        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 3), device=device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
        transformed_ray_directions_expanded[..., -1] = -1
        head = 0

        list_to_ret = []
        # with tqdm(total = samples.shape[1]) as pbar:
        with torch.no_grad():
            while head < samples.shape[1]:
                # torch.manual_seed(0)
                sigma = G.sample(
                    coordinates=samples[:, head : head + max_batch],
                    directions=transformed_ray_directions_expanded[:, : samples.shape[1] - head],
                    z=z,
                    c=c,
                    truncation_psi=truncation_psi,
                    truncation_cutoff=truncation_cutoff,
                    noise_mode=noise_mode,
                )  # ['rgb']
                list_to_ret.append(sigma)
                head += max_batch
                # pbar.update(max_batch)
        # sigmas = sigmas.reshape((N, N, N))#.cpu().numpy()

        # sigmas = np.flip(sigmas, 0)

        # out_fn='/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules/outputs/3d_feature_volume/run_model_volume_32C2.pt'

        # torch.save(obj=list_to_ret,f=out_fn)

        return list_to_ret

    def sample_entire_radiance_field_from_z(
        self,
        G,
        z,
        c,
        shape_res=128,
        device=torch.device("cuda"),
        truncation_psi=0.7,
        truncation_cutoff=14,
        noise_mode="const",
        cl_frac=1.0,
        rgbs_thru_sigmoid=True,
        **kwargs,
    ):
        N = int(shape_res * cl_frac)

        c_length = float(G.rendering_kwargs["box_warp"] * 1 * cl_frac)

        max_batch = 1000000
        samples, voxel_origin, voxel_size = self.create_samples(N=N, voxel_origin=[0, 0, 0], cube_length=c_length)  # .reshape(1, -1, 3)

        samples = samples.to(device)
        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 3), device=device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
        transformed_ray_directions_expanded[..., -1] = -1
        head = 0

        list_to_ret = []
        # with tqdm(total = samples.shape[1]) as pbar:
        with torch.no_grad():
            while head < samples.shape[1]:
                # torch.manual_seed(0)
                sigma = G.sample(
                    coordinates=samples[:, head : head + max_batch],
                    directions=transformed_ray_directions_expanded[:, : samples.shape[1] - head],
                    z=z,
                    c=c,
                    truncation_psi=truncation_psi,
                    truncation_cutoff=truncation_cutoff,
                    noise_mode=noise_mode,
                )  # ,rgbs_thru_sigmoid=rgbs_thru_sigmoid)#['rgb']
                list_to_ret.append(sigma)
                head += max_batch

        return list_to_ret

    def sample_radiance_field_sigma_rgb_from_z(
        self,
        G,
        z,
        c,
        shape_res=128,
        device=torch.device("cuda"),
        truncation_psi=0.7,
        truncation_cutoff=14,
        noise_mode="const",
        cl_frac=1.0,
        rgbs_thru_sigmoid=True,
        with_grad=False,
        **kwargs,
    ):
        return rfutils.sample_radiance_field_sigma_rgb_from_z(
            G=G,
            z=z,
            c=c,
            shape_res=shape_res,
            device=device,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            noise_mode=noise_mode,
            cl_frac=cl_frac,
            rgbs_thru_sigmoid=rgbs_thru_sigmoid,
            with_grad=with_grad,
        )

    def sample_radiance_field_sigma_rgb_from_ws(
        self,
        G,
        ws,
        c,
        shape_res=128,
        device=torch.device("cuda"),
        truncation_psi=0.7,
        truncation_cutoff=14,
        noise_mode="const",
        cl_frac=1.0,
        rgbs_thru_sigmoid=True,
        with_grad=False,
        **kwargs,
    ):
        N = int(shape_res * cl_frac)

        c_length = float(G.rendering_kwargs["box_warp"] * 1 * cl_frac)

        max_batch = 1000000
        samples, voxel_origin, voxel_size = self.create_samples(N=N, voxel_origin=[0, 0, 0], cube_length=c_length)  # .reshape(1, -1, 3)

        samples = samples.to(device)
        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 3), device=device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
        transformed_ray_directions_expanded[..., -1] = -1
        head = 0

        # return {"rgb": rgb, "sigma": sigma}

        rgbs = []
        sigmas = []
        # with tqdm(total = samples.shape[1]) as pbar:

        if with_grad:
            while head < samples.shape[1]:
                # torch.manual_seed(0)
                rfield = G.sample_mixed(
                    coordinates=samples.squeeze().transpose(0, 1)[:, head : head + max_batch],
                    directions=transformed_ray_directions_expanded.squeeze().transpose(0, 1)[:, : samples.shape[1] - head],
                    ws=ws,
                    # c=c,
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
                    # torch.manual_seed(0)
                    rfield = G.sample_mixed(
                        coordinates=samples[:, head : head + max_batch],
                        directions=transformed_ray_directions_expanded[:, : samples.shape[1] - head],
                        ws=ws,
                        # c=c,
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

    def get_ray_samples(self, x, y, G, ws, c, shape_res=128):
        rgbs = self.sample_radiance_field_sigma_rgb_from_ws(G, ws, c, shape_res=shape_res)

        ray_rgbs = rgbs[:, x, y, :, :]
        return ray_rgbs

    def get_samples_coordinates_from_dtype(self, dtype):
        pv = self.get_pads_vals_from_dtype(dtype)

        samples, shape = self.get_samples_coordinates_from_pads_vals_dict(pv, shape_res=pv.shape_res)

        return (samples, shape)

    def get_samples_coordinates_from_pads_vals_dict(self, pads_vals, G=None, shape_res=512):
        device = torch.device("cuda")
        # shape_res=512
        # out_name=da.out_name
        pads_vals = pads_vals  # padsvals is list then u dont have this problem

        cl_frac = 1

        N = int(shape_res * cl_frac)

        c_length = float(1.0 * 1 * cl_frac)

        max_batch = 1000000
        samples, voxel_origin, voxel_size = self.create_samples(N=N, voxel_origin=[0, 0, 0], cube_length=c_length)  # .reshape(1, -1, 3)

        rhs = cpad(pads_vals["rhs_pad"], shape_res=shape_res)
        lhs = cpad(pads_vals["lhs_pad"], shape_res=shape_res)
        bot = cpad(pads_vals["bot_pad"], shape_res=shape_res)
        top = cpad(pads_vals["top_pad"], shape_res=shape_res)
        rear = cpad(pads_vals["rear_pad"], shape_res=shape_res)
        front = cpad(pads_vals["front_pad"], shape_res=shape_res)

        sam_rs = samples.reshape(1, N, N, N, 3).cpu().numpy()

        sam_rs = np.flip(sam_rs, 0)

        samples_for_eg3d = sam_rs[
            :,
            0 + rhs : shape_res - lhs,
            0 + bot : shape_res - top,
            0 + rear : shape_res - front,
            :,
        ]

        from dataclasses import dataclass
        from omegaconf import OmegaConf

        @dataclass
        class AxisPairHoriz:
            right: int = 0
            left: int = 0

        @dataclass
        class AxisPairVert:
            bottom: int = 0
            top: int = 0

        @dataclass
        class AxisPairDepth:
            rear: int = 0
            front: int = 0

        @dataclass
        class TripleAxisIndex:
            ax1horiz: AxisPairHoriz = AxisPairHoriz()
            ax2vert: AxisPairVert = AxisPairVert()
            ax3depth: AxisPairDepth = AxisPairDepth()

        tripleaxis_index = OmegaConf.structured(TripleAxisIndex())

        tripleaxis_index.ax1horiz.left = shape_res - lhs
        tripleaxis_index.ax1horiz.right = rhs
        tripleaxis_index.ax2vert.top = shape_res - top
        tripleaxis_index.ax2vert.bottom = bot
        tripleaxis_index.ax3depth.front = shape_res - front
        tripleaxis_index.ax3depth.rear = rear

        storeshape = samples_for_eg3d.shape  # [1:3]
        samples_for_eg3d = samples_for_eg3d.reshape(1, -1, 3)

        samples = torch.from_numpy(samples_for_eg3d)

        samples = samples.to(device)

        return (samples, storeshape, tripleaxis_index)

    def get_samples_coordinates_entire_no_pads(self, shape_res=128, G=None):
        """Whole-scene sigma coordinates with no pad-based cropping.

        Returns the same tuple as get_samples_coordinates_from_pads_vals_dict
        but covers the full shape_res^3 cube (no rhs/lhs/bot/top/rear/front
        slicing). Used for the 128^3 entire-scene reward data type.
        """
        device = torch.device("cuda")
        N = int(shape_res)
        c_length = 1.0
        samples, _voxel_origin, _voxel_size = self.create_samples(
            N=N, voxel_origin=[0, 0, 0], cube_length=c_length
        )
        sam_rs = samples.reshape(1, N, N, N, 3).cpu().numpy()
        sam_rs = np.flip(sam_rs, 0)

        from dataclasses import dataclass
        from omegaconf import OmegaConf

        @dataclass
        class _AxisPairHoriz:
            right: int = 0
            left: int = 0

        @dataclass
        class _AxisPairVert:
            bottom: int = 0
            top: int = 0

        @dataclass
        class _AxisPairDepth:
            rear: int = 0
            front: int = 0

        @dataclass
        class _TripleAxisIndex:
            ax1horiz: _AxisPairHoriz = _AxisPairHoriz()
            ax2vert: _AxisPairVert = _AxisPairVert()
            ax3depth: _AxisPairDepth = _AxisPairDepth()

        tripleaxis_index = OmegaConf.structured(_TripleAxisIndex())
        tripleaxis_index.ax1horiz.left = N
        tripleaxis_index.ax1horiz.right = 0
        tripleaxis_index.ax2vert.top = N
        tripleaxis_index.ax2vert.bottom = 0
        tripleaxis_index.ax3depth.front = N
        tripleaxis_index.ax3depth.rear = 0

        storeshape = sam_rs.shape
        samples_for_eg3d = sam_rs.reshape(1, -1, 3)
        samples_t = torch.from_numpy(samples_for_eg3d).to(device)
        return (samples_t, storeshape, tripleaxis_index)

    def sample_sigma_rays_from_ws_as_tensor(
        self,
        G,
        ws,
        conditioning_params,
        shape_res=128,
        device=torch.device("cuda"),
        truncation_psi=0.7,
        truncation_cutoff=14,
        noise_mode="const",
        cl_frac=1.0,
        border=30,
        export_type="ply",
    ):
        return rsutils.sample_sigma_rays_from_ws_as_tensor(
            G=G,
            ws=ws,
            conditioning_params=conditioning_params,
            shape_res=shape_res,
            device=device,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            noise_mode=noise_mode,
            cl_frac=cl_frac,
            border=border,
            export_type=export_type,
        )

    def mesh_subset_of_points_from_samples_from_z_with_grad(
        self,
        G,
        z,
        conditioning_params,
        samples,
        truncation_psi=1.0,
        truncation_cutoff=14,
        noise_mode="const",
        update_emas=False,
        max_batch=1000000,
        torch_manual_seed=None,
    ):
        # max_batch=1000000

        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
        transformed_ray_directions_expanded[..., -1] = -1

        head = 0
        # sigmas_cpu=[]
        # with tqdm(total = samples.shape[1]) as pbar:
        # with torch.no_grad():
        while head < samples.shape[1]:
            if torch_manual_seed is not None:
                torch.manual_seed(torch_manual_seed)
            sigma = G.sample(
                samples[:, head : head + max_batch],
                transformed_ray_directions_expanded[:, : samples.shape[1] - head],
                z,
                conditioning_params,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
                noise_mode=noise_mode,
                update_emas=update_emas,
            )["sigma"]
            sigmas[:, head : head + max_batch] = sigma
            # sigmas_cpu[:, head:head+max_batch] = sigma.cpu().numpy()
            head += max_batch
            # pbar.update(max_batch)

        return sigmas

    def mesh_subset_of_points_from_samples_from_ws_with_grad(
        self,
        G,
        ws,
        conditioning_params,
        samples,
        truncation_psi=1.0,
        truncation_cutoff=14,
        noise_mode="const",
        update_emas=False,
        max_batch=1000000,
    ):
        # max_batch=1000000

        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=ws.device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=ws.device)
        transformed_ray_directions_expanded[..., -1] = -1

        head = 0
        # sigmas_cpu=[]
        # with tqdm(total = samples.shape[1]) as pbar:
        # with torch.no_grad():
        while head < samples.shape[1]:
            # torch.manual_seed(0)
            sigma = G.sample_mixed(
                coordinates=samples[:, head : head + max_batch],
                directions=transformed_ray_directions_expanded[:, : samples.shape[1] - head],
                ws=ws,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
                update_emas=update_emas,
            )["sigma"]
            sigmas[:, head : head + max_batch] = sigma
            # sigmas_cpu[:, head:head+max_batch] = sigma.cpu().numpy()
            head += max_batch
            # pbar.update(max_batch)

        return sigmas

    def get_pads_vals_from_dtype(self, dtype):
        project_root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[5]))
        sigmas_spec = pd.read_excel(
            project_root / "reward_model_training" / "reward_model_framework" / "core_modules" / "notebooks" / "sigma_vals_data_spec.xlsx",
            index_col="dtype",
        )
        pads_vals_fn = sigmas_spec.loc[dtype].pads_vals_config_fn
        pads_vals = omegaconf.OmegaConf.load(pads_vals_fn)

        return pads_vals

    def get_nose_samples_with_shape(self):
        pads_vals = self.get_nose_pads_dict()

        samples, shape = self.get_samples_coordinates_from_pads_vals_dict(pads_vals=pads_vals)

        return dict(samples=samples, shape=shape)

    def get_nose_pads_dict(self):
        nose_yaml_fn = cam_utils.get_static_configs_dir() / "pads_vals_nose.yaml"

        # import yaml
        # import omegaconf

        # with open(nose_yaml_fn,'r') as f:
        pads_vals = omegaconf.OmegaConf.load(nose_yaml_fn)

        return pads_vals

    def export_sample_mrc(self, sigmas, out_fn=None):
        import os

        import mrcfile

        export_dir = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[5])) / "reward_model_training" / "reward_model_framework" / "core_modules" / "notebooks" / "verifying_sigma_nose_512"
        export_dir.mkdir(parents=True, exist_ok=True)
        nfiles = len(os.listdir(export_dir))

        if out_fn is None:
            fn_export = export_dir / f"testing{nfiles + 1}.mrc"

        else:
            fn_export = out_fn

        with mrcfile.new_mmap(fn_export, overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
            mrc.data[:] = sigmas.detach().cpu()

        print(f"mrc written to: {fn_export}")

    def visualise_meshes_pv(self, list_of_meshes):
        import pyvista as pv

        pl = pv.Plotter(notebook=False, shape=(1, len(list_of_meshes) + 1))

        colourlist = ["blue", "red", "green", "yellow", "brown", "white"] * 3

        for k, m in enumerate(list_of_meshes):
            pl.subplot(0, k)
            pl.add_mesh(pv.wrap(m))

        pl.subplot(0, k + 1)

        for k, m in enumerate(list_of_meshes):
            pl.add_mesh(pv.wrap(m), color=colourlist[k])

        pl.show()

    def sample_sigmas_to_trimesh_from_ws(
        self,
        G,
        z,
        conditioning_params,
        shape_res=256,
        device=torch.device("cuda"),
        truncation_psi=0.7,
        truncation_cutoff=14,
        noise_mode="const",
        cl_frac=1.0,
        bordermain=30,
        bordersides=60,
        borderback=80,
        export_type="ply",
        level=10,
    ):
        return mutils.sample_sigmas_to_trimesh_from_ws(
            G=G,
            ws=z,
            conditioning_params=conditioning_params,
            shape_res=shape_res,
            device=device,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            noise_mode=noise_mode,
            cl_frac=cl_frac,
            bordermain=bordermain,
            bordersides=bordersides,
            borderback=borderback,
            level=level,
        )

    def query_and_print_grads(self, G):
        # rv=dict(hello='hello')
        # print(Pretty(rv))
        params = [param for param in G.parameters() if param.numel() > 0 and param.grad is not None]
        all_grads = torch.hstack([p.grad.flatten() for p in params])
        all_grads.min()

        len_params_list = len(params)
        grads_min = torch.min(all_grads)
        grads_max = torch.max(all_grads)
        grads_mean = torch.mean(all_grads)
        grads_std = torch.std(all_grads)

        rv = dict(
            len_params_list=len_params_list,
            grads_min=grads_min,
            grads_max=grads_max,
            grads_mean=grads_mean,
            grads_std=grads_std,
        )

        return rv

    def sample_sigmas_to_trimesh_from_ws_and_solidify(
        self,
        G,
        z,
        conditioning_params,
        shape_res=256,
        device=torch.device("cuda"),
        truncation_psi=0.7,
        truncation_cutoff=14,
        noise_mode="const",
        cl_frac=1.0,
        bordermain=30,
        bordersides=60,
        borderback=80,
        export_type="ply",
        level=10,
    ):
        return mutils.sample_sigmas_to_trimesh_from_ws_and_solidify(
            G=G,
            ws=z,
            conditioning_params=conditioning_params,
            shape_res=shape_res,
            device=device,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            noise_mode=noise_mode,
            cl_frac=cl_frac,
            bordermain=bordermain,
            bordersides=bordersides,
            borderback=borderback,
            level=level,
        )

    def convert_hollow_sigmas_to_solid(self, sigmas, shape_res):
        return mutils.convert_hollow_sigmas_to_solid(sigmas, shape_res=shape_res)

    def convert_field_volume_to_mesh(self, field_data):
        np_field = field_data.detach().cpu().numpy()
        voxel_grid_origin = [0.0, 0.0, 0.0]
        llf = list(field_data.shape)
        voxel_size = [1 / llf[0], 1 / llf[1], 1 / llf[2]]

        mesh_from_field = self.convert_sdf_samples_to_ply(
            numpy_3d_sdf_tensor=np_field,
            voxel_grid_origin=voxel_grid_origin,
            voxel_size=voxel_size,
            level=10.0,
        )

        return mesh_from_field

    def visualise_mesh(
        self,
        trimesh_object,
        ply_fn,
        n_angles=10,
        azimuth_angle_interval=-15,
        azimuth_angle_initial=-60,
        save=True,
        use_unit_cube=True,
        zoom=0.8,
        win_size=512,
        offset_vis=100,
        translate=[0.0, 0.0, 0.0],
        opacity_cube=1.0,
        specular=0.9,
        bkgd="#363940",
        plotting_kwargs={},
        # lighting='light kit',
        # silhouette=False,
    ):
        tmo = trimesh_object.copy()

        tmo.apply_translation(translate)
        rot = pv.wrap(tmo)

        xrot = 90

        rot = rot.rotate_x(xrot, inplace=False)

        st = time.time()

        pl = pv.Plotter(window_size=[win_size, win_size], off_screen=True)

        ims = []

        pl.set_background(bkgd)

        bounds = (0.0, 1.0, -1.0, 0.0, 0.0, 1.0)
        model = pv.Cube(bounds=bounds)
        unit_cube_for_bounding = pv.wrap(model.outline())
        # xrot = 90
        # pl.enable_shadows()
        pl.enable_ssao()
        pl.enable_stereo_render()
        if use_unit_cube:
            ucb = pl.add_mesh(
                unit_cube_for_bounding,
                smooth_shading=False,
                show_edges=True,
                color=[220 / 255, 243 / 255, 252 / 255],
                specular=0.9,
                opacity=opacity_cube,
            )

            pl.set_focus(unit_cube_for_bounding.center)

            pl.enable_anti_aliasing()
            pl.camera_position = "yz"
            pl.set_focus(unit_cube_for_bounding.center)

        else:
            bounds = tuple(tmo.bounds.ravel().tolist())
            model = pv.Cube(bounds=bounds)
            unit_cube_for_bounding = pv.wrap(model.outline())

            # bounds = (0.0, 1.0, -1.0, 0.0, 0.0, 1.0)
            # model = pv.Cube(bounds=bounds)
            # unit_cube_for_bounding = pv.wrap(model.outline())
            pl.set_focus(unit_cube_for_bounding.center)
            pl.enable_anti_aliasing()
            pl.camera_position = "yz"

        pl.camera.zoom(zoom)

        mesh1 = pl.add_mesh(
            rot,
            # smooth_shading=True,
            color=[220 / 255, 243 / 255, 252 / 255],
            **plotting_kwargs,
            # specular=specular,
            # silhouette=silhouette,
        )

        azimuth_angle = azimuth_angle_initial

        pl.camera.Azimuth(azimuth_angle)

        # cdim = 100

        aa = azimuth_angle_interval
        # for azimuth_angle in [-45,-30,-15,0,15,30,45]:
        for i in range(n_angles):
            image = pl.screenshot(filename=None, return_img=True)
            ims.append(np.asarray(image)[offset_vis : win_size - offset_vis, offset_vis : win_size - offset_vis])

            # pl.reset_camera()
            # pl.camera_position = "yz"

            # if use_unit_cube:
            pl.set_focus(unit_cube_for_bounding.center)
            # else:
            #    pl.set_focus(rot.center)

            # pl.camera.zoom(0.3)
            pl.camera.Azimuth(azimuth_angle_interval)

            aa = aa + azimuth_angle_interval
        out_fn = ply_fn.replace(".ply", ".jpg").replace("mesh_", "mesh_cat_")

        ims = ims[1:]

        import PIL

        # if save:
        visualised_mesh = PIL.Image.fromarray(np.hstack(ims)).convert("RGB")

        if save:
            visualised_mesh.save(out_fn)

        pl.close()
        et = time.time()
        tt = et - st

        return visualised_mesh

    def clean_nchal_res_dir(self):
        import os
        import pathlib

        pl = pathlib.Path("/media/krillman/1TB_DATA/NoW_challenge_16_04_2023/now_evaluation")
        ll = list(pl.glob("output*/results/*.obj"))
        print("total n *.obj files:")
        print(len(ll))

        to_del = [p for p in ll if "predicted_aligned.obj" not in p.as_posix()]

        print("to remove after also exclude predicted_aligned.obj files:")
        print(len(to_del))

        to_del = [p for p in to_del if "gt_scan_val.obj" not in p.as_posix()]  # p.as_posix()]

        print("to remove after also exclude gt_scan_val.obj files:")
        print(len(to_del))

        for f in to_del:
            os.remove(f)

        print("cleaned now challenge dir")

        # for d in to_del[:1000]:
        #    print(d)


class DataHelperForEG3DLoss(MeshUtilsDataClass):
    def __init__(self, hcfg_fn_rwd_model, rwd_model_data_type=None):
        super().__init__()

        if type(hcfg_fn_rwd_model) == str:
            hydra_cfg = omegaconf.OmegaConf.load(hcfg_fn_rwd_model)
            if "rwd_model_data_type" not in hydra_cfg.keys():
                hydra_cfg.rwd_model_data_type = rwd_model_data_type
        else:
            hydra_cfg = hcfg_fn_rwd_model

        self.hydra_cfg = hydra_cfg

        minimal_dclass = core_modules.data.dset_loaders.dset_single_stream_ordered_minimal(all_combined_rankings=[-1], dtype="point_cloud_entire", ddir_func=core_modules.data.misc_small_utils.ddir_func, seed_func=core_modules.data.misc_small_utils.seed_func_default, include_goodseed=False, dset_version="three")

        # minimal_dclass.dset_version = "three"

        self.dclass_minimal = minimal_dclass
        self.rwd_dtype = self.hydra_cfg.rwd_model_data_type

        if self.hydra_cfg.rwd_model_data_type is not None and "98" in self.hydra_cfg.rwd_model_data_type:
            self.M_aw98 = awl.return_awloss_model_98()  # only load if we get lmks, otherwise run out of memory

        self.set_lim_pts()
        self.upsampler_for_dmap = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Tune-time transform, inherited from the reward model's saved run-config
        # (single source of truth; see reward_loading.load_tune_augmentation_from_cfg).
        # Instantiated once here and applied as the FINAL step of
        # format_gen_out_for_rwd_input, so every generated reward input is
        # normalised exactly as the reward model was trained to expect -- instead
        # of being re-declared by hand per eg3d tune config (train/serve skew).
        # Defaults to identity when the run recorded no tune slot, so models
        # without one keep their previous behaviour.
        tune_aug_cfg = load_tune_augmentation_from_cfg(omegaconf.OmegaConf.select(self.hydra_cfg, "rwd_model_id"))
        self.tune_aug = hydra.utils.instantiate(tune_aug_cfg) if tune_aug_cfg is not None else torch.nn.Identity()

    def predict_landmarks_from_rgb_on_gpu(self, rgb, detach=False):  # rgb is shape (1,3,256,256)
        lmks = awl.predict_landmarks_from_rgb_on_gpu(model_ft=self.M_aw98, im=rgb)

        if detach == True:
            lmks = lmks.detach()
        return lmks

    def set_lim_pts(self):
        self.lim_pts = dpcd.build_lim_pts()
        return self

    def modules_depthmap_to_pcd_from_image(
        self,
        modules_depthmap_image,
        n_point_samples_per_pcd_batch=2048,
        return_im=False,
        downsample=False,
        gen_c=None,
        nrs=128,
        radius_cutoff=None,
        return_inverted=False,
        center_mean=False,
    ):  # ,canon_cam=None):
        return dpcd.modules_depthmap_to_pcd_from_image(
            modules_depthmap_image=modules_depthmap_image,
            ray_sampler=self.ray_sampler_static,
            n_point_samples_per_pcd_batch=n_point_samples_per_pcd_batch,
            return_im=return_im,
            downsample=downsample,
            gen_c=gen_c,
            nrs=nrs,
            radius_cutoff=radius_cutoff,
            return_inverted=return_inverted,
            center_mean=center_mean,
            lim_pts=self.lim_pts,
        )

    def sample_sigma_rays_from_z_as_tensor(
        self,
        G,
        z,
        c,
        shape_res=128,
        device=torch.device("cuda"),
        truncation_psi=1.0,
        truncation_cutoff=14,
        noise_mode="const",
        cl_frac=1.0,
        border=30,
        export_type="ply",
        bordermain=30,
        bordersides=60,
        borderback=80,
        update_emas=False,
    ):
        return rsutils.sample_sigma_rays_from_z_as_tensor(
            G=G,
            z=z,
            c=c,
            shape_res=shape_res,
            device=device,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            noise_mode=noise_mode,
            cl_frac=cl_frac,
            border=border,
            export_type=export_type,
            bordermain=bordermain,
            bordersides=bordersides,
            borderback=borderback,
            update_emas=update_emas,
        )

    def format_gen_out_for_rwd_input(self, **kwargs):
        # Build the raw reward representation for this dtype, then apply the
        # inherited tune transform (identity if the reward model recorded none)
        # as the single normalisation step -- see self.tune_aug in __init__.
        # EXCEPTION: point_cloud_entire applies self.tune_aug PER ITEM on the
        # (N,3) cloud inside its branch (before the (3,N) transpose), matching how
        # the reward model was trained (dset_loaders applies it on (N,3)); a second
        # blanket apply here would operate on the wrong axes of the batched
        # (B,3,N), so that dtype is excluded from the apply below.
        rwd_input = self._format_gen_out_for_rwd_input_raw(**kwargs)
        if self.hydra_cfg.rwd_model_data_type == "point_cloud_entire":
            return rwd_input
        return self.tune_aug(rwd_input)

    def _format_gen_out_for_rwd_input_raw(self, **kwargs):
        if self.hydra_cfg.rwd_model_data_type == "sigma_rays":
            return kwargs["sigma_rays"]

        elif self.hydra_cfg.rwd_model_data_type == "nose_512":
            return kwargs["nose_512"]

        else:
            gen_img = kwargs["gen_img"]
            gen_c = kwargs["gen_c"]
            bsize = kwargs["bsize"]
            ncams = kwargs["ncams"]
            # Forward-looking landmark/patch branches predict landmarks; default
            # to keeping gradients (detach=False) unless a caller overrides.
            detach = kwargs.get("detach", False)

            gen_depth = gen_img["image_depth"]  # [bsize*ncams,1,128,128]

            if self.hydra_cfg.data.get("upsample_64_to_128"):
                if gen_depth.shape[-1] == 128:
                    d = 1
                elif gen_depth.shape[-1] == 64:
                    gen_depth = self.upsampler_for_dmap(gen_depth)
                else:
                    assert False, "error gen depth not in your req!"

            gen_rgb = gen_img["image"]  # [bsize*ncams,3,512,512]

            if self.hydra_cfg.rwd_model_data_type == "canonical_rgb_lmks_98":
                retlist = []
                for k in range(gen_depth.shape[0]):
                    img = (gen_rgb[k] / 2 + 0.5).clamp(0, 1)

                    g_rgb = torch.nn.functional.interpolate(
                        img.unsqueeze(0),
                        size=(256, 256),
                        mode="bilinear",
                        align_corners=True,
                    )

                    lmks = self.predict_landmarks_from_rgb_on_gpu(g_rgb, detach=detach)

                    retval = lmks.reshape(-1, 98 * 2)
                    retlist.append(retval.unsqueeze(0))

                retval = torch.cat(retlist, 0)

                return retval

            if self.hydra_cfg.rwd_model_data_type == "aw98_patch_geom_nose_8":
                pcd_patches_new = []
                for k in range(gen_depth.shape[0]):
                    pcd = self.modules_depthmap_to_pcd_from_image(
                        modules_depthmap_image=gen_depth[k],
                        downsample=False,
                        return_im=False,
                        gen_c=gen_c[k][None, ...],
                        nrs=128,
                        radius_cutoff=None,
                    )
                    g_rgb = torch.nn.functional.interpolate(
                        gen_rgb[k].unsqueeze(0),
                        size=(256, 256),
                        mode="bilinear",
                        align_corners=True,
                    )
                    lmks_dict = self.predict_landmarks_from_rgb_on_gpu(g_rgb, detach=detach)

                    a = lmks_dict.shape[0]
                    b = lmks_dict.shape[1]

                    r1 = 10

                    offset = torch.FloatTensor(a, b).uniform_(-r1, r1).to(int).to(lmks_dict.device)

                    lmks_dict = (lmks_dict + offset).to(int)

                    centroids_dict = self.dclass_minimal.aw98_landmark_to_pcd_index(seed=-1, nrs=128, lmks_aw98=lmks_dict)
                    rndm_groups = np.array([54])

                    patch = centroid_patches.normalize_pcd_and_get_processed_patches_no_colour([pcd], [centroids_dict], rndm_groups)

                    pcd_patches_new.append(patch[0])

                retval = torch.cat(pcd_patches_new, 0)
                return retval

            if self.hydra_cfg.rwd_model_data_type == "point_cloud_entire":
                # Unified pcd path. Build the full depth-map cloud per item, then
                # apply the inherited tune transform (subsample -> center ->
                # mean-scale) on the (N,3) cloud -- the SAME object/shape the reward
                # model trained on (dset_loaders point_cloud_entire branch). This
                # replaces the legacy data.center_points / unit_scale_points /
                # downsample_pcd_points / n_point_samples_per_pcd_batch flags.
                pcds_new = []
                for k in range(gen_depth.shape[0]):
                    pcd = self.modules_depthmap_to_pcd_from_image(
                        modules_depthmap_image=gen_depth[k],
                        downsample=False,
                        return_im=False,
                        gen_c=gen_c[k][None, ...],
                        nrs=128,
                        radius_cutoff=None,
                    )  # (N,3)

                    pcd = self.tune_aug(pcd)  # (N,3) -> (M,3)

                    pcds_new.append(pcd.transpose(1, 0).unsqueeze(0))  # (1,3,M)
                return torch.cat(pcds_new, dim=0)  # (B,3,M)

            if self.hydra_cfg.rwd_model_data_type == "triple_dmap":
                gen_depth = gen_depth.view(bsize, ncams, 128, 128)
                gen_depth = gen_depth.unsqueeze(2)

                return gen_depth

            if self.hydra_cfg.rwd_model_data_type == "single_dmap":
                gen_depth = gen_depth.view(bsize, ncams, 128, 128)
                gen_depth = gen_depth.unsqueeze(2)

                return gen_depth

    def get_triple_dmap_cameras(self):
        return cam_utils.get_triple_dmap_cameras()

    def get_single_dmap_camera(self):
        return cam_utils.get_single_dmap_camera()

    def format_batch(self, gen_c_template, z):
        bsize = z.shape[0]
        ncams = gen_c_template.shape[0]
        gen_c = gen_c_template[None, ...].expand(bsize, -1, 25).to(z.device)
        z = z.unsqueeze(1).expand(bsize, gen_c.shape[1], 512)
        gen_z = z.reshape(bsize * ncams, 512)
        gen_c = gen_c.reshape(bsize * ncams, 25)

        return dict(bsize=bsize, ncams=ncams, gen_c=gen_c, gen_z=gen_z)


class ImageHelper:
    def __init__(self):
        return

    def compose_images_horizontally(self, image1_path, image2_path):
        # Open the images

        if isinstance(image1_path, "str"):
            image1 = Image.open(image1_path)

        if isinstance(image2_path, "str"):
            image2 = Image.open(image2_path)

        # Get the dimensions of the images
        width1, height1 = image1.size
        width2, height2 = image2.size

        gap_width = 200
        # Calculate the new dimensions of the composed image
        new_width = width1 + width2 + gap_width
        new_height = max(height1, height2)

        # Create a new image with the new dimensions
        new_image = Image.new("RGB", (new_width, new_height), color=(255, 0, 0))

        # Paste the first image onto the new image
        new_image.paste(image1, (0, 0))

        # Paste the second image onto the new image
        new_image.paste(image2, (width1 + gap_width, 0))

        # Return the new image
        return new_image

    def compose_images_vertically(self, list_of_images):
        # Open the images

        ilist = []

        heights = []
        widths = []

        for i in list_of_images:
            if isinstance(i, str):
                ii = Image.open(i)

            elif isinstance(i, PIL.Image.Image):
                ii = copy.deepcopy(i)

            else:
                assert False, f"type of the thing: {type(i)}"

            ilist.append(ii)

            w, h = ii.size

            heights.append(h)
            widths.append(w)

        gaps_concept = min([h * 0.1 for h in heights])

        gap_height = min(100, gaps_concept)
        gap_height = int(gap_height)
        # Calculate the new dimensions of the composed image
        new_width = max(widths)
        new_height = sum(heights) + gap_height * (len(heights) - 1)

        # Create a new image with the new dimensions
        new_image = Image.new("RGBA", (new_width, new_height), color=(255, 0, 0, 80))

        for i in range(len(ilist)):
            if i == 0:
                new_image.paste(ilist[i], (0, 0))
                continue
            # Paste the first image onto the new image

            # Paste the second image onto the new image
            new_image.paste(ilist[i], (0, sum(heights[0:i]) + gap_height * (i + 1)))

            # Return the new image
        return new_image


class UniversalMeshFormat:
    """#
    helper module to quickly convert between various mesh formats: dracopy, open3d, trimesh, torch_geometric. can update to also use from eg ipyvolume / pyvista later if need
    """

    def __init__(self, mesh_object):
        if type(mesh_object) == DracoPy.DracoMesh:
            self.points = mesh_object.points
            self.faces = mesh_object.faces
            # self.normals=mesh_object.normals

        if type(mesh_object) == o3d.cuda.pybind.geometry.TriangleMesh or type(mesh_object) == o3d.geometry.TriangleMesh:
            self.points = np.asarray(mesh_object.vertices)
            self.faces = np.asarray(mesh_object.triangles)
            # self.normals=mesh_object.normals
        if type(mesh_object) == trimesh.base.Trimesh:
            self.points = np.asarray(mesh_object.vertices)
            self.faces = np.asarray(mesh_object.faces)

        if type(mesh_object) == torch_geometric.data.data.Data:
            self.points = mesh_object.pos.numpy()
            self.faces = mesh_object.face.numpy().transpose()

        if type(mesh_object) == str:
            mfn = pathlib.Path(mesh_object)

            if mfn.suffix == ".drc":
                with open(mfn, "rb") as draco_file:
                    mesh_object = DracoPy.decode(draco_file.read())
                    self.points = mesh_object.points
                    self.faces = mesh_object.faces
                    # do some flags to check...
            elif mfn.suffix in [".ply", ".obj"]:
                mesh_object = trimesh.load(mesh_object)
                self.points = np.asarray(mesh_object.vertices)
                self.faces = np.asarray(mesh_object.faces)

    def as_dracopy(self):
        print("not implemented")

    def as_open3d(self):
        retval = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(self.points),
            triangles=o3d.utility.Vector3iVector(self.faces),
        )
        return retval

    def as_trimesh(self):
        retval = trimesh.Trimesh(vertices=self.points, faces=self.faces)
        return retval

    def as_ptg_data(self):
        retval = trimesh.Trimesh(vertices=self.points, faces=self.faces)
        trimesh.repair.fix_inversion(retval)
        retval = torch_geometric.utils.from_trimesh(retval)
        return retval

    def as_o3d_pcd(self):
        retval = o3d.geometry.TriangleMesh(points=o3d.utility.Vector3dVector(self.points))
        return retval

    def visualise_points(self):
        retval = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(self.points))
        o3d.visualization.draw_geometries([retval])

    def visualise_mesh_pv(self):
        tmo = self.as_trimesh()
        tmpdir = pathlib.Path("tmp")
        tmpdir.mkdir(exist_ok=True, parents=True)
        tmo.export(tmpdir.joinpath("tmp.obj"))

        pl = pv.Plotter(notebook=False)  # ,shape=(1,2))
        omesh = pv.PolyData(pv.wrap(tmo))
        pl.add_mesh(omesh)
        pl.show()

    def normalise_unit_scale(self):
        fmi = self.as_trimesh()
        rescale = max(fmi.extents) / 2.0
        tform = [-(fmi.bounds[1][i] + fmi.bounds[0][i]) / 2.0 for i in range(3)]
        matrix = np.eye(4)
        matrix[:3, 3] = tform
        fmi.apply_transform(matrix)
        matrix = np.eye(4)
        matrix[:3, :3] /= rescale
        fmi.apply_transform(matrix)

        self.points = np.asarray(fmi.vertices)
        self.faces = np.asarray(fmi.faces)

        return self

    def clean_mesh_to_60000(self):
        tmo = self.as_trimesh()
        tmpdir = pathlib.Path("tmp")
        tmpdir.mkdir(exist_ok=True, parents=True)
        mname = tmpdir.joinpath("tmp.obj")
        tmo.export(mname)
        clean_inverted_mesh(mname.as_posix())
        newmesh = trimesh.load(mname)
        self.points = np.asarray(newmesh.vertices)
        self.faces = np.asarray(newmesh.faces)
        os.remove(mname)
        return self
        # save it out


class LandmarksRaysTransformsHelper:
    def __init__(self, AW98_MODULE, MUDC):
        self.AW98_MODULE = AW98_MODULE
        self.MUDC = MUDC
        self.topil = TF.to_pil_image
        return

    def get_lmk_idx_for_now_challenge(self):
        dict_of_idx_now_challenge = {
            "left_eye_outer": {"landmark_index": 60, "NoW_lmk_order": 1},
            "left_eye_inner": {"landmark_index": 64, "NoW_lmk_order": 2},
            "right_eye_inner": {"landmark_index": 68, "NoW_lmk_order": 3},
            "right_eye_outer": {"landmark_index": 72, "NoW_lmk_order": 4},
            "septum": {"landmark_index": 57, "NoW_lmk_order": 5},
            "left_mouth": {"landmark_index": 76, "NoW_lmk_order": 6},
            "right_mouth": {"landmark_index": 82, "NoW_lmk_order": 7},
        }

        return dict_of_idx_now_challenge

    def export_marching_cubes_mesh(self, G, w_pivot, mesh_res, pose, level, tverts):
        with torch.no_grad():
            canon_mesh_sampled_fwd = self.MUDC.sample_sigmas_to_trimesh_from_ws_and_solidify(
                G=G,
                z=w_pivot,
                shape_res=mesh_res,
                conditioning_params=pose.view(1, 25),
                level=level,
                truncation_cutoff=14,
                truncation_psi=0.7,  # self.hyperparameters.export_mesh_truncation_psi,
                noise_mode="const",
            )

        canon_mesh_sampled_fwd = split_likely_main_mesh_component(canon_mesh_sampled_fwd)
        canon_mesh_sampled_fn = "tmp_canon_mesh_sampled.obj"
        tformed_mesh = apply_transforms_to_sampled_mesh(canon_mesh_sampled_fwd)
        tformed_mesh.export(canon_mesh_sampled_fn)
        clean_inverted_mesh(canon_mesh_sampled_fn, tverts=tverts)

        canon_mesh_sampled_fwd = trimesh.load(canon_mesh_sampled_fn)

        return canon_mesh_sampled_fwd

    def export_canonical_modules_depthmap_mesh(self, generated_depths, mesh_res, lmks_list, image_name):
        canon_cams = self.MUDC.get_canonical_dmap_cams_for_rlhf()
        cam2world_matrix = canon_cams["cam2world_matrix"]
        intrinsics = canon_cams["intrinsics"]
        nrs = min(mesh_res, 256)  # has to be set or OOM error
        d2m = self.MUDC.convert_modules_depthmap_to_mesh(dmap_tensor=generated_depths.cpu().squeeze(0), camera_params=cam2world_matrix, HW=min(mesh_res, 256))

        canon_verts = np.asarray(d2m.vertices).reshape(min(mesh_res, 256), min(mesh_res, 256), -1)
        centres = [canon_verts[yc, xc] for (xc, yc) in lmks_list]

        d2m_canonical_lmks_fn = f"{self.paths_config.experiments_output_dir}/{image_name}_canon_mesh_canonical_lmks.npy"
        d2m_canonical_mesh_fn = f"{self.paths_config.experiments_output_dir}/{image_name}_canon_mesh_canonical.obj"

        d2m.export(d2m_canonical_mesh_fn)
        np.save(arr=centres, file=d2m_canonical_lmks_fn)

    def export_orig_view_rgb(self, w_pivot, pose, image_name):
        with torch.no_grad():
            gen_img, _ = self.forward(w_pivot, pose, eval=True, nrs_dmap=128)

        pil_img = self.topil(gen_img.clamp(-1, 1).mul(0.5).add(0.5).clamp(0, 1).squeeze(0))
        pilimg_fn = f"{self.paths_config.experiments_output_dir}/{image_name}_img_orig_pose.jpg"
        print("pausing here")
        pil_img.save(pilimg_fn)

    def find_lmk_intersections_with_sampled_depth_mesh(self, canon_mesh_sampled_fwd, lmks_for_rays, ray_origins, ray_directions, image_name):
        dict_of_idx_now_challenge = omegaconf.DictConfig(self.get_lmk_idx_for_now_challenge())
        lkeys = list(dict_of_idx_now_challenge.keys())

        tms = trimesh.ray.ray_triangle.RayMeshIntersector(canon_mesh_sampled_fwd)

        cm_centers = canon_mesh_sampled_fwd.triangles_center
        lmk_origins = []
        lmk_dirs = []
        intersecting_triangles = []

        intersecting_barycenters = []

        lmks_list = []
        for lk in lkeys:
            idx = dict_of_idx_now_challenge[lk].landmark_index
            lmk = lmks_for_rays[idx].to(torch.int32).cpu()  # lmk should be (x,y) coord. can then get ray origin /dir

            lmks_list.append(lmk)
            lmk_origin = ray_origins[0, lmk[1], lmk[0], :]
            lmk_dir = ray_directions[0, lmk[1], lmk[0], :]

            lmk_origins.append(lmk_origin)
            lmk_dirs.append(lmk_dir)

            intersecting_triangle = tms.intersects_first(ray_origins=[lmk_origin.cpu().numpy()], ray_directions=[lmk_dir.cpu().numpy()])

            intersecting_triangles.append(intersecting_triangle)

            barycenter = cm_centers[intersecting_triangle]
            intersecting_barycenters.append(barycenter)

        nbc = np.concatenate(intersecting_barycenters)

        nbc_lmks_fn = f"{self.paths_config.experiments_output_dir}/{image_name}_canon_mesh_sampled_lmks.npy"
        np.save(arr=nbc, file=nbc_lmks_fn)

    def export_all_lmk_intersections_with_sampled_depth_mesh(self, lmks_for_rays, canon_mesh_sampled_fwd, ray_origins, ray_directions, image_name):
        tms = trimesh.ray.ray_triangle.RayMeshIntersector(canon_mesh_sampled_fwd)

        cm_centers = canon_mesh_sampled_fwd.triangles_center
        all_intersections = self.get_ray_intersections_with_marching_cubes_mesh(lmks_for_rays, tms, ray_origins, ray_directions, cm_centers)
        all_intersections_fn = f"{self.paths_config.experiments_output_dir}/{image_name}_canon_sampled_mesh_all_98_lmks.pt"
        ai = torch.from_numpy(all_intersections)
        torch.save(f=all_intersections_fn, obj=ai)

    def get_ray_intersections_with_marching_cubes_mesh(self, lmks_for_rays, tms, ray_origins_pinhole, ray_directions_pinhole, cm_centers):
        intersections = []

        if hasattr(tqdm, "tqdm"):
            counter_idea = tqdm.tqdm
        else:
            counter_idea = tqdm
        for idx in counter_idea(range(len(lmks_for_rays))):
            lmk = lmks_for_rays[idx].to(torch.int32).cpu()
            lmk_origin = ray_origins_pinhole[0, lmk[1], lmk[0], :]
            lmk_dir = ray_directions_pinhole[0, lmk[1], lmk[0], :]
            intersecting_triangle = tms.intersects_first(ray_origins=[lmk_origin.cpu().numpy()], ray_directions=[lmk_dir.cpu().numpy()])
            intersection_pt = cm_centers[intersecting_triangle]
            intersections.append(intersection_pt)
        intersections = np.concatenate(intersections)
        intersections = torch.from_numpy(intersections)
        return intersections

    def get_all_ray_intersections_with_mesh_inv(self, tms, ray_origins, ray_directions, cm_centers, mesh_res):
        intersections = []

        if hasattr(tqdm, "tqdm"):
            counter_idea = tqdm.tqdm
        else:
            counter_idea = tqdm

        xy = [(x, y) for x in range(ray_origins.shape[1]) for y in range(ray_origins.shape[2])]

        for cc in counter_idea(xy):
            # lmk=lmks_for_rays[idx].to(torch.int32).cpu()
            lmk_origin = ray_origins[0, cc[0], cc[1], :]
            lmk_dir = ray_directions[0, cc[0], cc[1], :]
            intersecting_triangle = tms.intersects_first(ray_origins=[lmk_origin.cpu().numpy()], ray_directions=[lmk_dir.cpu().numpy()])
            intersection_pt = cm_centers[intersecting_triangle]
            intersections.append(intersection_pt)
        intersections = np.concatenate(intersections)
        intersections = torch.from_numpy(intersections)
        return intersections

    def get_all_ray_intersections_with_mesh(self, tms, ray_origins, ray_directions, cm_centers, mesh_res):
        intersections = []

        if hasattr(tqdm, "tqdm"):
            counter_idea = tqdm.tqdm
        else:
            counter_idea = tqdm

        xy = [(x, y) for x in range(mesh_res) for y in range(mesh_res)]

        for cc in counter_idea(xy):
            # lmk=lmks_for_rays[idx].to(torch.int32).cpu()
            lmk_origin = ray_origins[0, cc[0], cc[1], :]
            lmk_dir = ray_directions[0, cc[0], cc[1], :]
            # lmk_origin=ray_origins[0,cc[0],cc[1],:]
            # lmk_dir=ray_directions[0,cc[0],cc[1],:]
            intersecting_triangle = tms.intersects_first(ray_origins=[lmk_origin.cpu().numpy()], ray_directions=[lmk_dir.cpu().numpy()])
            intersection_pt = cm_centers[intersecting_triangle]
            intersections.append(intersection_pt)
        intersections = np.concatenate(intersections)
        intersections = torch.from_numpy(intersections)
        return intersections

    def return_all_lmk_intersections_with_sampled_depth_mesh(self, lmks_for_rays, canon_mesh_sampled_fwd, ray_origins, ray_directions):
        tms = trimesh.ray.ray_triangle.RayMeshIntersector(canon_mesh_sampled_fwd)
        cm_centers = canon_mesh_sampled_fwd.triangles_center
        all_intersections = self.get_ray_intersections_with_marching_cubes_mesh(lmks_for_rays, tms, ray_origins, ray_directions, cm_centers)
        return all_intersections

    def find_lmk_intersections_with_sampled_depth_mesh_filtrum(self, canon_mesh_sampled_fwd, lmks_for_rays, ray_origins, ray_directions, image_name):
        dict_of_idx_now_challenge = omegaconf.DictConfig(self.get_lmk_idx_for_now_challenge())
        lkeys = list(dict_of_idx_now_challenge.keys())

        tms = trimesh.ray.ray_triangle.RayMeshIntersector(canon_mesh_sampled_fwd)

        cm_centers = canon_mesh_sampled_fwd.triangles_center
        lmk_origins = []
        lmk_dirs = []
        intersecting_triangles = []

        intersecting_barycenters = []

        lmks_list = []
        for lk in lkeys:
            idx = dict_of_idx_now_challenge[lk].landmark_index
            lmk = lmks_for_rays[idx].to(torch.int32).cpu()  # lmk should be (x,y) coord. can then get ray origin /dir

            lmks_list.append(lmk)
            lmk_origin = ray_origins[0, lmk[1], lmk[0], :]
            lmk_dir = ray_directions[0, lmk[1], lmk[0], :]

            lmk_origins.append(lmk_origin)
            lmk_dirs.append(lmk_dir)

            intersecting_triangle = tms.intersects_first(ray_origins=[lmk_origin.cpu().numpy()], ray_directions=[lmk_dir.cpu().numpy()])

            intersecting_triangles.append(intersecting_triangle)

            barycenter = cm_centers[intersecting_triangle]
            intersecting_barycenters.append(barycenter)

        nbc = np.concatenate(intersecting_barycenters)

        nbc_lmks_fn = f"{self.paths_config.experiments_output_dir}/{image_name}_canon_mesh_sampled_lmks_filtrum.npy"
        np.save(arr=nbc, file=nbc_lmks_fn)

    def export_all_lmk_intersections_with_sampled_depth_mesh_filtrum(self, lmks_for_rays, canon_mesh_sampled_fwd, ray_origins, ray_directions, image_name):
        tms = trimesh.ray.ray_triangle.RayMeshIntersector(canon_mesh_sampled_fwd)

        cm_centers = canon_mesh_sampled_fwd.triangles_center
        all_intersections = self.get_ray_intersections_with_marching_cubes_mesh(lmks_for_rays, tms, ray_origins, ray_directions, cm_centers)
        all_intersections_fn = f"{self.paths_config.experiments_output_dir}/{image_name}_canon_sampled_mesh_all_98_lmks_filtrum.pt"
        ai = torch.from_numpy(all_intersections)
        torch.save(f=all_intersections_fn, obj=ai)

    def export_canon_view_rgb(self, w_pivot, image_name):
        c = self.MUDC.get_triple_dmap_cams_for_rlhf()["gen_c"][1]
        with torch.no_grad():
            gen_img, _ = self.forward(w_pivot, c.cuda(), eval=True, nrs_dmap=128)
        pil_img = topil(gen_img.clamp(-1, 1).mul(0.5).add(0.5).clamp(0, 1).squeeze(0))
        pilimg_fn = f"{self.paths_config.experiments_output_dir}/{image_name}_img_canonical_pose.jpg"
        pil_img.save(pilimg_fn)

    def export_triple_modules_depthmap(self, w_pivot, image_name):
        tdmap_gens = []
        tdc = self.MUDC.get_triple_dmap_cams_for_rlhf()["gen_c"]
        dmap_res = 128
        for c in tdc:
            with torch.no_grad():
                _, generated_depths = self.forward(w_pivot, c.cuda(), eval=True, nrs_dmap=dmap_res)
            tdmap_gens.append(generated_depths)

        trip_dmap = torch.cat(tdmap_gens, 0)  # [3,1,128,128]
        tdmap_fn = f"{self.paths_config.experiments_output_dir}/{image_name}_tdmap.pt"
        torch.save(obj=trip_dmap, f=tdmap_fn)

    def get_aw98_lmks_from_pinhole_camera(self, mesh_res, generated_images):
        # this is using the pinhole camera model
        # ray_origins, ray_directions = MUDC.ray_sampler_static(cam2world_matrix, intrinsics, nrs)
        # ray_origins_pinhole=ray_origins.reshape(-1,nrs,nrs,3)
        # ray_directions_pinhole=ray_directions.reshape(-1,nrs,nrs,3)

        generated_images = generated_images.clamp(-1, 1) / 2 + 0.5
        im_for_lmk = torch.nn.functional.interpolate(generated_images, size=(256, 256), align_corners=True, mode="bilinear")
        lmks_for_rays = self.AW98_MODULE.predict_landmarks_from_rgb_on_gpu(im_for_lmk, detach=True)

        return lmks_for_rays

    def get_pinhole_ray_origins_directions(self, mesh_res, MUDC):
        canon_cams = MUDC.get_canonical_dmap_cams_for_rlhf()
        cam2world_matrix = canon_cams["cam2world_matrix"]
        intrinsics = canon_cams["intrinsics"]
        nrs = min(mesh_res, 256)  # has to be set or OOM error

        # this is using the pinhole camera model
        ray_origins, ray_directions = MUDC.ray_sampler_static(cam2world_matrix, intrinsics, nrs)
        ray_origins_pinhole = ray_origins.reshape(-1, nrs, nrs, 3)
        ray_directions_pinhole = ray_directions.reshape(-1, nrs, nrs, 3)

        return (ray_origins_pinhole, ray_directions_pinhole)

    def perform_idw_shifting_based_on_filtrum(self, original_mesh_in_trimesh_format, keypoints, further_offset=0.0):
        # filtrum of nose frmo faacial keypints
        kpts = keypoints

        old_mesh = original_mesh_in_trimesh_format

        bb = old_mesh.bounds
        minz = bb[0][2]
        minx = bb[0][1]
        miny = bb[0][0]
        maxx = bb[1][1]
        maxy = bb[1][0]

        minmin_xyz = [minx, miny, minz]
        minmax_xyz = [minx, maxy, minz]
        maxmin_xyz = [maxx, miny, minz]
        maxmax_xyz = [maxx, maxy, minz]

        mm = np.array([minmin_xyz, minmax_xyz, maxmin_xyz, maxmax_xyz])

        # print(kpts[90])

        violating_kpts = kpts[kpts[:, 2] > kpts[90, 2]]

        # print(violating_kpts.shape)

        residuals = kpts[90] - violating_kpts

        residuals_z = residuals[:, 2]

        # print(residuals_z)

        # print(kpts[55])
        # print(kpts[59])

        # get that mean filtrum offset concept
        mean_filtrum_z = (kpts[55, 2] + kpts[59, 2]) / 2 + further_offset

        # print('mean filtrum z pos')
        # print(mean_filtrum_z)

        mesh_points = np.asarray(old_mesh.vertices)

        control_kpts = kpts.clone()

        control_kpts[56, 2] = mean_filtrum_z
        control_kpts[57, 2] = mean_filtrum_z
        control_kpts[58, 2] = mean_filtrum_z

        undeformed_kpts = kpts

        fp = torch.arange(undeformed_kpts.shape[0])[0:50]
        sp = torch.arange(undeformed_kpts.shape[0])[55:]
        c_kpts = torch.hstack((fp, sp))

        fp = torch.arange(undeformed_kpts.shape[0])[0:50]
        sp = torch.arange(undeformed_kpts.shape[0])[55:]
        c_kpts = torch.hstack((fp, sp))

        undeformed = np.concatenate((mm, undeformed_kpts[[56, 57, 58]].cpu().numpy()), axis=0)
        deformed = np.concatenate((mm, control_kpts[[56, 57, 58]].cpu().numpy()), axis=0)

        import pygem

        idw = pygem.IDW(original_control_points=undeformed, deformed_control_points=deformed)

        new_mesh_points = idw(mesh_points)

        # trimesh.Trimesh(vertices=self.points, faces=self.faces)

        new_mesh = trimesh.Trimesh(vertices=new_mesh_points, faces=old_mesh.faces, process=False)

        return new_mesh

    def return_perpendicular_ray_intersections_with_mesh(self, mesh, mesh_res):
        ray_origins, ray_directions = self.get_rays_parallel(mesh_res)
        tms = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        cm_centers = mesh.triangles_center
        all_intersections = self.get_all_ray_intersections_with_mesh(tms, ray_origins, ray_directions, cm_centers, mesh_res)
        return all_intersections

    def return_perpendicular_ray_intersections_with_mesh_half_xy(self, mesh, mesh_res):
        ray_origins, ray_directions = self.get_rays_parallel_half_xy(mesh_res)
        tms = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        cm_centers = mesh.triangles_center
        all_intersections = self.get_all_ray_intersections_with_mesh(tms, ray_origins, ray_directions, cm_centers, mesh_res)
        return all_intersections

    def return_all_ray_intersections(self, mesh, mesh_res, ray_origins, ray_directions):
        # ray_origins,ray_directions=self.get_rays_parallel_half_xy(mesh_res)
        tms = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        cm_centers = mesh.triangles_center
        all_intersections = self.get_all_ray_intersections_with_mesh(tms, ray_origins, ray_directions, cm_centers, mesh_res)
        return all_intersections

    def return_all_ray_intersections_sf(self, mesh, mesh_res, ray_origins, ray_directions):
        # ray_origins,ray_directions=self.get_rays_parallel_half_xy(mesh_res)
        tms = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        cm_centers = mesh.triangles_center
        all_intersections = self.get_all_ray_intersections_with_mesh_inv(tms, ray_origins, ray_directions, cm_centers, mesh_res)
        return all_intersections

    def run_exit_sequence(self, G, w_pivot, pose):
        rv_dict = {}

        canonical_pose = self.MUDC.get_canonical_dmap_cams_for_rlhf()["gen_c"].reshape((1, 25)).float().cuda()

        # -----------------------------------------------------------------------------------
        #
        # MARCHING CUBES
        #
        # -----------------------------------------------------------------------------------

        level = self.hyperparameters.sampled_modules_depthmap_isolevel  # should be 10 by default.
        tverts = self.hyperparameters.sampled_mesh_target_n_vertices  # roughly 60k...if too large fitting procedure may not converge
        mesh_res = self.hyperparameters.depthmap_res  # resolution for mesh like...64 or 128 or 256 or 512

        if self.hyperparameters.export_sampled_modules_depthmap_mesh:
            canon_mesh_sampled_fwd = self.export_marching_cubes_mesh(w_pivot, mesh_res, canonical_pose, level, image_name, tverts)

        # -----------------------------------------------------------------------------------
        #
        # GENERATING SOME IMAGES FOR LANDMARKS
        #
        # -----------------------------------------------------------------------------------

        with torch.no_grad():
            generated_images, _ = G.synthesis(w, canonical_pose, noise_mode=noise_mode, neural_rendering_resolution=128, force_fp32=True)  # change here AM 20_06_2023
            _, generated_depths = G.synthesis(w, canonical_pose, noise_mode=noise_mode, neural_rendering_resolution=128, force_fp32=True)  # change here AM 20_06_2023

        # rescale appropriately..
        lmks_for_rays = self.get_aw98_lmks_from_pinhole_camera(mesh_res, generated_images)
        lmks_for_rays = (lmks_for_rays * min(mesh_res, 256) / 256).to(torch.int32)

        rv_dict["lmks_for_rays"] = lmks_for_rays

        dict_of_idx_now_challenge = omegaconf.DictConfig(self.get_lmk_idx_for_now_challenge())
        lkeys = list(dict_of_idx_now_challenge.keys())

        lmks_list = []
        for lk in lkeys:
            idx = dict_of_idx_now_challenge[lk].landmark_index
            lmk = lmks_for_rays[idx].to(torch.int32).cpu()  # lmk should be (x,y) coord. can then get ray origin /dir
            lmks_list.append(lmk)

        ray_origins_sampled, ray_directions_sampled = self.get_rays_parallel_half_xy(nrs)

        angle = 90
        with torch.no_grad():
            gen_perp = self.G.synthesis_with_custom_rays(ws=w_pivot, c=canonical_pose, ray_origins=ray_origins_sampled.reshape(1, nrs * nrs, 3), ray_directions=ray_directions_sampled.reshape(1, nrs * nrs, 3), neural_rendering_resolution=min(mesh_res, 256))
            # imo=topil(gen_perp['image'].squeeze(0).clamp(-1,1)/2+0.5)
            out = TF.rotate(gen_perp["image_raw"], angle)
            imo = self.topil(out.squeeze(0).clamp(-1, 1) / 2 + 0.5)
            sampled_rgb_for_lmks_fn = f"{self.paths_config.experiments_output_dir}/{image_name}_sampled_mesh_rgb_for_lmks.jpg"
            imo.save(sampled_rgb_for_lmks_fn)

            # saving full resolution image also
            self.topil(TF.rotate(gen_perp["image"], angle).squeeze(0).clamp(-1, 1) / 2 + 0.5).save(f"{self.paths_config.experiments_output_dir}/{image_name}_sampled_mesh_rgb_for_lmks_fullres.jpg")
            im_for_lmk = torch.nn.functional.interpolate(gen_perp["image"].clamp(-1, 1) / 2 + 0.5, size=(256, 256), align_corners=True, mode="bilinear")
            lmks_for_rays_perp = AW98_MODULE.predict_landmarks_from_rgb_on_gpu(im_for_lmk, detach=True)
            lmks_for_rays_perp_fn = f"{self.paths_config.experiments_output_dir}/{image_name}_lmks_for_rays_perp_98.pt"
            torch.save(obj=lmks_for_rays_perp, f=lmks_for_rays_perp_fn)

        if self.hyperparameters.export_sampled_modules_depthmap_mesh:
            self.find_lmk_intersections_with_sampled_depth_mesh(canon_mesh_sampled_fwd, lmks_for_rays_perp, ray_origins=ray_origins_sampled.reshape(1, nrs, nrs, 3), ray_directions=ray_directions_sampled.reshape(1, nrs, nrs, 3), image_name=image_name)
            self.export_all_lmk_intersections_with_sampled_depth_mesh(lmks_for_rays_perp, canon_mesh_sampled_fwd, ray_origins=ray_origins_sampled.reshape(1, nrs, nrs, 3), ray_directions=ray_directions_sampled.reshape(1, nrs, nrs, 3), image_name=image_name)

            all_intersections_fn = f"{self.paths_config.experiments_output_dir}/{image_name}_canon_sampled_mesh_all_98_lmks.pt"
            kpts = torch.load(all_intersections_fn)
            mesh_adjusted_filtrum = self.perform_idw_shifting_based_on_filtrum(original_mesh_in_trimesh_format=canon_mesh_sampled_fwd, keypoints=kpts, further_offset=self.hyperparameters.filtrum_offset)
            canon_mesh_sampled_fn = f"{self.paths_config.experiments_output_dir}/{image_name}_canon_mesh_sampled_adjusted_filtrum.obj"
            mesh_adjusted_filtrum.export(canon_mesh_sampled_fn)
            self.find_lmk_intersections_with_sampled_depth_mesh_filtrum(mesh_adjusted_filtrum, lmks_for_rays_perp, ray_origins=ray_origins_sampled.reshape(1, nrs, nrs, 3), ray_directions=ray_directions_sampled.reshape(1, nrs, nrs, 3), image_name=image_name)
            self.export_all_lmk_intersections_with_sampled_depth_mesh_filtrum(lmks_for_rays_perp, canon_mesh_sampled_fwd, ray_origins=ray_origins_sampled.reshape(1, nrs, nrs, 3), ray_directions=ray_directions_sampled.reshape(1, nrs, nrs, 3), image_name=image_name)

        if self.hyperparameters.export_3dmm_realigned_sampled_mesh:
            # load it up

            alignment_lmks = np.load(f"{self.paths_config.experiments_output_dir}/{image_name}_canon_mesh_sampled_lmks.npy")

            # canon_mesh_sampled_fwd

            vvs = get_idx_of_verts_by_dist(canon_mesh_sampled_fwd, alignment_lmks)

            all_intersections_fn = f"{self.paths_config.experiments_output_dir}/{image_name}_canon_sampled_mesh_all_98_lmks.pt"
            kpts_98 = torch.load(all_intersections_fn)

            sample_kpts_as_68_list = get_98_mesh_lmks_in_terms_of_68(kpts_98)
            c_points = sample_kpts_as_68_list.numpy()
            t_points = get_68_reference_kpts_from_3dmm_basis().numpy()

            tmsh_c = canon_mesh_sampled_fwd.copy()

            # https://vedo.embl.es/docs/vedo.html?search=non+rigid

            a_cloud = vedo.pointcloud.Points(tmsh_c.vertices)
            a_cloud.align_with_landmarks(source_landmarks=c_points, target_landmarks=t_points, least_squares=False, affine=False)
            tmsh_c.vertices = a_cloud.vertices

            new_lmks = []
            for vv in vvs:
                new_lmks.append(tmsh_c.vertices[vv])
            new_lmks = np.array(new_lmks)
            canon_mesh_sampled_fn = f"{self.paths_config.experiments_output_dir}/{image_name}_canon_mesh_sampled_adjusted_3dmm_basis.obj"
            tmsh_c.export(canon_mesh_sampled_fn)
            alignment_lmks_fn = f"{self.paths_config.experiments_output_dir}/{image_name}_canon_mesh_sampled_lmks_3dmm_basis.npy"
            np.save(arr=new_lmks, file=alignment_lmks_fn)

            rff = MUDC.sample_radiance_field_sigma_rgb_from_ws(self.G, ws=w_pivot, c=torch.zeros((1, 25)).cuda())

            rff_fn = f"{self.paths_config.experiments_output_dir}/{image_name}_radiance_field_final.pt"
            torch.save(obj=rff, f=rff_fn)

    def create_samples_finer(self, N=256, voxel_origin=[0, 0, 0], cube_length=[0.5, 0.5, 0.5]):
        cube_length = np.array(cube_length)
        # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
        # voxel_origin = np.array(voxel_origin) - cube_length/2
        voxel_origin = np.array([0.0, -0.25, -0.25])
        voxel_sizes = cube_length / (N - 1)
        overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
        samples = torch.zeros(N**3, 3)
        # transform first 3 columns
        # to be the x, y, z index
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.float() / N) % N
        samples[:, 0] = ((overall_index.float() / N) / N) % N
        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_sizes[2]) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_sizes[1]) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_sizes[0]) + voxel_origin[0]
        num_samples = N**3
        return samples.unsqueeze(0), voxel_origin, voxel_sizes

    def get_rays_parallel_half_xy(self, mesh_res):
        samples, voxel_origin, voxel_size = self.create_samples_finer(N=mesh_res, voxel_origin=[0, 0, 0], cube_length=[0.5, 0.5, 0.5])  # .reshape(1, -1, 3)
        samples = samples.to(torch.device("cuda"))
        samples = samples[:, :, :][samples[:, :, -1] == 0.0]
        samples = samples.reshape(1, -1, 3)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], samples.shape[1], 3), device=torch.device("cuda"))
        transformed_ray_directions_expanded[..., -1] = -1.0
        nrs = mesh_res
        ray_origins_sampled = samples.reshape(-1, nrs, nrs, 3)
        ray_origins_sampled[:, :, :, -1] = 2.7
        ray_directions_sampled = transformed_ray_directions_expanded.reshape(-1, nrs, nrs, 3)
        return ray_origins_sampled, ray_directions_sampled

    def get_rays_parallel_half_xy_samples(self, mesh_res):
        samples, voxel_origin, voxel_size = self.create_samples_finer(N=mesh_res, voxel_origin=[0.5, 0, 0], cube_length=[0.5, 0.5, 0.5])  # .reshape(1, -1, 3)
        # samples = samples.to(torch.device('cuda'))
        # samples=samples[:,:,:][samples[:,:,-1]==-0.5]
        # samples=samples.reshape(1,-1,3)
        return samples

    def create_samples(self, N=256, voxel_origin=[0, 0, 0], cube_length=1.0):
        return mutils.create_samples(N=N, voxel_origin=voxel_origin, cube_length=cube_length)

    def get_rays_parallel(self, mesh_res):
        samples, voxel_origin, voxel_size = self.create_samples(N=mesh_res, voxel_origin=[0, 0, 0], cube_length=1.0)  # .reshape(1, -1, 3)
        samples = samples.to(torch.device("cuda"))
        samples = samples[:, :, :][samples[:, :, -1] == -0.5]
        samples = samples.reshape(1, -1, 3)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], samples.shape[1], 3), device=torch.device("cuda"))
        transformed_ray_directions_expanded[..., -1] = -1.0
        nrs = mesh_res
        ray_origins_sampled = samples.reshape(-1, nrs, nrs, 3)
        ray_origins_sampled[:, :, :, -1] = 2.7
        ray_directions_sampled = transformed_ray_directions_expanded.reshape(-1, nrs, nrs, 3)
        return ray_origins_sampled, ray_directions_sampled

    def get_idx_of_verts_by_dist(self, original_mesh, orig_lmks):
        vvs = []

        for omk in orig_lmks:
            dists = np.linalg.norm(original_mesh.vertices - omk, ord=2, axis=-1)
            sel_v = np.argmin(dists)
            vvs.append(sel_v)

        return vvs

    def transform_nose_mouth_corners(self, input_trimesh, kpts_nearest):
        input_trimesh.export("tformed.obj")
        qt_list = [0.02, 0.015, 0.01]
        pc_list = [50, 50, 50]
        num_list = [20, 20, 20]

        for qt, pc, num in zip(qt_list, pc_list, num_list):
            newset = pymeshlab.MeshSet()
            newset.load_new_mesh("tformed.obj")
            m_bk = trimesh.load("tformed.obj")
            newset.set_current_mesh(0)
            ms = newset.current_mesh()

            # nose smoothing

            sel_pt = m_bk.vertices[kpts_nearest[54]]
            dists = np.linalg.norm(m_bk.vertices - sel_pt, ord=2, axis=1)
            sel_dists = np.zeros_like(dists).astype(bool)
            sel_dists[dists < np.quantile(dists, qt)] = True
            idx_of = np.arange(sel_dists.shape[0])[sel_dists]

            cond_str = [f"vi=={k} || " for k in idx_of.tolist()]
            join_cond = "".join(cond_str)[:-4]
            newset.compute_selection_by_condition_per_vertex(condselect=join_cond)
            ms = newset.current_mesh()
            ms.vertex_selection_array().sum()

            newset.apply_coord_depth_smoothing(stepsmoothnum=num, viewpoint=[0, 0, -1], selected=True, delta=pymeshlab.PercentageValue(pc))
            # left dimple smoothing

            sel_pt = m_bk.vertices[kpts_nearest[76]]
            dists = np.linalg.norm(m_bk.vertices - sel_pt, ord=2, axis=1)

            sel_dists = np.zeros_like(dists).astype(bool)
            sel_dists[dists < np.quantile(dists, qt)] = True

            eps = 0.02
            sel_dists[m_bk.vertices[:, 0] > sel_pt[0] + eps] = False  # select strictly to left of the key point only...plus eps

            idx_of = np.arange(sel_dists.shape[0])[sel_dists]
            cond_str = [f"vi=={k} || " for k in idx_of.tolist()]
            join_cond = "".join(cond_str)[:-4]
            newset.compute_selection_by_condition_per_vertex(condselect=join_cond)
            ms = newset.current_mesh()
            ms.vertex_selection_array().sum()

            newset.apply_coord_depth_smoothing(stepsmoothnum=num, viewpoint=[0, 0, -1], selected=True, delta=pymeshlab.PercentageValue(pc))
            # #right dimple smoothing

            sel_pt = m_bk.vertices[kpts_nearest[82]]
            dists = np.linalg.norm(m_bk.vertices - sel_pt, ord=2, axis=1)

            sel_dists = np.zeros_like(dists).astype(bool)
            sel_dists[dists < np.quantile(dists, qt)] = True

            eps = 0.01
            sel_dists[m_bk.vertices[:, 0] < sel_pt[0] - eps] = False  # select strictly to left of the key point only...plus eps
            idx_of = np.arange(sel_dists.shape[0])[sel_dists]
            # print(len(idx_of))

            cond_str = [f"vi=={k} || " for k in idx_of.tolist()]
            join_cond = "".join(cond_str)[:-4]
            newset.compute_selection_by_condition_per_vertex(condselect=join_cond)
            ms = newset.current_mesh()
            ms.vertex_selection_array().sum()

            newset.apply_coord_depth_smoothing(stepsmoothnum=num, viewpoint=[0, 0, -1], selected=True, delta=pymeshlab.PercentageValue(pc))
            # newset.apply_coord_taubin_smoothing(stepsmoothnum=num,selected=True)#,normalthr=90)

        newset.save_current_mesh("tformed.obj")
        newset.clear()

        tmsh = trimesh.load("tformed.obj")
        os.remove("tformed.obj")
        return tmsh

    def get_3d_lmks_from_rf_mesh(self, seed, G):
        import omegaconf
        import PIL

        topil = TF.to_pil_image

        rv_dict = {}
        canonical_pose = self.MUDC.get_canonical_dmap_cams_for_rlhf()["gen_c"].reshape((1, 25)).float().cuda()

        # -----------------------------------------------------------------------------------
        #
        # MARCHING CUBES
        #
        # -----------------------------------------------------------------------------------

        level = 10  # self.hyperparameters.sampled_modules_depthmap_isolevel #should be 10 by default.
        tverts = 60000  # self.hyperparameters.sampled_mesh_target_n_vertices #roughly 60k...if too large fitting procedure may not converge
        mesh_res = 256  # self.hyperparameters.depthmap_res #resolution for mesh like...64 or 128 or 256 or 512
        nrs = mesh_res

        noise_mode = "const"

        torch.manual_seed(seed)

        zs = torch.randn((1, 512)).float().cuda()
        with torch.no_grad():
            ws = G.mapping(zs, canonical_pose)

        canon_mesh_sampled_fwd = self.export_marching_cubes_mesh(G, ws, mesh_res, canonical_pose, level, tverts)
        # -----------------------------------------------------------------------------------
        #
        # GENERATING SOME IMAGES FOR LANDMARKS
        #
        # -----------------------------------------------------------------------------------

        with torch.no_grad():
            x = G.synthesis(ws, canonical_pose, noise_mode=noise_mode, neural_rendering_resolution=128, force_fp32=True)  # change here AM 20_06_2023
            generated_images = x["image"]
            generated_depths = x["image_depth"]

        # rescale appropriately..
        lmks_for_rays = self.get_aw98_lmks_from_pinhole_camera(mesh_res, generated_images)
        lmks_for_rays = (lmks_for_rays * min(mesh_res, 256) / 256).to(torch.int32)

        rv_dict["lmks_for_rays"] = lmks_for_rays

        dict_of_idx_now_challenge = omegaconf.DictConfig(self.get_lmk_idx_for_now_challenge())
        lkeys = list(dict_of_idx_now_challenge.keys())

        lmks_list = []
        for lk in lkeys:
            idx = dict_of_idx_now_challenge[lk].landmark_index
            lmk = lmks_for_rays[idx].to(torch.int32).cpu()  # lmk should be (x,y) coord. can then get ray origin /dir
            lmks_list.append(lmk)

        # the code below can be used to get some rays that are taken perpendicular to the mesh. these should in theory be more correct for use with the alignment task!
        ray_origins_sampled, ray_directions_sampled = self.get_rays_parallel_half_xy(mesh_res)  # make face bigger.

        angle = 90
        with torch.no_grad():
            gen_perp = G.synthesis_with_custom_rays(ws=ws, c=canonical_pose, ray_origins=ray_origins_sampled.reshape(1, nrs * nrs, 3), ray_directions=ray_directions_sampled.reshape(1, nrs * nrs, 3), neural_rendering_resolution=min(mesh_res, 256))
            out = TF.rotate(gen_perp["image_raw"], angle)
            imo = topil(out.squeeze(0).clamp(-1, 1) / 2 + 0.5)
            im_for_lmk = torch.nn.functional.interpolate(gen_perp["image"].clamp(-1, 1) / 2 + 0.5, size=(256, 256), align_corners=True, mode="bilinear")
            imo = topil(im_for_lmk.squeeze(0))
            lmks_for_rays_perp = self.AW98_MODULE.predict_landmarks_from_rgb_on_gpu(im_for_lmk, detach=True)
            lmks_dict = {k: lmks_for_rays_perp[k].detach().cpu().numpy() for k in range(lmks_for_rays_perp.shape[0])}
            ppl = self.place_landmarks_on_image(imo, lmks_dict)  # shape (256,256,3)
            out = TF.rotate(torch.from_numpy(ppl).permute(2, 0, 1), angle).to(torch.uint8).permute(1, 2, 0)
            im_wit_lmks = PIL.Image.fromarray(out.detach().cpu().numpy())

        smoothed_nose_mesh = None
        kpts_3d = self.return_all_lmk_intersections_with_sampled_depth_mesh(lmks_for_rays_perp, canon_mesh_sampled_fwd, ray_origins=ray_origins_sampled.reshape(1, nrs, nrs, 3), ray_directions=ray_directions_sampled.reshape(1, nrs, nrs, 3))

        return dict(orig=canon_mesh_sampled_fwd, smoothed=smoothed_nose_mesh, lmks_2d=lmks_for_rays_perp, ray_origins_sampled=ray_origins_sampled, ray_directions_sampled=ray_directions_sampled, nrs=nrs, im_wit_lmks=im_wit_lmks, kpts_3d=kpts_3d)

    def place_landmarks_on_image(self, im, ld_dict):
        import cv2

        im = np.array(im)
        imsize = im.shape[0]
        marker_size = 2
        if imsize <= 128:
            marker_size = 1
        for k in ld_dict.keys():
            x, y = ld_dict[k]
            # cv2.circle(im, (int(x), int(y)), marker_size, (0, 0, 255), -1)
            cv2.circle(im, (int(x), int(y)), marker_size, (0, 0, 255), -1)
        return im


def cpad(pc, shape_res):
    # used to set integer corrdinates for the sigma field sampling based on the pads_vals dict offsets for the 6 dimensions eg top bottom left right front back
    cp = int(pc * shape_res)
    if cp == 0:
        return 1
    return cp


def unit_scale_center_mesh(tmsh_object):
    mesh = tmsh_object.copy()
    rescale = max(mesh.extents) / 2.0
    tform = [-(mesh.bounds[1][i] + mesh.bounds[0][i]) / 2.0 for i in range(3)]
    matrix = np.eye(4)
    matrix[:3, 3] = tform
    mesh.apply_transform(matrix)
    matrix = np.eye(4)
    matrix[:3, :3] /= rescale
    mesh.apply_transform(matrix)
    return mesh


def half_unit_scale_center_mesh_for_vis(tmsh_object, translate=[0.5, 0.5, 0.75]):
    mesh = tmsh_object.copy()

    rescale = max(mesh.extents) / 2.0
    tform = [-(mesh.bounds[1][i] + mesh.bounds[0][i]) / 2.0 for i in range(3)]
    matrix = np.eye(4)
    matrix[:3, 3] = tform
    mesh.apply_transform(matrix)
    matrix = np.eye(4)
    matrix[:3, :3] /= rescale
    mesh.apply_transform(matrix)

    matrix = np.eye(4)

    matrix[0, 3] = translate[0]
    matrix[1, 3] = translate[1]
    matrix[2, 3] = translate[2]
    matrix[:3, :3] /= 2
    mesh.apply_transform(matrix)
    return mesh


def clean_inverted_mesh(mesh_fn, tverts=60000):
    ms = pymeshlab.MeshSet()
    ms.clear()
    ms.load_new_mesh(mesh_fn)
    nverts = ms.mesh(0).vertex_number()

    if nverts < 35000:
        print("n vert /f ace before subdivision (ie smaller than 35k)")
        print(ms.mesh(0).face_number())
        print(ms.mesh(0).vertex_number())
        # we want n vertices greater or equal to n verts in now chal, which max out at ~35k

        ms.apply_filter("generate_sampling_stratified_triangle", samplenum=40000)
        ms.apply_filter("apply_coord_hc_laplacian_smoothing", samplenum=40000)
        # subdivide_inverted_mesh(mesh_fn)
        ms.save_current_mesh(mesh_fn)

    ms.clear()

    ms.load_new_mesh(mesh_fn)
    nverts = ms.mesh(0).vertex_number()

    targetperc = tverts / nverts

    print(ms.print_status())

    print("n vert /f ace before")
    print(ms.mesh(0).face_number())
    print(ms.mesh(0).vertex_number())

    ms.set_current_mesh(0)
    ms.apply_filter("meshing_decimation_quadric_edge_collapse", targetperc=targetperc)

    print("n vert /face after")

    print(ms.mesh(0).face_number())
    print(ms.mesh(0).vertex_number())

    ms.save_current_mesh(mesh_fn)

    ms.clear()

    print(f"saved cleaned mesh with pymesh lab:\t{mesh_fn}")


def place_landmarks_on_image(im, ld_dict):
    import cv2

    im = np.array(im)
    imsize = im.shape[0]
    marker_size = 2
    if imsize <= 128:
        marker_size = 1
    for k in ld_dict.keys():
        x, y = ld_dict[k]
        cv2.circle(im, (int(x), int(y)), marker_size, (0, 0, 255), -1)
    return im


def get_adj_lengths(mesh):
    theadjacency = trimesh.graph.connected_components(mesh.face_adjacency)
    lengths = [len(a) for a in theadjacency]
    return lengths


def split_likely_main_mesh_component(mesh):
    lens = get_adj_lengths(mesh)
    max_shp_idx = np.argsort(lens)[-1]
    cmf = mesh.split()[max_shp_idx]
    cmf.fill_holes()
    cmf.fix_normals()
    return cmf


def add_kobe_for_alpha_step(cmf):
    uc = trimesh.load("/media/krillman/1TB_DATA/codes/HFGI3D/kobe.obj")
    cmf = trimesh.boolean.union([cmf, uc])
    return cmf


def calc_alpha_shape(cmf, alpha_param=0.0103):
    os.makedirs("./tmp", exist_ok=True)
    cmf_fn = "./tmp/mesh_w_possible_interior_for_alpha.off"
    Q_fn = cmf_fn.replace(".off", "_calc_alpha.off")
    cmf.export(cmf_fn)
    il = alpha_param
    jl = il / 100
    P = Polyhedron_3(cmf_fn)
    Q = Polyhedron_3()
    CGAL_Alpha_wrap_3.alpha_wrap_3(P, il, jl, Q)
    Q.write_to_file(Q_fn)
    alpha_shape = trimesh.load(Q_fn)
    os.remove(Q_fn)
    os.remove(cmf_fn)
    return alpha_shape


def calc_distance_for_verts(alpha_shape, orig_shape):
    sdf_of_alpha = SDF(alpha_shape.vertices, alpha_shape.faces)
    orig_verts = np.asarray(orig_shape.vertices)
    dists = [sdf_of_alpha.calc(v) for v in orig_verts]
    return np.hstack(dists)


def plot_dists_hist(dists):
    plt.clf()
    plt.hist(dists, bins=30)
    plt.savefig("hello.jpg")
    Image.open("hello.jpg").show()


def get_mask_for_dists(dists, thresh_dists=0.01):
    mask = np.array([False] * len(dists))
    mask[dists < thresh_dists] = True
    return mask


def apply_transforms_to_sampled_mesh(sampled_mesh):
    angle = -math.pi / 2
    direction = [0, 1, 0]
    center = [0, 0, 0]
    rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)
    trans_matrix = trimesh.transformations.translation_matrix([0.0, 0.01, 0.0])
    sampled_mesh.apply_transform(rot_matrix).apply_transform(trans_matrix)
    return sampled_mesh


def process_and_clean_face_mesh(tformed_mesh, alpha_param=0.0103, thresh_dists=0.01, showplot=False):
    sampled_mesh_main = split_likely_main_mesh_component(mesh=tformed_mesh)
    cmf = add_kobe_for_alpha_step(sampled_mesh_main)  # adding that fine grained cube for alpha step. has a few vertices in it. a very fine one.
    alpha_shape = calc_alpha_shape(cmf, alpha_param=alpha_param)
    dists = calc_distance_for_verts(alpha_shape, orig_shape=cmf)

    if showplot:
        plot_dists_hist(dists)

    mask = get_mask_for_dists(dists, thresh_dists=thresh_dists)
    cmf.update_vertices(mask)
    cleaned_mesh = split_likely_main_mesh_component(cmf)

    return cleaned_mesh


def clean_sampled_mesh(in_mesh):
    sampled_mesh_main = split_likely_main_mesh_component(mesh=in_mesh)
    sampled_mesh_main = sampled_mesh_main
    cmf = add_kobe_for_alpha_step(sampled_mesh_main)  # adding that fine grained cube for alpha step. has a few vertices in it. a very fine one.
    cmf = split_likely_main_mesh_component(mesh=cmf)

    return cmf


def clean_sampled_mesh_pymeshlab(in_mesh_fn, alpha_fraction=0.003, offset_fraction=0.001, suffix="cleaned"):
    in_mesh = trimesh.load(in_mesh_fn)
    sampled_mesh_main = split_likely_main_mesh_component(mesh=in_mesh)
    cmf = add_kobe_for_alpha_step(sampled_mesh_main)  # adding that fine grained cube for alpha step. has a few vertices in it. a very fine one.
    cmf = split_likely_main_mesh_component(mesh=cmf)
    os.makedirs("./tmp/", exist_ok=True)
    cmf.export("./tmp/cmf_mesh.obj")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh("./tmp/cmf_mesh.obj")
    ms.generate_alpha_wrap(alpha_fraction=alpha_fraction, offset_fraction=offset_fraction)

    suff = suffix.replace("_", "")

    if len(suff) > 0:
        suff = "_" + suff
    ms.save_current_mesh(in_mesh_fn.replace(".obj", suff + ".obj"))
    return in_mesh_fn.replace(".obj", suff + ".obj")


# convett scene to mesh ie ike the airplane in shapenet!
def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in scene_or_mesh.geometry.values()))
    else:
        assert isinstance(mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh
