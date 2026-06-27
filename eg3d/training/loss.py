# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Authoritative EG3D loss implementations.

Current training entrypoints construct loss classes from this module via
`training.loss.*`. RLHF-specific duplicate snapshots still exist in sibling
files for legacy comparison, but this file is the active runtime surface.
"""

import autoroot  # noqa: F401

import logging
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

try:
    import wandb
except ModuleNotFoundError:
    wandb = None
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix, upfirdn2d
from training.dual_discriminator import filtered_resizing
from training.volumetric_rendering.ray_sampler import RaySampler

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------

# code for RLHF_AM


rs = RaySampler()

import glob

import cv2
import dlib

# Load the pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
_DEFAULT_DLIB_LANDMARK_MODEL = Path(__file__).resolve().parents[2] / "external" / "dlib" / "shape_predictor_5_face_landmarks.dat"
_DLIB_LANDMARK_MODEL = Path(os.environ.get("DLIB_LANDMARK_MODEL", _DEFAULT_DLIB_LANDMARK_MODEL))
if not _DLIB_LANDMARK_MODEL.exists():
    raise FileNotFoundError(f"Missing dlib landmark model. Set DLIB_LANDMARK_MODEL or place shape_predictor_5_face_landmarks.dat at {_DEFAULT_DLIB_LANDMARK_MODEL}.")
predictor = dlib.shape_predictor(str(_DLIB_LANDMARK_MODEL))

import sys

import matplotlib.pyplot as plt
import omegaconf
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Legacy note: reward-model code used to live under eg3d/RLHF_nbooks before the refactor.


def depth_map_to_points(c, image_depth, neural_rendering_resolution):
    imd = image_depth
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
    dd_n = dd_np.reshape(-1, 3)
    dd_n[:, 2] -= 0.4
    depth_points = dd[:, np.linalg.norm(dd_n, ord=2, axis=1) <= 0.385, :]  # ,:]
    if depth_points.shape[1] > 0:
        depth_points[:, :, 2] += 0.4
    else:
        depth_points = depth_points
    return depth_points


# returns estimated normals (detached from computation graph gradient)
def clone_pcd_estimate_normals(pcd, dummy_tensor=None):
    pcd_o3d = o3d.geometry.PointCloud()

    pcd_o3d.points = o3d.utility.Vector3dVector(pcd.detach().cpu().numpy())
    pcd_o3d.estimate_normals()
    pcd_o3d.orient_normals_towards_camera_location(camera_location=[0, 0, 2.7])
    normals = torch.from_numpy(np.asarray(pcd_o3d.normals)).float()

    return normals


# converts the image to a point cloud given some depth values
def imd_to_xyz(image_depth, ray_origins, ray_directions, neural_rendering_resolution):
    final_dim = neural_rendering_resolution * neural_rendering_resolution
    imd = image_depth.view(-1, final_dim).unsqueeze(2).expand(1, final_dim, 3)
    retval = ray_origins + imd * ray_directions
    return retval


def get_list_of_z(seeds, zdim=512, use_fat_tail=False):
    # print(f'synthing z list with {len(seeds)} seeds and use fat tail: {use_fat_tail}')
    zd = 512
    device = torch.device("cuda")
    zs = [torch.from_numpy(np.random.RandomState(s).randn(1, 512)).to(device) for s in seeds]

    z_list = torch.cat(zs)
    # print('z list synth, returning')
    return z_list


USING_POINTNET_REWARD = False

USING_DMAP_REWARD = False

USING_DMAP_REWARD_3 = True

assert not (USING_POINTNET_REWARD and USING_DMAP_REWARD and USING_DMAP_REWARD_3)


STATIC_CONFIGS_DIR = Path(os.environ["STATIC_CONFIGS_DIR"])
RLHF_TUNE_CONFIG_DIR = Path(__file__).resolve().parent / "rlhf_tune_configs"
DEFAULT_RWD_SIGMA_PADS_CONFIG = str(STATIC_CONFIGS_DIR / "pads_vals_entire.yaml")
SIGMA_FIELD_REWARD_DTYPES = {"sigma_field_512", "sigma_field_256", "nose_512"}


# ----------------------------------------------------------------------------


def _compute_geometric_presurface_mask(sigma_old):
    """Per-ray pre-surface (camera-side) mask from the orig sigma cube.

    Treats axis 2 as world-z with iz=0 = FAR, iz=N-1 = NEAR-camera. For each
    (ix, iy) ray, evaluates the standard volume-rendering depth integral on the
    orig sigma and returns mask=1 for voxels strictly closer to the camera than
    the rendered depth surface. Background rays (opacity <= 0.5) get mask=1
    everywhere along the ray.
    """
    N = sigma_old.shape[-1]
    sigma_rev = sigma_old.flip(-1)
    density = torch.nn.functional.softplus(sigma_rev)
    alpha = 1.0 - torch.exp(-density)
    T_cum = torch.cumprod(1.0 - alpha, dim=-1)
    T = torch.cat([torch.ones_like(T_cum[..., :1]), T_cum[..., :-1]], dim=-1)
    weights = T * alpha
    k_grid_rev = torch.arange(N, device=sigma_old.device, dtype=sigma_old.dtype).view(1, 1, N)
    depth_rev = (weights * k_grid_rev).sum(dim=-1)
    opacity = weights.sum(dim=-1)
    iz_surface = (N - 1) - depth_rev
    iz_surface = torch.where(opacity > 0.5, iz_surface, torch.full_like(iz_surface, -1.0))
    iz_grid = torch.arange(N, device=sigma_old.device, dtype=sigma_old.dtype).view(1, 1, N)
    return (iz_grid > iz_surface.unsqueeze(-1)).to(sigma_old.dtype)


class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):  # to be overridden by subclass
        raise NotImplementedError()


# ----------------------------------------------------------------------------


def subset_from_nose_radius(pcd, radius_cutoff=1.1):
    nose_idx = torch.tensor([8127, 8128, 8255, 8256])
    points = pcd[:, [0, 1, 2]]

    nose_mean_point = points[nose_idx].mean(0)
    nose_mask = torch.norm(points - nose_mean_point, dim=1, p=2) < radius_cutoff
    return pcd[nose_mask]


ray_sampler_static = RaySampler()


def sample_rays(H, W):
    # if scale_anneal>0:
    #     k_iter = iterations // 1000 * 3
    #     min_scale = max(min_scale, max_scale * exp(-k_iter*scale_anneal))
    #     min_scale = min(0.9, min_scale)
    # else:
    min_scale = 0.25

    N_samples_sqrt = 60

    random_shift = True
    random_scale = True

    max_scale = 1.0

    # nn.functional.grid_sample grid value range in [-1,1]
    w, h = torch.meshgrid([torch.linspace(-1, 1, N_samples_sqrt), torch.linspace(-1, 1, N_samples_sqrt)])
    h = h.unsqueeze(2)
    w = w.unsqueeze(2)

    scale = 1
    if random_scale:
        scale = torch.Tensor(1).uniform_(min_scale, max_scale)
        h = h * scale
        w = w * scale

    if random_shift:
        max_offset = 0.25
        h_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2, (1,)).float() - 0.5) * 2
        w_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2, (1,)).float() - 0.5) * 2

        h += h_offset
        w += w_offset

    scale = scale

    return torch.cat([h, w], dim=2)


def sample_grid_flex():
    H = W = 128

    rse = sample_rays(H, W)

    # import matplotlib.pyplot as plt

    rse_i = ((rse.squeeze(0)[:, :, 1] + 1) / 2 * 128).clamp(0, 127)
    rse_i = rse_i.int().numpy()
    blank_t = torch.zeros(128, 128).numpy()
    blank_t[rse_i[:, 0], rse_i[:, ::-1]] = 1

    rse_i = ((rse.squeeze(0)[:, :, 0] + 1) / 2 * 128).clamp(0, 127)
    rse_i = rse_i.int().numpy()
    # blank_t=torch.zeros(128,128).numpy()
    blank_t[rse_i[:, 1], rse_i[:, 0]] = 1

    blank_random = torch.zeros(128, 128).numpy()

    # extra_pts=blank_random[blank_t==0][:]
    # extra_pts=torch.randperm(blank_random[blank_t==0].shape[0])[:blank_t.sum().int()*2]

    # blank_t becomes the mask

    # downsample point cloud thru this

    return blank_t


def normalise_all_dims_of_pcd(ttl):
    maxvals = ttl.max(0)[0]
    minvals = ttl.min(0)[0]
    normali = ((maxvals - ttl) / (maxvals - minvals) - 0.5) * 2
    return normali


# modify the Loss to add the pointnet
class StyleGAN2Loss_with_RLHF_pnet(Loss):
    def __init__(
        self,
        device,
        G,
        D,
        augment_pipe=None,
        r1_gamma=10,
        style_mixing_prob=0,
        pl_weight=0,
        pl_batch_shrink=2,
        pl_decay=0.01,
        pl_no_weight_grad=False,
        blur_init_sigma=0,
        blur_fade_kimg=0,
        r1_gamma_init=0,
        r1_gamma_fade_kimg=0,
        neural_rendering_resolution_initial=64,
        neural_rendering_resolution_final=None,
        neural_rendering_resolution_fade_kimg=0,
        gpc_reg_fade_kimg=1000,
        gpc_reg_prob=None,
        dual_discrimination=False,
        filter_mode="antialiased",
    ):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.augment_pipe = augment_pipe
        self.r1_gamma = r1_gamma
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_no_weight_grad = pl_no_weight_grad
        self.pl_mean = torch.zeros([], device=device)
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.r1_gamma_init = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1, 3, 3, 1], device=device)
        self.blur_raw_target = True
        self.G_ema_rlhf = None  # this ig G wtih EMA averaging
        self.G_rlhf = None  # this is G without EMA averaging
        self.rwd_affine_offset = None
        self.rwd_scale = 1.0

        self.global_step = 0
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

        self.global_embeddings_are_initialised = False

    def get_canonical_dmap_cams_for_rlhf(self):
        tdmap_cams = torch.load(
            STATIC_CONFIGS_DIR / "triple_dmap_cameras.pt",
            map_location=torch.device("cpu"),
        )
        canon_cam = tdmap_cams[1].unsqueeze(0)
        c = canon_cam
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        return dict(cam2world_matrix=cam2world_matrix, intrinsics=intrinsics, gen_c=c)

    def get_triple_dmap_cams_for_rlhf(self):
        tdmap_cams = torch.load(
            STATIC_CONFIGS_DIR / "triple_dmap_cameras.pt",
            map_location=torch.device("cpu"),
        )

        intrinsics_list = []
        gen_c_list = []
        c2w_mat_list = []

        for i in range(3):
            canon_cam = tdmap_cams[i].unsqueeze(0)
            c = canon_cam
            cam2world_matrix = c[:, :16].view(-1, 4, 4)
            intrinsics = c[:, 16:25].view(-1, 3, 3)

            intrinsics_list.append(intrinsics)

            gen_c_list.append(c)
            c2w_mat_list.append(c)

        return dict(cam2world_matrix=c2w_mat_list, intrinsics=intrinsics_list, gen_c=gen_c_list)

    def run_G(
        self,
        z,
        c,
        swapping_prob,
        neural_rendering_resolution,
        update_emas=False,
        c_gen_conditioning=None,
        truncation_psi=1,
        truncation_cutoff=None,
        drop_super_res=False,
        depth_only=False,
        current_G=None,
        noise_mode=None,
        allow_style_mixing=True,
    ):
        if c_gen_conditioning is not None:
            c_gen_conditioning = c_gen_conditioning

        elif swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(
                torch.rand((c.shape[0], 1), device=c.device) < swapping_prob,
                c_swapped,
                c,
            )
        else:
            c_gen_conditioning = torch.zeros_like(c)
        G_for_model = self.G

        if current_G is not None:
            G_for_model = current_G  # G_for_sample
        ws = G_for_model.mapping(z, c_gen_conditioning, update_emas=update_emas, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        if allow_style_mixing and self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function("style_mixing"):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(
                    torch.rand([], device=ws.device) < self.style_mixing_prob,
                    cutoff,
                    torch.full_like(cutoff, ws.shape[1]),
                )
                ws[:, cutoff:] = G_for_model.mapping(torch.randn_like(z), c, update_emas=update_emas, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)[:, cutoff:]
        synthesis_kwargs = dict(
            neural_rendering_resolution=neural_rendering_resolution,
            update_emas=update_emas,
            drop_super_res=drop_super_res,
            depth_only=depth_only,
        )
        if noise_mode is not None:
            synthesis_kwargs["noise_mode"] = noise_mode
        gen_output = G_for_model.synthesis(ws, c, **synthesis_kwargs)
        return gen_output, ws

    def return_l1_loss_sigma_ema(self, gen_z, gen_c):  # ,swapping_prob,gen_img,rays_from_new):
        # Optional sigma anchor against old_G_ema. The protected live configs use
        # this for sigma-field tuning; other reward types may leave it at zero.
        self.old_G_ema.cuda()
        self.old_G_ema.eval()

        l1_loss_sigma = torch.tensor([0.0], device=gen_z.device)
        for k in range(gen_z.shape[0]):
            gen_z[k, :].requires_grad_(True)
            gen_c[k, :].requires_grad_(True)
            sigma_samples_new = self.MUDC.sample_sigma_rays_from_z(self.G, z=gen_z[k, :].unsqueeze(0), c=gen_c[k, :].unsqueeze(0), shape_res=64)
            with torch.no_grad():
                sigma_samples_old = self.MUDC.sample_sigma_rays_from_z(self.old_G_ema, z=gen_z[k, :].unsqueeze(0), c=gen_c[k, :].unsqueeze(0), shape_res=64)

            l1_loss_sigma_k = torch.nn.functional.l1_loss(sigma_samples_new, sigma_samples_old).mean()

            l1_loss_sigma += l1_loss_sigma_k

        l1_loss_sigma = l1_loss_sigma * self.hydra_cfg.rlhf_tune_hpms.lambda_sigma_l1
        return l1_loss_sigma

    def return_front_growth_loss_sigma_ema(self, gen_z, gen_c):
        # Optional sigma front-growth regulariser. This path is coherent but left
        # off by the protected live finetune configs unless the dedicated lambda is set.
        self.old_G_ema.cuda()
        self.old_G_ema.eval()

        cfg = self.hydra_cfg.rlhf_tune_hpms
        margin = float(cfg.get("sigma_positive_flip_margin", 0.0))
        old_threshold = float(cfg.get("sigma_positive_flip_old_threshold", 0.0))
        shape_res = int(cfg.get("sigma_front_growth_shape_res", 64))
        use_geometric_mask = bool(cfg.get("sigma_use_geometric_flip_mask", False))
        use_mse_penalty = bool(cfg.get("sigma_mse_geometric_flip_penalty", False))
        flip_kernel = int(cfg.get("sigma_flip_local_mean_kernel", 1))
        flip_kernel_max = bool(cfg.get("sigma_flip_local_kernel_max", False))

        front_growth_loss = torch.tensor([0.0], device=gen_z.device)

        for k in range(gen_z.shape[0]):
            gen_z[k, :].requires_grad_(True)
            gen_c[k, :].requires_grad_(True)
            sigma_samples_new = self.MUDC.sample_sigma_rays_from_z(self.G, z=gen_z[k, :].unsqueeze(0), c=gen_c[k, :].unsqueeze(0), shape_res=shape_res)
            with torch.no_grad():
                sigma_samples_old = self.MUDC.sample_sigma_rays_from_z(self.old_G_ema, z=gen_z[k, :].unsqueeze(0), c=gen_c[k, :].unsqueeze(0), shape_res=shape_res)

            if use_geometric_mask:
                mask = _compute_geometric_presurface_mask(sigma_samples_old).to(sigma_samples_new.dtype)
                raw_delta = sigma_samples_new - sigma_samples_old - margin
                if flip_kernel > 1:
                    pad = flip_kernel // 2
                    pooled_in = raw_delta.unsqueeze(0).unsqueeze(0)
                    if flip_kernel_max:
                        pooled = torch.nn.functional.max_pool3d(pooled_in, kernel_size=flip_kernel, stride=1, padding=pad)
                    else:
                        pooled = torch.nn.functional.avg_pool3d(pooled_in, kernel_size=flip_kernel, stride=1, padding=pad)
                    delta_pos = torch.nn.functional.relu(pooled.squeeze(0).squeeze(0))
                else:
                    delta_pos = torch.nn.functional.relu(raw_delta)
                positive_flip = (delta_pos**2 if use_mse_penalty else delta_pos) * mask
            else:
                mask = (sigma_samples_old <= old_threshold).to(sigma_samples_new.dtype)
                positive_flip = torch.nn.functional.relu(sigma_samples_new - margin) * mask
            denom = torch.clamp(mask.sum(), min=1e-8)
            front_growth_loss_k = positive_flip.sum() / denom
            front_growth_loss += front_growth_loss_k

        front_growth_loss = front_growth_loss * self.hydra_cfg.rlhf_tune_hpms.lambda_sigma_front_growth
        return front_growth_loss

    def get_depth_reg_camera_template(self):
        camera_mode = self.hydra_cfg.rlhf_tune_hpms.get("dmap_reg_camera_mode", "triple")
        if camera_mode == "triple":
            return self.get_triple_dmap_cameras()
        if camera_mode in ["canonical", "single"]:
            return self.get_single_dmap_camera()
        raise ValueError(f"Unsupported dmap_reg_camera_mode={camera_mode}")

    def render_depth_stack_from_model(self, G_model, z_batch, gen_c_template, neural_rendering_resolution, truncation_psi, truncation_cutoff, noise_mode):
        bform = self.format_batch(gen_c_template=gen_c_template, z=z_batch)
        depth_img, _ = self.run_G(
            bform["gen_z"],
            bform["gen_c"],
            swapping_prob=0.0,
            neural_rendering_resolution=neural_rendering_resolution,
            update_emas=False,
            c_gen_conditioning=bform["gen_c"],
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            drop_super_res=True,
            depth_only=True,
            current_G=G_model,
            noise_mode=noise_mode,
            allow_style_mixing=False,
        )
        return depth_img["image_depth"]

    def render_consistency_stack_from_model(self, G_model, z_batch, gen_c_template, neural_rendering_resolution, truncation_psi, truncation_cutoff, noise_mode, with_grad):
        bform = self.format_batch(gen_c_template=gen_c_template, z=z_batch)

        def _render():
            gen_output, _ = self.run_G(
                bform["gen_z"],
                bform["gen_c"],
                swapping_prob=0.0,
                neural_rendering_resolution=neural_rendering_resolution,
                update_emas=False,
                c_gen_conditioning=bform["gen_c"],
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
                drop_super_res=True,
                depth_only=False,
                current_G=G_model,
                noise_mode=noise_mode,
                allow_style_mixing=False,
            )
            return gen_output

        if with_grad:
            return _render()
        with torch.no_grad():
            return _render()

    def get_depth_reg_center_weight_mask(self, image_depth):
        _, _, height, width = image_depth.shape
        yy = torch.linspace(-1.0, 1.0, height, device=image_depth.device, dtype=image_depth.dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=image_depth.device, dtype=image_depth.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        sigma_x = float(self.hydra_cfg.rlhf_tune_hpms.get("dmap_forward_center_sigma_x", 0.22))
        sigma_y = float(self.hydra_cfg.rlhf_tune_hpms.get("dmap_forward_center_sigma_y", 0.26))
        center_x = float(self.hydra_cfg.rlhf_tune_hpms.get("dmap_forward_center_x", 0.0))
        center_y = float(self.hydra_cfg.rlhf_tune_hpms.get("dmap_forward_center_y", 0.0))
        gauss = torch.exp(-0.5 * (((grid_x - center_x) / sigma_x) ** 2 + ((grid_y - center_y) / sigma_y) ** 2))
        gauss = gauss / gauss.mean().clamp_min(1e-6)
        return gauss.unsqueeze(0).unsqueeze(0)

    def return_depth_reg_losses_ema(self, gen_z_base, gen_img):
        # Optional depth/consistency regularisation against old_G_ema. The
        # protected configs leave this branch off, but the path is live when
        # enabled (including the restored dmap-MSE / LPIPS terms).
        self.old_G_ema.cuda()
        self.old_G_ema.eval()

        noise_mode = self.hydra_cfg.rlhf_tune_hpms.get("dmap_reg_noise_mode", self.hydra_cfg.rlhf_tune_hpms.G_sample_noise_mode)
        truncation_psi = self.hydra_cfg.rlhf_tune_hpms.get("dmap_reg_truncation_psi", self.hydra_cfg.rlhf_tune_hpms.G_sample_truncation_psi)
        truncation_cutoff = self.hydra_cfg.rlhf_tune_hpms.get("dmap_reg_truncation_cutoff", self.hydra_cfg.rlhf_tune_hpms.G_sample_truncation_cutoff)
        neural_rendering_resolution = int(self.hydra_cfg.rlhf_tune_hpms.get("dmap_reg_resolution", gen_img["image_depth"].shape[-1]))
        gen_c_template = self.get_depth_reg_camera_template()

        gen_outputs = self.render_consistency_stack_from_model(
            self.G,
            z_batch=gen_z_base,
            gen_c_template=gen_c_template,
            neural_rendering_resolution=neural_rendering_resolution,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            noise_mode=noise_mode,
            with_grad=True,
        )
        gen_img_ema_outputs = self.render_consistency_stack_from_model(
            self.old_G_ema,
            z_batch=gen_z_base,
            gen_c_template=gen_c_template,
            neural_rendering_resolution=neural_rendering_resolution,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            noise_mode=noise_mode,
            with_grad=False,
        )

        gen_img_depth = gen_outputs["image_depth"]
        gen_img_ema_depth = gen_img_ema_outputs["image_depth"]
        gen_img_raw = gen_outputs["image_raw"]
        gen_img_ema_raw = gen_img_ema_outputs["image_raw"]

        l1_loss_dmap = torch.nn.functional.l1_loss(gen_img_depth, gen_img_ema_depth).mean() * self.hydra_cfg.rlhf_tune_hpms.lambda_dmap_l1
        mse_loss_dmap = torch.nn.functional.mse_loss(gen_img_depth, gen_img_ema_depth).mean() * self.hydra_cfg.rlhf_tune_hpms.lambda_dmap_mse
        center_mask = self.get_depth_reg_center_weight_mask(gen_img_depth)
        depth_threshold = float(self.hydra_cfg.rlhf_tune_hpms.get("dmap_forward_depth_threshold", 2.6))
        forward_margin = float(self.hydra_cfg.rlhf_tune_hpms.get("dmap_forward_margin", 0.01))
        forward_delta = torch.relu((gen_img_ema_depth - gen_img_depth) - forward_margin)
        face_mask = (gen_img_ema_depth <= depth_threshold).to(gen_img_depth.dtype)
        weighted_mask = center_mask * face_mask
        if weighted_mask.sum() > 0:
            forward_mse_dmap = ((forward_delta**2) * weighted_mask).sum() / weighted_mask.sum().clamp_min(1e-6)
        else:
            forward_mse_dmap = torch.zeros([], device=gen_img_depth.device, dtype=gen_img_depth.dtype)
        forward_mse_dmap = forward_mse_dmap * float(self.hydra_cfg.rlhf_tune_hpms.get("lambda_dmap_forward_mse", 0.0))
        lambda_lpips = float(self.hydra_cfg.rlhf_tune_hpms.get("lambda_lpips", 0.0))
        lpips_loss = torch.zeros([], device=gen_img_depth.device, dtype=gen_img_depth.dtype)
        if lambda_lpips != 0.0:
            lpips_loss = self.LPIPS.forward(gen_img_raw, gen_img_ema_raw).mean() * lambda_lpips

        total_depth_reg = l1_loss_dmap + mse_loss_dmap + forward_mse_dmap + lpips_loss

        return dict(
            total=total_depth_reg,
            l1=l1_loss_dmap,
            mse=mse_loss_dmap,
            forward_mse=forward_mse_dmap,
            lpips=lpips_loss,
        )

    def run_old_G_ema(
        self,
        z,
        c,
        swapping_prob,
        neural_rendering_resolution,
        update_emas=False,
        c_gen_conditioning=None,
        truncation_psi=1,
        truncation_cutoff=None,
        noise_mode=None,
    ):
        with torch.no_grad():
            if c_gen_conditioning is not None:
                c_gen_conditioning = c_gen_conditioning

            elif swapping_prob is not None:
                c_swapped = torch.roll(c.clone(), 1, 0)
                c_gen_conditioning = torch.where(
                    torch.rand((c.shape[0], 1), device=c.device) < swapping_prob,
                    c_swapped,
                    c,
                )
            else:
                c_gen_conditioning = torch.zeros_like(c)
            ws = self.old_G_ema.mapping(z, c_gen_conditioning, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
            if self.style_mixing_prob > 0:
                # with torch.autograd.profiler.record_function("style_mixing"):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(
                    torch.rand([], device=ws.device) < self.style_mixing_prob,
                    cutoff,
                    torch.full_like(cutoff, ws.shape[1]),
                )
                ws[:, cutoff:] = self.old_G_ema.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            synthesis_kwargs = dict(
                neural_rendering_resolution=neural_rendering_resolution,
                update_emas=update_emas,
            )
            if noise_mode is not None:
                synthesis_kwargs["noise_mode"] = noise_mode
            gen_output = self.old_G_ema.synthesis(ws, c, **synthesis_kwargs)
        return gen_output, ws

    def run_old_G(
        self,
        z,
        c,
        swapping_prob,
        neural_rendering_resolution,
        update_emas=False,
        c_gen_conditioning=None,
    ):
        with torch.no_grad():
            if c_gen_conditioning is not None:
                c_gen_conditioning = c_gen_conditioning

            elif swapping_prob is not None:
                c_swapped = torch.roll(c.clone(), 1, 0)
                c_gen_conditioning = torch.where(
                    torch.rand((c.shape[0], 1), device=c.device) < swapping_prob,
                    c_swapped,
                    c,
                )
            else:
                c_gen_conditioning = torch.zeros_like(c)
            ws = self.old_G.mapping(z, c_gen_conditioning, update_emas=update_emas)
            if self.style_mixing_prob > 0:
                # with torch.autograd.profiler.record_function("style_mixing"):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(
                    torch.rand([], device=ws.device) < self.style_mixing_prob,
                    cutoff,
                    torch.full_like(cutoff, ws.shape[1]),
                )
                ws[:, cutoff:] = self.old_G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            gen_output = self.old_G.synthesis(
                ws,
                c,
                neural_rendering_resolution=neural_rendering_resolution,
                update_emas=update_emas,
            )
        return gen_output, ws

    def setup_coordinates_for_sigma_field_512_sample(self):
        coordinates, shape, _ = self.MUDC.get_samples_coordinates_from_pads_vals_dict(pads_vals=self.pads_vals, G=self.G, shape_res=self.pads_vals.shape_res)  # make it as 256, then do trilinear upsampling once have extracted sigma, to fit it all i nmemory
        self.coordinates_for_sigma_field_512_sample = coordinates
        self.shape_of_sigma_coords = shape
        self.trilinear_upsampler_256_to_512 = torch.nn.Identity()
        if self._reward_dtype() == "sigma_field_512" and self.pads_vals.shape_res == 256:
            self.trilinear_upsampler_256_to_512 = torch.nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

    def setup_sig_field_params(self):
        if not hasattr(self, "pads_vals"):
            pads_vals_fn = getattr(self.hydra_cfg.rlhf_tune_hpms, "rwd_sigma_pads_config_path", DEFAULT_RWD_SIGMA_PADS_CONFIG)
            self.pads_vals = omegaconf.OmegaConf.load(pads_vals_fn)

        if not hasattr(self, "coordinates_for_sigma_field_512_sample"):
            self.setup_coordinates_for_sigma_field_512_sample()

        import hydra

        # Tune-time transform fed to the reward model. Prefer a hand-written
        # override in the eg3d tune config; otherwise inherit the reward model's
        # recorded `tune` slot from its saved run_config (single source of truth
        # -> the tune transform can't silently desync from what the reward model
        # was trained on; see reward_loading.load_tune_augmentation_from_cfg).
        tune_aug = omegaconf.OmegaConf.select(self.hydra_cfg, "rlhf_tune_hpms.augmentations")
        if tune_aug is None:
            from core_modules.utils.finetuning_utils import load_tune_augmentation_from_cfg

            rwd_model_id = omegaconf.OmegaConf.select(self.hydra_cfg, "rwd_model_id")
            tune_aug = load_tune_augmentation_from_cfg(rwd_model_id)

        self.augmentations = hydra.utils.instantiate(tune_aug) if tune_aug is not None else torch.nn.Identity()

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function("blur"):
                f = torch.arange(-blur_size, blur_size + 1, device=img["image"].device).div(blur_sigma).square().neg().exp2()
                img["image"] = upfirdn2d.filter2d(img["image"], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(
                torch.cat(
                    [
                        img["image"],
                        torch.nn.functional.interpolate(
                            img["image_raw"],
                            size=img["image"].shape[2:],
                            mode="bilinear",
                            antialias=True,
                        ),
                    ],
                    dim=1,
                )
            )
            img["image"] = augmented_pair[:, : img["image"].shape[1]]
            img["image_raw"] = torch.nn.functional.interpolate(
                augmented_pair[:, img["image"].shape[1] :],
                size=img["image_raw"].shape[2:],
                mode="bilinear",
                antialias=True,
            )

        logits = self.D(img, c, update_emas=update_emas)
        return logits

    # putting this one to simplify code AM 06_07_2023
    def run_d_on_g(self, gen_img, gen_c, blur_sigma=0):
        gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
        loss_Gmain = torch.nn.functional.softplus(-gen_logits)
        return loss_Gmain

    def get_triple_dmap_cameras(self):
        gen_c_template = self.get_triple_dmap_cams_for_rlhf()["gen_c"]
        gen_c_template = torch.vstack(gen_c_template)

        return gen_c_template

    def get_single_dmap_camera(self):
        gen_c_template = self.get_canonical_dmap_cams_for_rlhf()["gen_c"]
        return gen_c_template

    def format_batch(self, gen_c_template, z):
        bsize = z.shape[0]
        ncams = gen_c_template.shape[0]
        gen_c = gen_c_template[None, ...].expand(bsize, -1, 25).to(z.device)
        z = z.unsqueeze(1).expand(bsize, gen_c.shape[1], 512)
        gen_z = z.reshape(bsize * ncams, 512)
        gen_c = gen_c.reshape(bsize * ncams, 25)

        return dict(bsize=bsize, ncams=ncams, gen_c=gen_c, gen_z=gen_z)

    def _reward_dtype(self):
        return self.hydra_cfg.rwd_model_data_type

    def _is_triple_dmap_reward_dtype(self, dtype=None):
        dtype = self._reward_dtype() if dtype is None else dtype
        return dtype == "triple_dmap"

    def _is_sigma_rays_reward_dtype(self, dtype=None):
        dtype = self._reward_dtype() if dtype is None else dtype
        return dtype == "sigma_rays"

    def _is_sigma_field_reward_dtype(self, dtype=None):
        dtype = self._reward_dtype() if dtype is None else dtype
        return dtype in SIGMA_FIELD_REWARD_DTYPES

    def _pairwise_reward_ltype(self):
        ltype = self.hydra_cfg.ltype
        if ltype == "pairs":
            # Backward compatibility for archived configs; the maintained name is
            # now pairs_refset.
            return "pairs_refset"
        return ltype

    def _pairwise_reward_tune_type(self):
        tune_type = self.hydra_cfg.rlhf_tune_hpms.tune_type
        if tune_type == "pairs":
            # Backward compatibility for archived configs; the maintained name is
            # now pairs_refset.
            return "pairs_refset"
        return tune_type

    def _uses_scalar_reward_head(self):
        return self._pairwise_reward_ltype() != "pairs_refset" and (self._is_sigma_rays_reward_dtype() or self._is_sigma_field_reward_dtype() or self._reward_dtype() == "sigma_field_64")

    def _reward_debug_enabled(self):
        return bool(self.hydra_cfg.rlhf_tune_hpms.get("debug_reward_loss", False))

    def _reward_debug(self, *args):
        if self._reward_debug_enabled():
            print(*args)

    def _wandb_enabled(self):
        return bool(getattr(self.hydra_cfg, "using_wandb", False)) and wandb is not None

    def _report_reward_scalar(self, name, value, logdict=None, log_key=None):
        # Report a reward-loss scalar to training_stats and TensorBoard (and,
        # when a logdict is supplied, the W&B log dict) in one place. Mirrors
        # the three-sink pattern repeated throughout the G_depth reward branch.
        mean_val = value.mean()
        training_stats.report(name, mean_val)
        self.stats_tfevents.add_scalar(name, mean_val, global_step=self.global_step)
        if logdict is not None:
            logdict[log_key if log_key is not None else name] = mean_val
        logger.info("%s: %s", name, mean_val)
        return mean_val

    def _get_reward_camera_template(self):
        if self._is_triple_dmap_reward_dtype():
            return self.get_triple_dmap_cameras()
        return self.get_single_dmap_camera()

    def _format_sigma_field_reward_volume(self, sigmas, apply_zero_grad_threshold=False):
        if apply_zero_grad_threshold:
            zero_grad_threshold = self.hydra_cfg.rlhf_tune_hpms.get("zero_grad_below_sigma_threshold", None)
            if zero_grad_threshold is not None:
                zero_grad_mask = (sigmas < float(zero_grad_threshold)).detach()
                sigmas = torch.where(zero_grad_mask, sigmas.detach(), sigmas)

        sigmas = self.augmentations(sigmas)
        if self._is_sigma_field_reward_dtype():
            sigmas = sigmas.permute(2, 1, 0)

        sigmas = sigmas.unsqueeze(0).unsqueeze(0)
        return self.trilinear_upsampler_256_to_512(sigmas)

    def _build_sigma_rays_reward_input_tensor(self, G_model, gen_z, gen_c, truncation_psi, truncation_cutoff, update_emas):
        batch_rays = []
        for k in range(gen_z.shape[0]):
            rays = self.DST.sample_sigma_rays_from_z_as_tensor(
                G_model,
                z=gen_z[k, :].unsqueeze(0),
                c=gen_c[k, :].unsqueeze(0),
                shape_res=128,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
                update_emas=update_emas,
            )
            batch_rays.append(rays.unsqueeze(0))

        return torch.cat(batch_rays, 0) * self.hydra_cfg.rlhf_tune_hpms.scale_sigma_vals

    def _build_sigma_field_reward_inputs(
        self,
        G_model,
        gen_z,
        gen_c,
        truncation_psi,
        truncation_cutoff,
        noise_mode,
        with_grad,
        update_emas,
        apply_zero_grad_threshold=False,
    ):
        self.setup_sig_field_params()
        G_model.requires_grad_(with_grad)

        rwd_model_input = []
        global_vecs = []
        coordinates = self.coordinates_for_sigma_field_512_sample

        for k in range(gen_z.shape[0]):
            self._reward_debug(f"building sigma-field reward input {k + 1}/{gen_z.shape[0]}")
            if with_grad:
                gz = gen_z[k, :].requires_grad_(True)
                gc = gen_c[k, :].requires_grad_(True)
            else:
                gz = gen_z[k, :].detach().clone()
                gc = gen_c[k, :].detach().clone()

            sigmas = self.MUDC.mesh_subset_of_points_from_samples_from_z_with_grad(
                G_model,
                z=gz.unsqueeze(0),
                conditioning_params=gc.unsqueeze(0) * self.hydra_cfg.rlhf_tune_hpms.gen_c_mult_rwd,
                samples=coordinates,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
                noise_mode=noise_mode,
                update_emas=update_emas,
            )
            sigmas = sigmas.view(self.shape_of_sigma_coords[1:4])
            sigmas_us = self._format_sigma_field_reward_volume(sigmas, apply_zero_grad_threshold=apply_zero_grad_threshold)

            rwd_model_input.append(sigmas_us)
            global_vecs.append(self.reward_model.forward_to_global_feature_vec(sigmas_us))

        return rwd_model_input, global_vecs

    def _build_volumetric_reward_inputs(
        self,
        G_model,
        gen_z,
        gen_c,
        truncation_psi,
        truncation_cutoff,
        noise_mode,
        with_grad,
        update_emas,
        apply_zero_grad_threshold=False,
    ):
        if self._is_sigma_rays_reward_dtype():
            return (
                self._build_sigma_rays_reward_input_tensor(
                    G_model,
                    gen_z,
                    gen_c,
                    truncation_psi=truncation_psi,
                    truncation_cutoff=truncation_cutoff,
                    update_emas=update_emas,
                ),
                None,
                False,
            )

        if self._is_sigma_field_reward_dtype():
            rwd_model_input, global_vecs = self._build_sigma_field_reward_inputs(
                G_model,
                gen_z,
                gen_c,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
                noise_mode=noise_mode,
                with_grad=with_grad,
                update_emas=update_emas,
                apply_zero_grad_threshold=apply_zero_grad_threshold,
            )
            return torch.cat(rwd_model_input, 0), torch.cat(global_vecs, 0), True

        raise ValueError(f"Unsupported volumetric reward dtype={self._reward_dtype()}")

    def _sample_reference_feature_batch(self, reference_features, n_needed):
        n_available = reference_features.shape[0]
        assert n_available > 0, "reward reference features must be initialised before pairwise scoring"
        repeats = max(1, (n_needed + n_available - 1) // n_available)
        list_of_perms = torch.hstack([torch.randperm(n_available) for _ in range(repeats)])
        return reference_features[list_of_perms[:n_needed]].cuda()

    def _forward_reward_global_vectors_per_item(self, rwd_model_input):
        return torch.cat([self.reward_model.forward_to_global_feature_vec(r.unsqueeze(0)) for r in rwd_model_input], dim=0)

    def _combine_bidirectional_pair_logits(self, lhs_batch, rhs_batch, batch_size, n_comparisons):
        pwin_logit = self.reward_model.forward_from_cat_global_vectors(lhs_batch, rhs_batch)
        pwin_logit = pwin_logit.reshape(batch_size, n_comparisons, -1)
        pwin_logit_rev = self.reward_model.forward_from_cat_global_vectors(rhs_batch, lhs_batch)
        pwin_logit_rev = pwin_logit_rev.flip(1).reshape(batch_size, n_comparisons, -1)
        return (pwin_logit + pwin_logit_rev) / 2

    def _compute_pairs_refset_reward_score(self, rwd_model_input, global_vectors, n_comparisons, has_calc_global):
        if self.reward_model.external is not None:
            rwd_model_input = self.reward_model.external(rwd_model_input)

        if self._is_sigma_rays_reward_dtype() and str(type(self.reward_model)).find("Conv3DNetworkEnsemble") == -1:
            global_vectors = self.reward_model.forward_to_global_feature_vec(rwd_model_input)
            gv = global_vectors.unsqueeze(1).expand(-1, n_comparisons, -1, -1)
            gvb = gv.reshape(n_comparisons * global_vectors.shape[0], 128, 128)
            reference_features = self._sample_reference_feature_batch(self.returned_global_features_start, gvb.shape[0])
            pair_logits = self._combine_bidirectional_pair_logits(gvb, reference_features, global_vectors.shape[0], n_comparisons)
            pwin = torch.nn.functional.softmax(pair_logits, -1)
            posneg = pwin - 0.5
            return posneg[:, :, 0].mean(1), global_vectors

        if not has_calc_global:
            global_vectors = self._forward_reward_global_vectors_per_item(rwd_model_input)

        gv = global_vectors.unsqueeze(1).expand(-1, n_comparisons, -1)
        gvb = gv.reshape(n_comparisons * global_vectors.shape[0], -1)
        reference_features = self._sample_reference_feature_batch(self.returned_global_features_start, gvb.shape[0])
        pair_logits = self._combine_bidirectional_pair_logits(gvb, reference_features, global_vectors.shape[0], n_comparisons)
        log_s = torch.nn.LogSoftmax(dim=-1)(pair_logits)
        plose = torch.exp(log_s[:, :, 1])
        posneg = plose - 0.5
        return -posneg.mean(1), global_vectors

    def _compute_pointwise_reward_score(self, rwd_model_input):
        if self.hydra_cfg.normalise_scalar_rwd:
            # Ziegler et al. (2019) standardisation: (r - mu_hat) / sigma_hat, with
            # mu_hat = rwd_affine_offset and sigma_hat captured via rwd_scale =
            # 1/sigma_hat from the initial-generator baseline (computed at init in
            # training_loop). This is the SAME affine now applied on the scalar head
            # (see _compute_reward_score_raw), so pcd / dmap / sigma share one scale.
            # The old fixed `rescale_to` multiply is DROPPED -- the proper scale is
            # now 1/sigma_hat, so rwd_loss_scale stays 1.0 (rescale_to is now dead).
            if self.rwd_affine_offset is None:
                self.rwd_affine_offset = 0.0
            self.hydra_cfg.rwd_loss_scale = 1.0
        else:
            self.rwd_affine_offset = 0.0
            self.rwd_scale = 1.0
            self.hydra_cfg.rwd_loss_scale = 1.0

        if self.reward_model.external is not None:
            rwd_model_input = self.reward_model.external(rwd_model_input)

        return (self.reward_model.forward(rwd_model_input) - self.rwd_affine_offset) * self.rwd_scale * self.hydra_cfg.rwd_loss_scale

    def _compute_reward_score_raw(self, rwd_model_input, global_vectors, n_comparisons, has_calc_global, gen_z, gen_c):
        if self._uses_scalar_reward_head():
            if global_vectors is None:
                global_vectors = self._forward_reward_global_vectors_per_item(rwd_model_input)
            scalar_rwd_score = self.reward_model.scalar_rwd_head(global_vectors).mean()
            if not self.hydra_cfg.normalise_scalar_rwd:
                # RAW reward score (no standardisation). This is the REPORTED config
                # for the sigma-field model: the reward term is
                #   L_reward = -lambda * clip(r_phi(x), -c, c)
                # on the raw reward-model score r_phi (clip = rwd_clamp_min/max,
                # lambda = lambda_rwd_model). The clip bounds each sample's reward and
                # zeroes its gradient once the score passes the cap. The baseline
                # mu_hat/sigma_hat may still be computed for the histogram, but is NOT
                # applied here. (Run 01446 used this raw form.)
                return scalar_rwd_score, global_vectors
            # Ziegler et al. (2019) standardisation (ablation): map the reward onto a
            # common (mean 0, var 1) scale via the initial-generator baseline
            # (rwd_affine_offset = mu_hat, rwd_scale = 1/sigma_hat; see training_loop).
            offset = self.rwd_affine_offset if self.rwd_affine_offset is not None else 0.0
            return (scalar_rwd_score - offset) * self.rwd_scale, global_vectors
        pair_ltype = self._pairwise_reward_ltype()
        if pair_ltype == "pairs_refset":
            # Pairwise reward against sampled reference-set embeddings. This is the
            # maintained pairwise reward path; older pairs_old / pairs_patches
            # branches were experimental and have been removed from the active
            # runtime.
            return self._compute_pairs_refset_reward_score(rwd_model_input, global_vectors, n_comparisons, has_calc_global)
        if pair_ltype in {"pairs_old", "pairs_patches"}:
            raise ValueError(f"Removed legacy pairwise reward backend ltype={self.hydra_cfg.ltype}. Use ltype=pairs_refset for the maintained pairwise reference-set path.")

        return self._compute_pointwise_reward_score(rwd_model_input), global_vectors

    def _compute_reward_score(self, rwd_score_raw):
        tune_type = self._pairwise_reward_tune_type()
        lambda_rwd_model = self.hydra_cfg.rlhf_tune_hpms.lambda_rwd_model

        if tune_type == "pairs_refset":
            # Pairwise reward path still uses the same bounded clamped objective as
            # the historical pairs runs; only the upstream reward computation differs.
            return (-torch.clamp(rwd_score_raw, max=self.hydra_cfg.rwd_clamp_max, min=self.hydra_cfg.rwd_clamp_min) + 1e-6) * lambda_rwd_model

        if tune_type == "clamped":
            # clamped (reported config): -lambda * clip(r_phi, rwd_clamp_min,
            # rwd_clamp_max). Bounds each sample's reward and zeroes its gradient past
            # the cap. On the reported raw reward, scores sit near the cap so this
            # binds routinely (under the Ziegler ablation above it seldom binds).
            return (-torch.clamp(rwd_score_raw, max=self.hydra_cfg.rwd_clamp_max, min=self.hydra_cfg.rwd_clamp_min) + 1e-6) * lambda_rwd_model

        if tune_type == "clamped_iqr":
            # Same clamped loss shape as tune_type=clamped, but the clamp bounds are
            # taken from the initial reward distribution's retained min/max after IQR
            # filtering instead of the fixed rwd_clamp_min/max config values.
            return (-torch.clamp(rwd_score_raw, max=self.max_r, min=self.min_r) + 1e-6) * lambda_rwd_model

        if tune_type == "neg_softplus":
            neg_softplus_scale = getattr(self.hydra_cfg.rlhf_tune_hpms, "neg_softplus_scale", None)
            if neg_softplus_scale is None:
                # Backward compatibility for archived configs that still use the
                # old scalar_sp name.
                neg_softplus_scale = self.hydra_cfg.rlhf_tune_hpms.scalar_sp
            assert neg_softplus_scale < 0.0, "neg_softplus requires a negative scale"
            # Smooth reward-maximization surrogate:
            # - high positive rewards -> loss approaches 0
            # - rewards near 0 -> smooth curved transition
            # - negative rewards -> penalty grows roughly linearly
            #
            # Unlike tune_type=clamped, this branch has no hard cap where the
            # gradient becomes exactly zero; it only decays smoothly for large
            # positive rewards.
            return torch.nn.functional.softplus(neg_softplus_scale * rwd_score_raw) * lambda_rwd_model

        if tune_type in {"PPO", "median", "pairs_old", "pairs_patches"}:
            raise ValueError(f"Removed legacy RLHF tune_type={self.hydra_cfg.rlhf_tune_hpms.tune_type}. Supported tune_type values are clamped, clamped_iqr, neg_softplus, and pairs_refset.")

        raise ValueError(f"Unsupported RLHF tune_type={tune_type}")

    def _render_reward_depth_images(self, gen_z, gen_c, swapping_prob):
        gd = []
        for k in range(gen_z.shape[0]):
            gen_img, _gen_ws = self.run_G(gen_z[k].unsqueeze(0), gen_c[0].unsqueeze(0), swapping_prob=swapping_prob, neural_rendering_resolution=64, drop_super_res=True, update_emas=False, depth_only=True)
            gde = gen_img["image_depth"]
            gde = torch.nn.functional.interpolate(gde, size=(128, 128), mode="bilinear", align_corners=True)
            gd.append(gde.clone())
            del gen_img
            del _gen_ws
        return dict(image_depth=torch.cat(gd, 0))

    def _get_canonical_reward_depth(self, gen_img, ncams):
        if ncams == 3:
            nrs_im = gen_img["image_depth"].shape[-1]
            gen_depth = gen_img["image_depth"].view(1, ncams, nrs_im, nrs_im)
            return gen_depth[:, 1, :, :].unsqueeze(1)
        if ncams == 1:
            return gen_img["image_depth"]
        raise ValueError(f"Unsupported reward camera count ncams={ncams}")

    def _compute_nose_loss(self, gen_depth_canonical):
        nose_loss = torch.tensor(0.0).cuda()
        if self.hydra_cfg.rlhf_tune_hpms.lambda_nose_hard == 0.0:
            return nose_loss

        nose_violations = gen_depth_canonical[gen_depth_canonical < self.hydra_cfg.rlhf_tune_hpms.hard_nose_depth]
        stoppage_loss = 2.7 - self.hydra_cfg.rlhf_tune_hpms.hard_nose_depth
        if nose_violations.shape[0] > 0:
            nose_violations = (2.7 - nose_violations - stoppage_loss) * self.hydra_cfg.rlhf_tune_hpms.lambda_nose_hard
            nose_loss = nose_violations.mean()
        return nose_loss

    def get_rwd_scores_eval_from_loss_G(self, seeds, G_ema=None):
        G_for_sample = self.G
        if G_ema is not None:
            G_for_sample = G_ema

        truncation_psi = 1.0  # self.hydra_cfg.hpms_first_dmaps.truncation_psi
        truncation_cutoff = 14  # self.hydra_cfg.hpms_first_dmaps.truncation_cutoff

        dtype = self._reward_dtype()

        zvals = get_list_of_z(seeds)  # seeds should be random seeds

        scores = []
        nrs = 128
        with torch.no_grad():
            rwd_scores = []
            gen_z = torch.tensor([1], device=torch.device("cuda:0"))

            gen_c_template = self._get_reward_camera_template()

            # TAKE A DATALOADER OF IT
            zloader = torch.utils.data.DataLoader(zvals, batch_size=self.hydra_cfg.plot.batchsize, shuffle=False, drop_last=False)

            for z in tqdm(zloader):
                bform = self.format_batch(gen_c_template=gen_c_template, z=z)

                bsize = bform["bsize"]
                ncams = bform["ncams"]

                gen_z = bform["gen_z"]
                gen_c = bform["gen_c"]

                if self._is_sigma_rays_reward_dtype(dtype) or self._is_sigma_field_reward_dtype(dtype):
                    rwd_model_input, _global_vecs, _has_calc_global = self._build_volumetric_reward_inputs(
                        G_for_sample,
                        gen_z,
                        gen_c,
                        truncation_psi=truncation_psi,
                        truncation_cutoff=truncation_cutoff,
                        noise_mode="const",
                        with_grad=False,
                        update_emas=False,
                    )
                    scores.append(self.reward_model.forward(rwd_model_input))

                else:
                    gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=0.0, neural_rendering_resolution=nrs, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, current_G=G_for_sample)  # change gen_c to gc_rep
                    rwd_model_input = self.DST.format_gen_out_for_rwd_input(gen_img=gen_img, bsize=bsize, ncams=ncams, gen_c=gen_c)

                    if self.reward_model.external is not None:
                        rwd_model_input = self.reward_model.external(rwd_model_input)

                    # embeddings=self.reward_model.forward_to_global_feature_vec(rwd_model_input)
                    # self.returned_global_features.append(embeddings)
                    score = self.reward_model.forward(rwd_model_input)
                    scores.append(score)

        all_rwd_scores = torch.vstack(scores).flatten()

        return all_rwd_scores  # rwd_scores

    def init_global_embeddings_at_start(self):
        assert self.global_embeddings_are_initialised == False
        nrs = 128
        with torch.no_grad():
            rwd_scores = []
            gen_z = torch.tensor([1], device=torch.device("cuda:0"))

            gen_c_template = self._get_reward_camera_template()

            # TAKE A DATALOADER OF IT
            zloader = torch.utils.data.DataLoader(self.zvals, batch_size=self.hydra_cfg.plot.batchsize, shuffle=False, drop_last=False)

            # hpms_first_dmaps:
            #     #truncation_psi: 0.7
            #     truncation_psi: 0.25
            #     truncation_cutoff: 14

            truncation_psi = self.hydra_cfg.hpms_first_dmaps.truncation_psi
            truncation_cutoff = self.hydra_cfg.hpms_first_dmaps.truncation_cutoff

            for z in tqdm(zloader):
                bform = self.format_batch(gen_c_template=gen_c_template, z=z)

                bsize = bform["bsize"]
                ncams = bform["ncams"]
                gen_z = bform["gen_z"]
                gen_c = bform["gen_c"]

                if self._is_sigma_rays_reward_dtype() or self._is_sigma_field_reward_dtype():
                    _rwd_model_input, embeddings, _has_calc_global = self._build_volumetric_reward_inputs(
                        self.G,
                        gen_z,
                        gen_c,
                        truncation_psi=truncation_psi,
                        truncation_cutoff=truncation_cutoff,
                        noise_mode="const",
                        with_grad=False,
                        update_emas=False,
                    )
                    if embeddings is None:
                        embeddings = self.reward_model.forward_to_global_feature_vec(_rwd_model_input)

                    gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=0.0, neural_rendering_resolution=nrs, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)  # change gen_c to gc_rep
                    gen_depth_canonical = gen_img["image_depth"]

                else:
                    gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=0.0, neural_rendering_resolution=nrs, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)  # change gen_c to gc_rep
                    rwd_model_input = self.DST.format_gen_out_for_rwd_input(gen_img=gen_img, bsize=bsize, ncams=ncams, gen_c=gen_c)

                    if self.reward_model.external is not None:
                        rwd_model_input = self.reward_model.external(rwd_model_input)

                    embeddings = self.reward_model.forward_to_global_feature_vec(rwd_model_input)
                    # self.returned_global_features.append(embeddings)

                    if ncams == 3:
                        # take middle dmap
                        # img_depth_middle
                        nrs_im = gen_img["image_depth"].shape[-1]
                        gen_depth = gen_img["image_depth"].reshape(bsize, ncams, nrs_im, nrs_im)
                        gen_depth_canonical = gen_depth[:, 1, :, :].unsqueeze(1)
                    elif ncams == 1:
                        gen_depth_canonical = gen_img["image_depth"]

                    else:
                        assert False, "error ncams not ==1 or 3"

                self.returned_global_features_start.append(embeddings.cpu())

                self.returned_dmaps_start.append(gen_depth_canonical.cpu())

        return self

    def extract_sig_field_to_rwd_model(self, G, gen_z, gen_c, with_grad=True):
        return self._build_sigma_field_reward_inputs(
            G,
            gen_z,
            gen_c,
            truncation_psi=self.hydra_cfg.rlhf_tune_hpms.G_sample_truncation_psi,
            truncation_cutoff=self.hydra_cfg.rlhf_tune_hpms.G_sample_truncation_cutoff,
            noise_mode=self.hydra_cfg.rlhf_tune_hpms.G_sample_noise_mode,
            with_grad=with_grad,
            update_emas=True,
        )

    def initialise_global_embeddings_eval_from_loss_G(self, seeds, return_global_feature_vec=False):  # get the depth map, get rwd model,
        self.returned_dmaps_start = []
        self.seeds = []
        self.zvals = []

        self.returned_global_features_start = []

        self.seeds = seeds
        zvals = get_list_of_z(self.seeds)  # seeds should be random seeds
        self.zvals = zvals

        self.zvals[0].shape

        self.init_global_embeddings_at_start()

        self.global_embeddings_are_initialised = True

        return self

    def get_global_embeddings_eval_from_loss_G(self, seeds, return_global_feature_vec=False):  # get the depth map, get rwd model,
        self.returned_dmaps = []
        self.seeds = []
        self.zvals = []

        self.returned_global_features = []

        self.seeds = seeds
        zvals = get_list_of_z(self.seeds)  # seeds should be random seeds
        self.zvals = zvals

        self.zvals[0].shape

        self.init_global_embeddings()

        return self

    def _compute_and_backward_reward_loss(self, gen_z, gen_c, swapping_prob, accum_grad_gain, G_gain):
        nrs = self.hydra_cfg.data.nrs  # using neural render res of 256
        # with torch.autograd.profiler.record_function("Gmain_forward"):

        gen_c_template = self._get_reward_camera_template()

        # TAKE A DATALOADER OF IT
        gen_z_base = gen_z

        bform = self.format_batch(gen_c_template=gen_c_template, z=gen_z)

        bsize = bform["bsize"]  # make sure batch size = 2. if bigger split the batches
        ncams = bform["ncams"]
        gen_z = bform["gen_z"]
        gen_c = bform["gen_c"]

        # gen_c.requires_grad_(True)      #<- needed???????
        # gen_z.requires_grad_(True)

        bform["gen_c"].shape

        if self._reward_debug_enabled():
            cuda_mem_summary = torch.cuda.memory_summary()
            with open(RLHF_TUNE_CONFIG_DIR / "memory_cuda" / "cuda_mem_summary.txt", "w") as f:
                f.write(cuda_mem_summary)

        # n_iter=int(bsize/ncams)

        # g#en_z=gen_z.view(bsize,ncams,-1)
        # gen_c=gen_c.view(bsize,ncams,-1)

        iters_res = []

        if self._is_triple_dmap_reward_dtype():
            # run_config.data.dset_dict.selected_dtypes==['triple_dmap']
            gz_split = gen_z.split(ncams)  # tirpledmap
            gc_split = gen_c.split(ncams)  # tripledmap
            bsize = 1  # tripledmap

        else:
            gz_split = [gen_z]  # .split(ncams) #all other dtype
            gc_split = [gen_c]  # .split(ncams) #all other dtype

        accumulated_losses = []  # torch.tensor(0.0).cuda()

        for gen_z, gen_c in zip(gz_split, gc_split):
            has_calc_global = False
            global_vectors = None
            gen_img = None

            if self._is_sigma_rays_reward_dtype() or self._is_sigma_field_reward_dtype():
                rwd_model_input, global_vectors, has_calc_global = self._build_volumetric_reward_inputs(
                    self.G,
                    gen_z,
                    gen_c,
                    truncation_psi=self.hydra_cfg.rlhf_tune_hpms.G_sample_truncation_psi,
                    truncation_cutoff=self.hydra_cfg.rlhf_tune_hpms.G_sample_truncation_cutoff,
                    noise_mode=self.hydra_cfg.rlhf_tune_hpms.G_sample_noise_mode,
                    with_grad=self._is_sigma_field_reward_dtype(),
                    update_emas=True,
                    apply_zero_grad_threshold=True,
                )

            else:
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=nrs, drop_super_res=True, update_emas=self.hydra_cfg.rlhf_tune_hpms.update_emas_gdepth)  # change gen_c to gc_rep
                gen_img["image_raw"] = gen_img["image_raw"].detach()
                del _gen_ws
                torch.cuda.empty_cache()
                iters_res.append(gen_img)
                # self.DST.upsample_dmap_64_to_128(gen_depth)
                # gen_depth=self.upsample_dmap_64_to_128(gen_depth)

                rwd_model_input = self.DST.format_gen_out_for_rwd_input(gen_img=gen_img, bsize=bsize, ncams=ncams, gen_c=gen_c)
                has_calc_global = False

            # nb only for triple dmap!
            # rwd_model_input=self.DST.format_gen_out_for_rwd_input(gen_img=gen_img,bsize=1,ncams=ncams,gen_c=gen_c)

            n_comparisons = self.hydra_cfg.rlhf_tune_hpms.n_comparisons
            rwd_score_raw, global_vectors = self._compute_reward_score_raw(
                rwd_model_input,
                global_vectors,
                n_comparisons=n_comparisons,
                has_calc_global=has_calc_global,
                gen_z=gen_z,
                gen_c=gen_c,
            )
            rwd_score = self._compute_reward_score(rwd_score_raw)

            if gen_img is None:
                gen_img = self._render_reward_depth_images(gen_z, gen_c, swapping_prob)

            gen_depth_canonical = self._get_canonical_reward_depth(gen_img, ncams)
            nose_loss = self._compute_nose_loss(gen_depth_canonical)
            self._reward_debug("rwd_loss", rwd_score)

            rwd_losses = rwd_score + nose_loss

            logdict = {}
            self._report_reward_scalar("Loss/rwd_tuning/rwd_loss", rwd_score, logdict, "rwd_loss")
            self._report_reward_scalar("Loss/rwd_tuning/nose_loss", nose_loss, logdict, "nose_loss")

            # can do multibatch here

            self._reward_debug("rwd loss mean", rwd_losses.mean().mul(G_gain))
            self._reward_debug("nose loss", nose_loss)
            if self.hydra_cfg.rlhf_tune_hpms.lambda_rwd_model != 0.0:
                self._reward_debug("indiv reward scores", rwd_score_raw.detach().cpu().numpy().flatten())
                logdict["raw_rwd_score_mean"] = rwd_score_raw.mean()
                logger.info("raw_rwd_score_mean: %s", rwd_score_raw.mean())

            accumulated_losses.append(rwd_losses)

        rwd_losses = torch.hstack(accumulated_losses).mean()

        l1_loss_dmap = torch.zeros_like(rwd_losses)
        mse_loss_dmap = torch.zeros_like(rwd_losses)
        forward_mse_dmap = torch.zeros_like(rwd_losses)
        lpips_loss = torch.zeros_like(rwd_losses)
        depth_reg_total = torch.zeros_like(rwd_losses)
        lambda_dmap_forward_mse = float(self.hydra_cfg.rlhf_tune_hpms.get("lambda_dmap_forward_mse", 0.0))
        lambda_lpips = float(self.hydra_cfg.rlhf_tune_hpms.get("lambda_lpips", 0.0))
        # Keep the full consistency-loss chain together: these gates match
        # train_rlhf.py's old_G / old_G_ema / LPIPS wiring above.
        if self.hydra_cfg.rlhf_tune_hpms.lambda_dmap_l1 != 0.0 or self.hydra_cfg.rlhf_tune_hpms.lambda_dmap_mse != 0.0 or lambda_dmap_forward_mse != 0.0 or lambda_lpips != 0.0:
            depth_reg_losses = self.return_depth_reg_losses_ema(gen_z_base, gen_img)
            l1_loss_dmap = depth_reg_losses["l1"]
            mse_loss_dmap = depth_reg_losses["mse"]
            forward_mse_dmap = depth_reg_losses["forward_mse"]
            lpips_loss = depth_reg_losses["lpips"]
            depth_reg_total = depth_reg_losses["total"]
            self._reward_debug("l1 loss dmap", l1_loss_dmap)
            self._reward_debug("mse loss dmap", mse_loss_dmap)
            self._reward_debug("forward mse dmap", forward_mse_dmap)
            self._reward_debug("lpips loss", lpips_loss)
            self._report_reward_scalar("Loss/rwd_tuning/l1_loss_dmap", l1_loss_dmap)
            self._report_reward_scalar("Loss/rwd_tuning/mse_loss_dmap", mse_loss_dmap)
            self._report_reward_scalar("Loss/rwd_tuning/forward_mse_dmap", forward_mse_dmap)
            self._report_reward_scalar("Loss/rwd_tuning/lpips_loss", lpips_loss)
            self._report_reward_scalar("Loss/rwd_tuning/depth_reg_total", depth_reg_total)

        logdict["Loss/rwd_tuning/l1_loss_dmap"] = l1_loss_dmap.mean()
        logdict["Loss/rwd_tuning/mse_loss_dmap"] = mse_loss_dmap.mean()
        logdict["Loss/rwd_tuning/forward_mse_dmap"] = forward_mse_dmap.mean()
        logdict["Loss/rwd_tuning/lpips_loss"] = lpips_loss.mean()
        logdict["Loss/rwd_tuning/depth_reg_total"] = depth_reg_total.mean()
        logger.info(
            "l1_loss_dmap: %s | mse_loss_dmap: %s | forward_mse_dmap: %s | lpips_loss: %s | depth_reg_total: %s",
            l1_loss_dmap.mean(),
            mse_loss_dmap.mean(),
            forward_mse_dmap.mean(),
            lpips_loss.mean(),
            depth_reg_total.mean(),
        )

        # training_stats.report("Loss/rwd_tuning/nose_loss", nose_loss.mean())
        total_loss = rwd_losses + depth_reg_total

        l1_loss_sigma = torch.zeros_like(rwd_losses)
        front_growth_loss_sigma = torch.zeros_like(rwd_losses)
        if self.hydra_cfg.rlhf_tune_hpms.lambda_sigma_l1 != 0.0:
            # Keep the sigma-anchor branch coupled to the old_G_ema reference path.
            l1_loss_sigma = self.return_l1_loss_sigma_ema(gen_z, gen_c)  # ,swapping_prob,gen_img,rays_from_new=rwd_model_input)

            self._reward_debug("l1 loss sigma", l1_loss_sigma)
            self._report_reward_scalar("Loss/rwd_tuning/l1_loss_sigma", l1_loss_sigma)

        if float(self.hydra_cfg.rlhf_tune_hpms.get("lambda_sigma_front_growth", 0.0)) != 0.0:
            front_growth_loss_sigma = self.return_front_growth_loss_sigma_ema(gen_z, gen_c)
            self._reward_debug("front growth loss sigma", front_growth_loss_sigma)
            self._report_reward_scalar("Loss/rwd_tuning/front_growth_loss_sigma", front_growth_loss_sigma)

        logdict["Loss/rwd_tuning/l1_loss_sigma"] = l1_loss_sigma.mean()
        logdict["Loss/rwd_tuning/front_growth_loss_sigma"] = front_growth_loss_sigma.mean()
        logger.info("l1_loss_sigma: %s | front_growth_loss_sigma: %s", l1_loss_sigma.mean(), front_growth_loss_sigma.mean())

        total_loss = rwd_losses + depth_reg_total + l1_loss_sigma + front_growth_loss_sigma

        with torch.autograd.profiler.record_function("Gmain_backward"):
            # gloss_gain=0.5
            total_loss.mul(self.hydra_cfg.rlhf_tune_hpms.G_depth_gain).mul(accum_grad_gain).backward()

        self.global_step += 1
        if self._wandb_enabled():
            wandb.log(logdict, commit=True)

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, accum_grad_gain, cur_nimg):
        assert phase in ["Gmain", "Greg", "Gboth", "Dmain", "Dreg", "Dboth", "G_depth"]
        if self.G.rendering_kwargs.get("density_reg", 0) == 0:
            phase = {"Greg": "none", "Gboth": "Gmain"}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {"Dreg": "none", "Dboth": "Dmain"}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        # for pretrained PKL

        # neural_rendering_resolution = 128

        real_img_raw = filtered_resizing(
            real_img,
            size=neural_rendering_resolution,
            f=self.resample_filter,
            filter_mode=self.filter_mode,
        )

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {"image": real_img, "image_raw": real_img_raw}

        # PanoHead: real-side head-silhouette mask for the D seg channel.
        # Strictly additive + gated: only runs when a HeadMasker has been
        # attached (PanoHead arch). When absent (EG3D path) the real dict
        # is exactly {"image", "image_raw"} as before. Computed under
        # no_grad and detached -> a constant D input that does not affect
        # the R1 gradient (which differentiates D wrt the real RGB only).
        if getattr(self, "head_masker", None) is not None:
            with torch.no_grad():
                real_img["image_mask"] = self.head_masker(real_img["image"]).detach()
        # NOGEN_NODISC = False

        G_gain = 1.0 * self.hydra_cfg.rlhf_tune_hpms.G_gain_scale  # self.rlhf_opts.regularisation_terms["lambda_G_gain"]
        D_gain = 1.0 * self.hydra_cfg.rlhf_tune_hpms.D_gain_scale  # self.rlhf_opts.regularisation_terms["lambda_D_gain"]

        # Gmain: Maximize logits for generated images.
        if phase in ["Gmain", "Gboth"] and G_gain > 0:  # G_ema_rlhf
            with torch.autograd.profiler.record_function("Gmain_forward"):
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(
                    torch.rand((gen_c.shape[0], 1), device=gen_c.device) < swapping_prob,
                    c_swapped,
                    gen_c,
                )  # make identical for this one and historical G

                # swap prob should be same for both...
                gen_img, _gen_ws = self.run_G(
                    gen_z,
                    gen_c,
                    swapping_prob=swapping_prob,
                    neural_rendering_resolution=neural_rendering_resolution,
                    c_gen_conditioning=c_gen_conditioning,
                )

                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report("Loss/scores/fake", gen_logits)
                training_stats.report("Loss/signs/fake", gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report("Loss/G/loss", loss_Gmain)

                self.stats_tfevents.add_scalar("Loss/scores/fake", gen_logits.mean())
                self.stats_tfevents.add_scalar("Loss/signs/fake", gen_logits.mean().sign())
                self.stats_tfevents.add_scalar("Loss/G/loss", loss_Gmain.mean())
                # print("Gmain loss")
                # print(loss_Gmain)

            # --------------------------------------------------
            # RWD MODEL IN THIS POSE (END)
            # ---------------------------------------------------

            with torch.autograd.profiler.record_function("Gmain_backward"):
                # gloss_gain=0.5
                loss_Gmain.mean().mul(G_gain).mul(accum_grad_gain).backward()
                # rwd_losses.mul(G_gain).backward()

        # ----------------------------------------------------------------------
        # Calculating RLHF loss with rwd model in main loss function for G_depth (AM)
        if phase in ["G_depth"] and self.hydra_cfg.rlhf_tune_hpms.lambda_rwd_model > 0:  # G_ema_rlhf
            self._compute_and_backward_reward_loss(gen_z, gen_c, swapping_prob, accum_grad_gain, G_gain)

        # -------------------------------------------------------------------------
        # Density Regularization
        if phase in ["Greg", "Gboth"] and self.G.rendering_kwargs.get("density_reg", 0) > 0 and self.G.rendering_kwargs["reg_type"] == "l1" and G_gain > 0:
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(
                    torch.rand([], device=gen_c.device) < swapping_prob,
                    c_swapped,
                    gen_c,
                )
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function("style_mixing"):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(
                        torch.rand([], device=ws.device) < self.style_mixing_prob,
                        cutoff,
                        torch.full_like(cutoff, ws.shape[1]),
                    )
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs["density_reg_p_dist"]
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(
                all_coordinates,
                torch.randn_like(all_coordinates),
                ws,
                update_emas=False,
            )["sigma"]
            sigma_initial = sigma[:, : sigma.shape[1] // 2]
            sigma_perturbed = sigma[:, sigma.shape[1] // 2 :]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs["density_reg"]
            TVloss.mul(G_gain).mul(accum_grad_gain).backward()

        # Alternative density regularization
        if phase in ["Greg", "Gboth"] and self.G.rendering_kwargs.get("density_reg", 0) > 0 and self.G.rendering_kwargs["reg_type"] == "monotonic-detach" and G_gain > 0:
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(
                    torch.rand([], device=gen_c.device) < swapping_prob,
                    c_swapped,
                    gen_c,
                )
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1  # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1 / 256) * self.G.rendering_kwargs["box_warp"]  # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(
                all_coordinates,
                torch.randn_like(all_coordinates),
                ws,
                update_emas=False,
            )["sigma"]
            sigma_initial = sigma[:, : sigma.shape[1] // 2]
            sigma_perturbed = sigma[:, sigma.shape[1] // 2 :]

            monotonic_loss = torch.relu(sigma_initial.detach() - sigma_perturbed).mean() * 10
            monotonic_loss.mul(G_gain).mul(accum_grad_gain).backward()

            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(
                    torch.rand([], device=gen_c.device) < swapping_prob,
                    c_swapped,
                    gen_c,
                )
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function("style_mixing"):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(
                        torch.rand([], device=ws.device) < self.style_mixing_prob,
                        cutoff,
                        torch.full_like(cutoff, ws.shape[1]),
                    )
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1 / 256) * self.G.rendering_kwargs["box_warp"]
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(
                all_coordinates,
                torch.randn_like(all_coordinates),
                ws,
                update_emas=False,
            )["sigma"]
            sigma_initial = sigma[:, : sigma.shape[1] // 2]
            sigma_perturbed = sigma[:, sigma.shape[1] // 2 :]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs["density_reg"]
            TVloss.mul(G_gain).mul(accum_grad_gain).backward()

        # Alternative density regularization
        if phase in ["Greg", "Gboth"] and self.G.rendering_kwargs.get("density_reg", 0) > 0 and self.G.rendering_kwargs["reg_type"] == "monotonic-fixed" and G_gain > 0:
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(
                    torch.rand([], device=gen_c.device) < swapping_prob,
                    c_swapped,
                    gen_c,
                )
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1  # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1 / 256) * self.G.rendering_kwargs["box_warp"]  # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(
                all_coordinates,
                torch.randn_like(all_coordinates),
                ws,
                update_emas=False,
            )["sigma"]
            sigma_initial = sigma[:, : sigma.shape[1] // 2]
            sigma_perturbed = sigma[:, sigma.shape[1] // 2 :]

            monotonic_loss = torch.relu(sigma_initial - sigma_perturbed).mean() * 10
            monotonic_loss.mul(G_gain).mul(accum_grad_gain).backward()

            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(
                    torch.rand([], device=gen_c.device) < swapping_prob,
                    c_swapped,
                    gen_c,
                )
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function("style_mixing"):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(
                        torch.rand([], device=ws.device) < self.style_mixing_prob,
                        cutoff,
                        torch.full_like(cutoff, ws.shape[1]),
                    )
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1 / 256) * self.G.rendering_kwargs["box_warp"]
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(
                all_coordinates,
                torch.randn_like(all_coordinates),
                ws,
                update_emas=False,
            )["sigma"]
            sigma_initial = sigma[:, : sigma.shape[1] // 2]
            sigma_perturbed = sigma[:, sigma.shape[1] // 2 :]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs["density_reg"]
            TVloss.mul(G_gain).mul(accum_grad_gain).backward()

        # Dmain: Minimize logits for generated images.
        # USING_DISC = True
        loss_Dgen = 0
        if phase in ["Dmain", "Dboth"] and D_gain > 0:
            with torch.autograd.profiler.record_function("Dgen_forward"):
                gen_img, _gen_ws = self.run_G(
                    gen_z,
                    gen_c,
                    swapping_prob=swapping_prob,
                    neural_rendering_resolution=neural_rendering_resolution,
                    update_emas=True,
                )
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report("Loss/scores/fake", gen_logits)
                training_stats.report("Loss/signs/fake", gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function("Dgen_backward"):
                loss_Dgen.mean().mul(D_gain).mul(accum_grad_gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ["Dmain", "Dreg", "Dboth"] and D_gain > 0:
            name = "Dreal" if phase == "Dmain" else "Dr1" if phase == "Dreg" else "Dreal_Dr1"
            with torch.autograd.profiler.record_function(name + "_forward"):
                real_img_tmp_image = real_img["image"].detach().requires_grad_(phase in ["Dreg", "Dboth"])
                real_img_tmp_image_raw = real_img["image_raw"].detach().requires_grad_(phase in ["Dreg", "Dboth"])
                real_img_tmp = {
                    "image": real_img_tmp_image,
                    "image_raw": real_img_tmp_image_raw,
                }
                # PanoHead: forward the (detached, constant) head mask into
                # the D input. Gated/additive: only present when a HeadMasker
                # is attached. R1 grads are taken wrt image/image_raw only,
                # so the mask does not enter the penalty.
                if "image_mask" in real_img:
                    real_img_tmp["image_mask"] = real_img["image_mask"]

                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report("Loss/scores/real", real_logits)
                training_stats.report("Loss/signs/real", real_logits.sign())

                loss_Dreal = 0
                if phase in ["Dmain", "Dboth"]:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report("Loss/D/loss", loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ["Dreg", "Dboth"]:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function("r1_grads"), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(
                                outputs=[real_logits.sum()],
                                inputs=[
                                    real_img_tmp["image"],
                                    real_img_tmp["image_raw"],
                                ],
                                create_graph=True,
                                only_inputs=True,
                            )
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1, 2, 3]) + r1_grads_image_raw.square().sum([1, 2, 3])
                    else:  # single discrimination
                        with torch.autograd.profiler.record_function("r1_grads"), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(
                                outputs=[real_logits.sum()],
                                inputs=[real_img_tmp["image"]],
                                create_graph=True,
                                only_inputs=True,
                            )
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report("Loss/r1_penalty", r1_penalty)
                    training_stats.report("Loss/D/reg", loss_Dr1)

            with torch.autograd.profiler.record_function(name + "_backward"):
                (loss_Dreal + loss_Dr1).mean().mul(D_gain).mul(accum_grad_gain).backward()


# ----------------------------------------------------------------------------
