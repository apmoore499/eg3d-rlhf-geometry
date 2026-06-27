# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Main training loop."""

import autoroot  # noqa: F401

import copy
import glob
import json
import os
import pickle
import time

import dnnlib
import legacy
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from camera_utils import FOV_to_intrinsics, LookAtPoseSampler
from metrics import metric_main
from torch.utils.tensorboard import SummaryWriter
from torch_utils import misc, training_stats
from torch_utils.ops import conv2d_gradfix, grid_sample_gradfix
from training.crosssection_utils import sample_cross_section

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

import seaborn as sns

import pandas as pd
import pyvista as pv
from training.mesh_preview_utils import create_samples, stack_snapshot_images_fn
from training.analyse_tuned_rscore_perepoch import run_combined_epoch_analysis
from tqdm import tqdm


def filter_IR_series(input_column, pc=0.8):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = input_column.quantile(0.5 - pc / 2)
    Q3 = input_column.quantile(0.5 + pc / 2)

    # Calculate Interquartile Range (IQR)
    IQR = Q3 - Q1

    # Determine the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Create a mask for values within the IQR
    within_IQR_mask = (input_column >= Q1) & (input_column <= Q3)

    # Return the mask as a boolean Series
    return within_IQR_mask


def _wandb_enabled(loss):
    return bool(getattr(loss.hydra_cfg, "using_wandb", False)) and wandb is not None


def plot_double_mesh_for_survey(input_mesh, views=[-60, -30, 0, 30, 60], specular=0.35, scale_factor=0.5):
    window_size = 2048

    rot = pv.wrap(input_mesh)

    xrot = 90

    rot = rot.rotate_x(xrot, inplace=False)
    pl = pv.Plotter(window_size=[window_size, window_size], off_screen=True)

    pl.set_background("#363940")

    mesh1 = pl.add_mesh(
        rot,
        smooth_shading=False,
        show_edges=False,
        color=[220 / 255, 243 / 255, 252 / 255],
        specular=specular,
    )

    pl.enable_ssao(kernel_size=32, blur=False)

    pl.set_focus(rot.center)
    pl.enable_anti_aliasing()
    pl.camera_position = "yz"
    pl.camera.zoom(1.8)

    initial_angle = views[0]
    rotate_angles = views[1:]

    angle_intervals = [second - first for second, first in zip(views[1:], views[:-1])]

    # am_angle=initial_angle #19
    pl.camera.Azimuth(initial_angle)
    pl.camera.Elevation(3)

    ims = []

    for a in angle_intervals:  # in range(2):
        pl.render()
        img1 = pl.screenshot(filename=None, return_img=True)

        from PIL import Image

        pillow_image = Image.fromarray(img1)
        border_size = 40
        border_color = "#000000"
        image_with_border = Image.new(
            "RGB",
            (pillow_image.width + 2 * border_size, pillow_image.height + 2 * border_size),
            border_color,
        )
        image_with_border.paste(pillow_image, (border_size, border_size))

        ims.append(image_with_border)
        pl.camera.Azimuth(a)

    pl.render()
    img1 = pl.screenshot(filename=None, return_img=True)

    from PIL import Image

    pillow_image = Image.fromarray(img1)
    border_size = 40
    border_color = "#000000"
    image_with_border = Image.new(
        "RGB",
        (pillow_image.width + 2 * border_size, pillow_image.height + 2 * border_size),
        border_color,
    )
    image_with_border.paste(pillow_image, (border_size, border_size))

    ims.append(image_with_border)

    spacing = 75

    img = Image.new("RGB", (ims[0].width * len(views) + spacing * len(views), ims[0].height))

    for i in range(len(views)):
        img.paste(ims[i], (ims[0].width * i + i * spacing, 0))

    # Define the scale factor as a float between 0.0 and 1.0
    # scale_factor = 0.5  # Replace with your desired scale factor

    # Calculate the new width and height
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)

    # Resize the image
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    pl.clear()

    pl.close()

    return resized_img


def visualise_mesh(trimesh_object, ply_fn, window_size, zoom=1):
    rot = pv.wrap(trimesh_object)

    # yrot=0
    # zrot=0
    xrot = 90

    # rot = rot.rotate_y(yrot, inplace=False)
    # rot = rot.rotate_z(zrot, inplace=False)
    rot = rot.rotate_x(xrot, inplace=False)

    st = time.time()

    pl = pv.Plotter(window_size=[window_size, window_size], off_screen=True)

    # n_runs=1

    ims = []

    pl.set_background("#363940")
    # _ = pl.add_mesh(rot,pbr=True,metallic=0.1, roughness=0.5,smooth_shading=True)
    mesh1 = pl.add_mesh(rot, smooth_shading=False, color=[220 / 255, 243 / 255, 252 / 255], specular=0.25)
    pl.enable_ssao(kernel_size=64, blur=False)

    pl.set_focus(rot.center)
    pl.camera_position = "yz"
    pl.zoom(zoom)
    azimuth_angle = -60
    pl.camera.Azimuth(azimuth_angle)
    cdim = 100
    azimuth_angle = 30
    # for azimuth_angle in [-45,-30,-15,0,15,30,45]:
    for i in range(6):
        image = pl.screenshot(filename=None, return_img=True)
        ims.append(np.asarray(image)[cdim : window_size - cdim, cdim : window_size - cdim])
        pl.reset_camera()
        pl.camera.Azimuth(azimuth_angle)
        # pl.camera.zoom(zoom)

    out_fn = ply_fn.replace(".ply", ".jpg").replace("_mesh", f"_mesh_cat_zoom_{zoom}")

    ims = ims[2:-1]
    PIL.Image.fromarray(np.hstack(ims)).convert("RGB").save(out_fn)

    # ii=ims[-6:][2:-1]

    # pl.remove_actor(mesh1)

    pl.close()

    # print(f'time taken visualise mesh: {tt_avg:.3f} second')


from shape_utils import convert_sdf_samples_to_ply


# seeds=[1,2,3],G=G_ema,run_dir,shape_res=256,epoch=0
def visualise_mesh_using_pyvista(
    seeds,
    G,
    run_dir,
    shape_res=256,
    epoch=0,
    delete_ply=True,
    window_size=512,
    cl_frac=0.8,
    views=[-60, -30, 0, 30, 60],
    specular=0.35,
    scale_factor=0.5,
    outputting_bigmesh=False,
):
    G.eval()
    # This helper is called from tick-export code, not from the main training
    # scope, so it must derive its own device instead of relying on a free name.
    device = next(G.parameters()).device
    intrinsics = FOV_to_intrinsics(18.8333, device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get("avg_camera_pivot", [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get("avg_camera_radius", 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=device)

    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    # print('torch cat complete =')
    for seed in seeds:
        st = time.time()

        out_name = os.path.join(run_dir, f"seed_{seed}_epoch_{epoch}_mesh.ply")
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        # cl_frac=0.8
        if outputting_bigmesh:
            # print(f'exporting for seeed: {seed}')
            N = int(shape_res * cl_frac)

            c_length = float(G.rendering_kwargs["box_warp"] * 1 * cl_frac)

            max_batch = 1000000
            samples, voxel_origin, voxel_size = create_samples(N=N, voxel_origin=[0, 0, 0], cube_length=c_length)  # .reshape(1, -1, 3)

            samples = samples.to(z.device)
            sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
            transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
            transformed_ray_directions_expanded[..., -1] = -1
            head = 0

            # with tqdm(total=samples.shape[1]) as pbar:
            with torch.no_grad():
                while head < samples.shape[1]:
                    torch.manual_seed(0)
                    sigma = G.sample(
                        samples[:, head : head + max_batch],
                        transformed_ray_directions_expanded[:, : samples.shape[1] - head],
                        z,
                        conditioning_params,
                        truncation_psi=0.7,
                        truncation_cutoff=14,
                        noise_mode="const",
                    )["sigma"]
                    sigmas[:, head : head + max_batch] = sigma
                    head += max_batch
                    # pbar.update(max_batch)
            sigmas = sigmas.reshape((N, N, N)).cpu().numpy()
            sigmas = np.flip(sigmas, 0)
            # Trim the border of the extracted cube
            pad = int(10 * shape_res / 256)
            pad_value = -1000
            sigmas[:pad] = pad_value
            sigmas[-pad:] = pad_value
            sigmas[:, :pad] = pad_value
            sigmas[:, -pad:] = pad_value
            sigmas[:, :, :pad] = pad_value
            sigmas[:, :, -pad:] = pad_value
            # if shape_format == '.ply':

            # with mrcfile.new_mmap(out_name.replace('.ply','.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
            #    mrc.data[:] = sigmas
            # if seed==2:
            trimesh_object = convert_sdf_samples_to_ply(
                numpy_3d_sdf_tensor=np.transpose(sigmas, (2, 1, 0)),
                voxel_grid_origin=[-0.5, -0.5, -0.5],
                voxel_size=1.0 / shape_res,
                ply_filename_out=f"{out_name}",
                level=20,
                process=False,
            )

            et = time.time()

            tt = et - st

            # print(f"exported ply mesh in {tt:.2f} seconds")

            img_plot = plot_double_mesh_for_survey(trimesh_object, views=views, specular=specular, scale_factor=scale_factor)
            img_plot.save(out_name.replace(".ply", f"_stacked_views_{len(views)}.png"))

            if delete_ply:
                os.remove(out_name)

        out = G(z=z, c=conditioning_params, noise_mode="const", neural_rendering_resolution=128)  # for z, c in zip(grid_z, grid_c)]
        # images = torch.cat([o['image'].cpu() for o in out]).numpy()
        # images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
        # images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()

        imrgb = out["image"]

        import torchvision.transforms as tf

        rgb_im = tf.functional.to_pil_image((imrgb.squeeze(0) / 2 + 0.5).clip(0, 1))
        rgb_im.save(out_name.replace("_mesh.ply", "_rgb.jpg"))

        torch.save(obj=out, f=out_name.replace(".ply", f"output_for_dmap_rwd_val_seed_{seed}.pt"))

    return


# ----------------------------------------------------------------------------


"""
we need:
- camera vector for conditoning cam2world pose
- camera vector for the depth map projection angle
"""


def return_stacked_cams_ffhq(
    dmap_angles=[1.0, 0, -1.0],
    cam_radius=2.7,
    angle_p=0.0,
    cam_pivot=torch.tensor([0, 0, 0.0]),
    fov_deg=18.837,
):
    camera_stack = []
    # y0,y1,y2=dmap_angles
    intrinsics = FOV_to_intrinsics(fov_deg)
    for angle_y in dmap_angles:
        cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        camera_stack.append(camera_params)
    return torch.cat(camera_stack, 0)


# ----------------------------------------------------------------------------


def get_list_of_z(seeds, zdim=512, use_fat_tail=False):
    print(f"synthing z list with {len(seeds)} seeds and use fat tail: {use_fat_tail}")
    zd = 512
    device = torch.device("cuda")
    zs = [torch.from_numpy(np.random.RandomState(s).randn(1, 512)).to(device) for s in seeds]

    z_list = torch.cat(zs)
    print("z list synth, returning")
    return z_list


# ----------------------------------------------------------------------------


def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict()  # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = list(label_groups.keys())
        rnd.shuffle(label_order)
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)


# ----------------------------------------------------------------------------


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], "L").save(fname)
    if C == 3:
        PIL.Image.fromarray(img, "RGB").save(fname)

    return img


# Legacy note: reward-model imports used to rely on ad hoc sys.path injection before the framework moved to reward_model_training.
# import src as rlhf_src

# reward_model_pnet=rlhf_src.models.point_cloud.scalar_reward_pointnet2()
# reward_model_pnet.load_state_dict(torch.load('/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules/tmp/best_model.pt'))


# def get_canonical_dmap_cams_for_rlhf():

#     tdmap_cams=torch.load('/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/static_configs/triple_dmap_cameras.pt',map_location=torch.device('cpu'))
#     canon_cam=tdmap_cams[1].unsqueeze(0)
#     c=canon_cam
#     cam2world_matrix = c[:, :16].view(-1, 4, 4)
#     intrinsics = c[:, 16:25].view(-1, 3, 3)

#     return(dict(cam2world_matrix=cam2world_matrix,intrinsics=intrinsics))

# canon_cam=get_canonical_dmap_cams_for_rlhf()
# cam2world_matrix=canon_cam['cam2world_matrix']
# intrinsics=canon_cam['intrinsics']
# ray_sampler_static=RaySampler()


import hydra
import pandas as pd
import core_modules
import torch
import torch.nn as nn
from IPython.core.debugger import set_trace
from omegaconf import OmegaConf

# ------------------------------------------
# ------------------------------------------
# from core_modules.utils.finetuning_utils import aw98_helper
# ------------------------------------------
# from core_modules.utils.finetuning_utils import get_cfg_fn_from_id
# ------------------------------------------
from core_modules.utils.finetuning_utils import (
    DataHelperForEG3DLoss,
    get_datatype_from_model_id,
    load_rwd_model_from_cfg,
    MeshUtilsDataClass,
)

# ------------------------------------------
# from core_modules.utils.finetuning_utils import load_rwd_model_from_cfg_id
# ------------------------------------------
# from core_modules.utils.finetuning_utils import MeshUtilsDataClass


# ----------------------------------------------------------------------------


def calculate_tick_rewards(seedslist_visualisation, loss, cur_tick, zvals, run_dir, cur_nimg):
    # Uses loss.hydra_cfg throughout. `zvals` (z-latents for the seeds) and
    # `run_dir` are now passed by the caller (export_seedslist_visualisation),
    # which already computes both -- they used to be referenced here as free
    # names (NameError, previously hidden by a bare except), as was an unused
    # `hydra_cfg` param. `cur_nimg` is the current image count: the per-seed mesh
    # .pt files embed the epoch `cur_nimg // 1000` in their name (see
    # visualise_mesh_using_pyvista), so the glob below filters to THIS tick's files
    # only -- otherwise it matches every accumulated epoch's file per seed and the
    # `assert len(seeds_for_vis) == 1` below fails.

    # n_for_avg = self.hydra_cfg.n_samples_for_rwd_offset
    if loss.hydra_cfg.rwd_model_data_type == "triple_dmap":
        nrs = 128
        with torch.no_grad():
            rwd_scores = []
            gen_z = torch.tensor([1], device=torch.device("cuda:0"))

            gen_c_template = loss.get_triple_dmap_cams_for_rlhf()["gen_c"]

            gen_c_template = torch.vstack(gen_c_template)
            # TAKE A DATALOADER OF IT
            zloader = torch.utils.data.DataLoader(zvals, batch_size=2, shuffle=False, drop_last=False)

            for z in tqdm(zloader):
                bsize = z.shape[0]
                gen_c = gen_c_template[None, ...].expand(z.shape[0], -1, -1).to(gen_z.device).view(bsize * 3, -1)
                gen_z = z.unsqueeze(1).expand(-1, 3, -1).reshape(bsize * 3, -1)

                gz_split = gen_z.split(3)
                gc_split = gen_c.split(3)

                gen_imgs = []
                for gz, gc in zip(gz_split, gc_split):
                    gen_img, _gen_ws = loss.run_G(
                        gz,
                        gc,
                        swapping_prob=0.0,
                        neural_rendering_resolution=nrs,
                    )  # change gen_c to gc_rep
                    gen_imgs.append(gen_img["image_depth"])

                gen_depth = torch.vstack(gen_imgs)  # should be of batch_size x 3 x 128x128
                gen_depth = gen_depth.reshape(bsize, 3, nrs, nrs)  # torch.Size([2, 3, 128, 128])

                embeddings = loss.reward_model.external(gen_depth.unsqueeze(2))
                rwd_from_triple_dmap = loss.reward_model.forward(embeddings)

                rwd_scores.append(rwd_from_triple_dmap)

    if loss.hydra_cfg.rwd_model_data_type == "point_cloud_entire":
        epoch = f"{cur_nimg // 1000:06d}"
        all_files = glob.glob(os.path.join(run_dir, f"seed_*_epoch_{epoch}_meshoutput_for_dmap_rwd_val_seed_*.pt"))

        dict_of_seeds_fn = {}

        depth_maps = []
        for s in seedslist_visualisation:
            seeds_for_vis = []
            for a in all_files:
                if a.endswith(f"_seed_{s}.pt"):
                    seeds_for_vis.append(a)

            assert len(seeds_for_vis) == 1, "error more than 1 or no matching fn for seed "

            seed_files = torch.load(seeds_for_vis[0])

            im_depth = seed_files["image_depth"]
            depth_maps.append(im_depth)

        all_dmaps = torch.cat(depth_maps, 0)
        # the canonical (single-view) reward camera, shape (1, 25). Keep this as a
        # STABLE source and expand a fresh per-batch copy below -- previously `gen_c`
        # was reassigned in-loop (gen_c = gen_c.expand(...)), so after the first
        # batch it was (B, 25) and the final size-1 batch (odd # of vis seeds ->
        # batches [2,2,1]) failed to expand 2 -> 1.
        gen_c_canon = loss.get_canonical_dmap_cams_for_rlhf()["gen_c"]
        # perm = torch.randperm(nrs*nrs)
        # idx = perm[:self.hydra_cfg.data.n_point_samples_per_pcd_batch]

        # TAKE A DATALOADER OF IT
        dloader = torch.utils.data.DataLoader(all_dmaps, batch_size=2, shuffle=False, drop_last=False)
        rwd_scores = []

        with torch.no_grad():
            for gen_depth in tqdm(dloader):
                gen_c = gen_c_canon.expand(gen_depth.shape[0], -1).to(gen_depth.device)

                pcds_new = []

                for k in range(gen_depth.shape[0]):
                    pcd = loss.DST.modules_depthmap_to_pcd_from_image(
                        modules_depthmap_image=gen_depth[k],
                        downsample=False,
                        return_im=False,
                        gen_c=gen_c[k][None, ...],
                        nrs=128,
                        radius_cutoff=None,
                    )  # (N,3)

                    # inherited tune transform (subsample -> center -> mean-scale),
                    # applied per item on (N,3); replaces the retired data.* pcd
                    # flags (center_points/unit_scale_points/downsample_pcd_points/
                    # n_point_samples_per_pcd_batch).
                    pcd = loss.DST.tune_aug(pcd)

                    pcds_new.append(pcd.transpose(1, 0).unsqueeze(0))

                pcd_batch = torch.cat(pcds_new, dim=0)
                rwd_from_point_cloud = loss.reward_model.forward(pcd_batch)
                rwd_scores.append(rwd_from_point_cloud)

        rwd_scores = torch.vstack(rwd_scores).flatten()

        dict_of_rwds = {s: r.detach().cpu().item() for s, r in zip(seedslist_visualisation, rwd_scores)}

        rwds_df = pd.DataFrame.from_dict(dict_of_rwds, orient="index")
        rwds_df.to_csv(os.path.join(run_dir, f"rwds_df_tick_{cur_tick}_visualised_seeds.csv"))

        all_files = glob.glob(os.path.join(run_dir, f"seed_*_epoch*_meshoutput_for_dmap_rwd_val_seed_*.pt"))

        for a in all_files:
            os.remove(a)

        print("removed all .pt intermediate files")

        if _wandb_enabled(loss):
            wandb.log({f"FIRST_10_RANKINGS/": wandb.Table(dataframe=rwds_df), "tick": cur_tick})


def export_images(grid_z, grid_c, G_ema, grid_size, cur_nimg, loss, run_dir):
    """save the rgb and depth maps from the generator"""
    print(torch.cuda.memory_summary())

    out = [G_ema(z=z, c=c, noise_mode="random") for z, c in zip(grid_z, grid_c)]
    images = torch.cat([o["image"].cpu() for o in out]).numpy()
    images_raw = torch.cat([o["image_raw"].cpu() for o in out]).numpy()
    images_depth = -torch.cat([o["image_depth"].cpu() for o in out]).numpy()
    # comment out this one is 45mb save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
    imraw = save_image_grid(
        images_raw,
        os.path.join(run_dir, f"fakes{cur_nimg // 1000:06d}_raw.png"),
        drange=[-1, 1],
        grid_size=grid_size,
    )
    imdep = save_image_grid(
        images_depth,
        os.path.join(run_dir, f"fakes{cur_nimg // 1000:06d}_depth.png"),
        drange=[images_depth.min(), images_depth.max()],
        grid_size=grid_size,
    )

    if _wandb_enabled(loss):
        wandb.log({"images_raw": wandb.Image(imraw)})
        wandb.log({"images_depth": wandb.Image(imdep)})


def export_seedslist_visualisation(loss, cur_tick, first_tick_export, seedslist_visualisation, G_ema, run_dir, cur_nimg):
    """given a list of seeds seedslist_visualisation, sample seed random noise z and then visualise the geometries, sampled from G_ema"""
    zvals = get_list_of_z(seedslist_visualisation)  # seeds should be random seeds

    if cur_tick >= first_tick_export:
        print("pausing here")

        views = [-60, -30, 0, 30, 60]

        visualise_mesh_using_pyvista(
            seeds=seedslist_visualisation,
            G=G_ema,
            run_dir=run_dir,
            shape_res=loss.hydra_cfg.vis_shape_res,
            epoch=f"{cur_nimg // 1000:06d}",
            delete_ply=True,
            views=views,
            specular=0.9,
            outputting_bigmesh=True,
        )

        # img_plot.save(out_name.replace('.ply',f'_stacked_views_{len(views)}.jpg'))

        image_files = glob.glob(os.path.join(run_dir, f"seed_*_stacked_views_{len(views)}.png"))
        # sort images by time created
        image_files.sort(key=os.path.getmtime)
        stacked_image = stack_snapshot_images_fn(image_files)

        # convert to thumbnail
        # ------------------------------
        # width, height = stacked_image.size
        # width_ratio = width / 1000
        # new_width, new_height = int(1 / width_ratio * width), int(1 / width_ratio * height)
        # stacked_image.thumbnail(size=(new_width, new_height))
        # ------------------------------

        stacked_im_fn = os.path.join(run_dir, f"stacked_im_cur_nimg_{cur_nimg // 1000:06d}.jpg")
        stacked_image.save(stacked_im_fn, quality=95)
        for i in image_files:
            os.remove(i)

        if _wandb_enabled(loss):
            wandb.log({"meshes": wandb.Image(stacked_im_fn)})

    # get the dmaps......
    # f"output_for_dmap_rwd_val_seed_{seed}.pt"))
    # torch.save(obj=out, f=out_name.replace(".ply", f"output_for_dmap_rwd_val_seed_{seed}.pt"))
    # out_name = os.path.join(run_dir, f"output_for_dmap_rwd_val_seed_{seed}.pt")

    if loss.reward_model is None:
        print("no reward model loaded, not calculating tick rewards")
    else:
        try:
            calculate_tick_rewards(seedslist_visualisation, loss, cur_tick, zvals, run_dir, cur_nimg)
        except Exception as e:
            # tick-reward visualisation is non-essential -- don't crash the run,
            # but surface the REAL error instead of mislabeling it "no reward model
            # loaded" (it usually means the per-seed mesh .pt files this path globs
            # for weren't written, e.g. in a short/smoke run).
            import traceback

            print(f"calculate_tick_rewards failed at tick {cur_tick}; skipping tick-reward logging: {e}")
            traceback.print_exc()

    return


def save_network_snapshot(training_set_kwargs, G, D, G_ema, augment_pipe, run_dir, cur_nimg, rank, num_gpus):
    snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
    for name, module in [
        ("G", G),
        ("D", D),
        ("G_ema", G_ema),
        ("augment_pipe", augment_pipe),
    ]:
        if module is not None:
            if num_gpus > 1:
                misc.check_ddp_consistency(module, ignore_regex=r".*\.[^.]+_(avg|ema)")
            module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
        snapshot_data[name] = module
        del module  # conserve memory

        # return snapshot_data

    snapshot_pkl = os.path.join(run_dir, f"network-snapshot-{cur_nimg // 1000:06d}.pkl")
    if rank == 0:
        with open(snapshot_pkl, "wb") as f:
            pickle.dump(snapshot_data, f)

    return (snapshot_data, snapshot_pkl)


def _save_reward_baseline_histogram(loss, rwd_scores, rwd_scores_orig, rwd_scores_IQR, seeds, run_dir, cur_tick):
    # Visualisation/logging side-effects of the initial-generator reward baseline
    # (histogram PNG + per-seed CSV + optional wandb upload). Separated from the
    # baseline *computation* (mu_hat / sigma_hat / hard_nose_depth) so the latter can
    # run whenever standardisation is needed, while this stays gated on plot_rwd_dist.
    plt.hist(rwd_scores)
    out_fn = os.path.join(run_dir, f"reward_histogram_tick_{cur_tick}.png")
    plt.savefig(out_fn)
    plt.clf()

    rwd_scores_orig = pd.DataFrame(rwd_scores_orig)
    rwd_scores_orig.columns = ["rwd_score"]
    rwd_scores_orig["seed"] = seeds
    rwd_scores_orig["in_ir_r"] = rwd_scores_IQR
    rwd_scores_orig.to_csv(os.path.join(run_dir, f"rwds_df_tick_{cur_tick}.csv"))

    if _wandb_enabled(loss):
        wandb.log({"rewards_table": wandb.Table(dataframe=rwd_scores_orig), "tick": cur_tick})
        wandb.log({"reward_histogram": wandb.Image(out_fn), "tick": cur_tick})


def plot_rwd_dist_top_bottom(loss, cur_tick, G_ema, run_dir, cur_nimg):
    """plot reward histogram and visualise top / bottom - ranked seed geometries"""
    seeds = [i for i in range(loss.hydra_cfg.plot.minseed, loss.hydra_cfg.plot.maxseed)]
    rwd_scores = loss.get_rwd_scores_eval_from_loss_G(seeds, G_ema).detach().cpu()

    # 1. get all original reward scores and log them as a csv file

    rwd_scores_orig = pd.Series(rwd_scores)
    rwd_scores_orig = pd.DataFrame(rwd_scores_orig)

    rwd_scores_orig.columns = ["rwd_score"]
    rwd_scores_orig["seed"] = seeds
    rwd_scores_orig["in_ir_r"] = filter_IR_series(rwd_scores_orig.rwd_score)

    rwd_scores_orig.to_csv(os.path.join(run_dir, f"rwds_df_tick_{cur_tick}.csv"))

    if _wandb_enabled(loss):
        wandb.log({f"1000_SEEDS_RANKINGS/": wandb.Table(dataframe=rwd_scores_orig), "tick": cur_tick})

    rwd_scores_cleaned = rwd_scores_orig[rwd_scores_orig["in_ir_r"]]

    plt.clf()
    plt.cla()
    plt.close()

    plt.hist(rwd_scores_cleaned["rwd_score"])  # , bins=30)
    # save histogram
    # plt.savefig(os.path.join(self.hydra_cfg.click_legacy_args.outdir,'reward_histogram_initial.png'))
    out_fn = os.path.join(run_dir, f"reward_histogram_tick_{cur_tick}.png")
    plt.savefig(out_fn)

    if _wandb_enabled(loss):
        wandb.log({"reward_histogram": wandb.Image(out_fn), "tick": cur_tick})

    # EXPORT TOP AND BOTTOM MESHES

    plt.clf()

    rwds_dfs = {}
    rwds_dfs_ims = {}

    rwds_df_top = rwd_scores_cleaned.sort_values(by="rwd_score", ascending=False)
    rwds_df_bottom = rwd_scores_cleaned.sort_values(by="rwd_score", ascending=True)

    rwds_dfs["top"] = rwds_df_top
    rwds_dfs["bottom"] = rwds_df_bottom

    print("pausing here")

    for condition in ["top", "bottom"]:
        rwds_df = rwds_dfs[condition]

        # get top 10 seeds....

        rwds_df_seeds = rwds_df.head(loss.hydra_cfg.nm_vis)["seed"].values.tolist()

        views = [-60, -30, 0, 30, 60]

        visualise_mesh_using_pyvista(
            seeds=rwds_df_seeds,
            G=G_ema,
            run_dir=run_dir,
            shape_res=loss.hydra_cfg.vis_shape_res,
            epoch=cur_tick,
            delete_ply=True,
            views=views,
            specular=0.9,
            outputting_bigmesh=True,
        )

        # img_plot.save(out_name.replace('.ply',f'_stacked_views_{len(views)}.jpg'))

        image_files = glob.glob(os.path.join(run_dir, f"seed_*_stacked_views_{len(views)}.png"))
        # sort images by time created
        image_files.sort(key=os.path.getmtime)
        stacked_image = stack_snapshot_images_fn(image_files)

        # convert to thumbnail
        # ------------------------------
        width, height = stacked_image.size
        width_ratio = width / 1000
        new_width, new_height = int(1 / width_ratio * width), int(1 / width_ratio * height)
        stacked_image.thumbnail(size=(new_width, new_height))
        # ------------------------------

        stacked_im_fn = os.path.join(run_dir, f"stacked_im_cur_nimg_{cur_nimg // 1000:06d}_{condition}_OF_1000.jpg")
        stacked_image.save(stacked_im_fn, quality=95)

        rwds_dfs_ims[condition] = stacked_image

        for i in image_files:
            os.remove(i)

        all_files = glob.glob(os.path.join(run_dir, f"seed_*_epoch*_meshoutput_for_dmap_rwd_val_seed_*.pt"))

        for a in all_files:
            os.remove(a)

    if _wandb_enabled(loss):
        wandb.log({f"TOP_OF_1000": wandb.Image(rwds_dfs_ims["top"]), "tick": cur_tick})  # IMAGE
        wandb.log({f"BOTTOM_OF_1000": wandb.Image(rwds_dfs_ims["bottom"]), "tick": cur_tick})  # IMAGE
        # DATFRAME

    if loss.hydra_cfg.export_indiv_dmaps:
        loss_dmaps = torch.cat(loss.returned_dmaps)  # .flatten()
        loss_seeds = torch.tensor(loss.seeds).flatten()

        # then output.

        out_folder = os.path.join(run_dir, f"tick_out_{cur_tick}")
        os.makedirs(out_folder, exist_ok=True)

        for k in torch.arange(loss_seeds.shape[0]):
            fn_out = os.path.join(out_folder, f"dmap_seed_{k}.pt")
            torch.save(obj=loss_dmaps[k], f=fn_out)


def training_loop(
    run_dir=".",  # Output directory.
    training_set_kwargs={},  # Options for training set.
    data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
    G_kwargs={},  # Options for generator network.
    D_kwargs={},  # Options for discriminator network.
    G_opt_kwargs={},  # Options for generator optimizer.
    D_opt_kwargs={},  # Options for discriminator optimizer.
    augment_kwargs=None,  # Options for augmentation pipeline. None = disable.
    loss_kwargs={},  # Options for loss function.
    metrics=[],  # Metrics to evaluate during training.
    random_seed=0,  # Global random seed.
    num_gpus=1,  # Number of GPUs participating in the training.
    rank=0,  # Rank of the current process in [0, num_gpus[.
    batch_size=4,  # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu=4,  # Number of samples processed at a time by one GPU.
    ema_kimg=10,  # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup=0.05,  # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval=None,  # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval=16,  # How often to perform regularization for D? None = disable lazy regularization.
    augment_p=0,  # Initial value of augmentation probability.
    ada_target=None,  # ADA target value. None = fixed p.
    ada_interval=4,  # How often to perform ADA adjustment?
    ada_kimg=500,  # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg=25000,  # Total length of the training, measured in thousands of real images.
    kimg_per_tick=4,  # Progress snapshot interval.
    image_snapshot_ticks=1,  # How often to save image snapshots? None = disable.
    network_snapshot_ticks=10,  # How often to save network snapshots? None = disable.
    resume_pkl=None,  # Network pickle to resume training from.
    resume_kimg=0,  # First kimg to report when resuming training.
    cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
    abort_fn=None,  # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn=None,  # Callback function for updating training progress. Called for all ranks.
    hydra_cfg=None,
):
    # Initialize.

    import torch

    start_time = time.time()
    device = torch.device("cuda", rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False  # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False  # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True
    # torch.set_float32_matmul_precision('medium')

    conv2d_gradfix.enabled = True  # Improves training speed. # TODO: ENABLE
    grid_sample_gradfix.enabled = False  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print("Loading training set...")
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # subclass of training.dataset.Dataset

    data_loader_kwargs.update({"num_workers": 0, "prefetch_factor": None})
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=training_set,
            sampler=training_set_sampler,
            batch_size=batch_size // num_gpus,
            **data_loader_kwargs,
        )
    )
    if rank == 0:
        print()
        print("Num images: ", len(training_set))
        print("Image shape:", training_set.image_shape)
        print("Label shape:", training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print("Constructing networks...")
    common_kwargs = dict(
        c_dim=training_set.label_dim,
        img_resolution=training_set.resolution,
        img_channels=training_set.num_channels,
    )
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
    G.register_buffer("dataset_label_std", torch.tensor(training_set.get_label_std()).to(device))
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # G_ema_rlhf=copy.deepcopy(G_ema).eval().to('cpu')
    # G_rlhf=copy.deepcopy(G).eval().to('cpu')
    # Resume from existing pickle.

    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [("G", G), ("D", D), ("G_ema", G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        misc.print_module_summary(D, [img, c])

    # Setup augmentation.
    if rank == 0:
        print("Setting up augmentation...")
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex="Loss/signs/real")

    # Distribute across GPUs.
    if rank == 0:
        print(f"Distributing across {num_gpus} GPUs...")
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print("Setting up training phases...")

    # from hydra import compose, initialize
    # from omegaconf import OmegaConf

    # loss_kwargs['G_ema_rlhf']=G_ema_rlhf
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs)  # subclass of training.loss.Loss

    # from the run id, you read in relevant info

    # -----

    # -----
    import pandas as pd

    reward_model = load_rwd_model_from_cfg(current_id=hydra_cfg.rwd_model_id, strict=False)

    if hydra_cfg.get("rwd_id_final_tuned_weights"):
        newer_sd_fn = hydra_cfg.rwd_id_final_tuned_weights
        sd = torch.load(newer_sd_fn)
        reward_model.load_state_dict(sd, strict=False)
        # update here...
    dtype = get_datatype_from_model_id(current_id=hydra_cfg.rwd_model_id)
    hydra_cfg.rwd_model_data_type = dtype
    # reward_model = load_rwd_model_rlhf(hydra_cfg)
    print("reward model load")

    loss.reward_model = reward_model

    loss.hydra_cfg = hydra_cfg

    if loss.hydra_cfg.pretrained_modules.old_G:
        old_G = copy.deepcopy(G).eval().to("cpu")
        loss.old_G = old_G

    if loss.hydra_cfg.pretrained_modules.old_G_ema:
        old_G_ema = copy.deepcopy(G_ema).eval().to("cpu")
        loss.old_G_ema = old_G_ema

    if loss.hydra_cfg.pretrained_modules.LPIPS:
        import lpips

        LPIPS = lpips.LPIPS(net="vgg").cuda()
        loss.LPIPS = LPIPS

    loss.MSE_LOSS = torch.nn.functional.mse_loss
    loss.L1_LOSS = torch.nn.functional.l1_loss

    if type(loss.reward_model) == None:
        loss.DST = None
    else:
        loss.DST = DataHelperForEG3DLoss(hcfg_fn_rwd_model=hydra_cfg)  # hydra cfg is for the EG3D training!!!
        loss.MUDC = MeshUtilsDataClass()

    # PanoHead: attach an on-the-fly BiSeNet head-silhouette masker so the
    # real images get an "image_mask" seg channel for MaskDualDiscriminator.
    # Strictly additive + gated: only the PanoHead arch uses the masked D
    # (class name contains "MaskDualDiscriminator"); the EG3D DualDiscriminator
    # never does, so loss.head_masker stays unset and loss.py is untouched.
    if "MaskDualDiscriminator" in D_kwargs.get("class_name", ""):
        from training.panohead_nets.head_mask import HeadMasker

        loss.head_masker = HeadMasker(device=device)
        print("[PanoHead] attached BiSeNet HeadMasker for D seg channel")

    phases = []
    for name, module, opt_kwargs, reg_interval in [
        ("G", G, G_opt_kwargs, G_reg_interval),
        ("D", D, D_opt_kwargs, D_reg_interval),
    ]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs)  # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name + "both", module=module, opt=opt, interval=1)]
        else:  # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta**mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs)  # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name + "main", module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name + "reg", module=module, opt=opt, interval=reg_interval)]

    if loss.hydra_cfg.rlhf_opt.type == "original":
        print("using orig optimiser")
        # add in another phase for depth map regularisaer (RLHF)
        rwd_optimizer = dnnlib.util.construct_class_by_name(G.parameters(), **G_opt_kwargs)  # subclass of torch.optim.Optimizer

    else:
        raise ValueError(f"Unsupported eg3d RLHF optimizer type={loss.hydra_cfg.rlhf_opt.type}. Only rlhf_opt.type=original is maintained in the active fine-tune runtime.")

    phases += [dnnlib.EasyDict(name="G_depth", module=G, opt=rwd_optimizer, interval=1)]

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print("Exporting sample images...")
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, "reals.png"), drange=[0, 255], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(min(batch_gpu, 8))
        grid_c = torch.from_numpy(labels).to(device).split(min(batch_gpu, 8))

    # Initialize logs.
    if rank == 0:
        print("Initializing logs...")
    stats_collector = training_stats.Collector(regex=".*")
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "w")
        try:
            import torch.utils.tensorboard as tensorboard

            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print("Skipping tfevents export:", err)

    loss.stats_tfevents = stats_tfevents
    # Train.
    if rank == 0:
        print(f"Training for {total_kimg} kimg...")
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)

    # set train tick stop value...

    # rlhf_o=loss_kwargs['rlhf_opts']
    train_tick_stop = loss.hydra_cfg.train_tick_stop

    print("initialising global vecs for comparison")

    print("train for this many epoc")
    print(train_tick_stop)
    print("----------------------")
    if not loss.hydra_cfg.get("output_mesh_images_only", False) and not loss.hydra_cfg.get("eval_metrics_only", False):
        while True:
            # Fetch training data.
            with torch.autograd.profiler.record_function("data_fetch"):
                phase_real_img, phase_real_c = next(training_set_iterator)
                phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
                phase_real_c = phase_real_c.to(device).split(batch_gpu)
                all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
                all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
                all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
                all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
                all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

                dmap_cameras_three = [return_stacked_cams_ffhq().detach().cpu().numpy() for _ in range(batch_size)]
                dmap_cameras_three = torch.from_numpy(np.stack(dmap_cameras_three)).pin_memory().to(device)
                dmc_all_three = [dmc.split(batch_gpu) for dmc in dmap_cameras_three.split(batch_size)]

                dmap_cameras_canonical = [return_stacked_cams_ffhq(dmap_angles=[0.0]).detach().cpu().numpy() for _ in range(batch_size)]
                dmap_cameras_canonical = torch.from_numpy(np.stack(dmap_cameras_canonical)).pin_memory().to(device)
                dmc_canonical = [dmc.split(batch_gpu) for dmc in dmap_cameras_canonical.split(batch_size)]

                rl_cameras = dict(dmc_all_three=dmc_all_three, dmc_canonical=dmc_canonical)

            plt.clf()
            plt.cla()
            plt.close()

            if loss.rwd_affine_offset is None and (loss.hydra_cfg.plot_rwd_dist or loss.hydra_cfg.normalise_scalar_rwd) and loss.hydra_cfg.init_seeds_first and cur_tick == 0:
                seeds = [i for i in range(loss.hydra_cfg.plot.minseed, loss.hydra_cfg.plot.maxseed)]  # GET INITIAL SEED IDX
                seeds_for_pairs = [i for i in range(loss.hydra_cfg.pair_seeds.minseed, loss.hydra_cfg.pair_seeds.maxseed)]  # GET COMPARISON SEED IDX

                # calculate the embeddings
                # loss.get_global_embeddings_eval_from_loss_G(seeds)

                loss.initialise_global_embeddings_eval_from_loss_G(seeds)

                all_dmaps = loss.returned_dmaps_start  # hard_nose_depth
                i = 10
                # stack and get min depth for middle region of image, ie the nose
                tvs = torch.vstack(loss.returned_dmaps_start)
                tvvs = tvs[:, :, 63 - i : 63 + i, 63 - i : 63 + i]
                nose_min = tvvs.min().item()

                loss.hydra_cfg.rlhf_tune_hpms.hard_nose_depth = nose_min

                loss.returned_global_features_start = torch.vstack(loss.returned_global_features_start)  # ,1)

                rwd_scores = loss.get_rwd_scores_eval_from_loss_G(seeds).detach().cpu().numpy().flatten()

                # -----

                # -----
                # filter it..

                rwd_scores_orig = pd.Series(rwd_scores)
                rwd_scores_IQR = filter_IR_series(rwd_scores_orig)
                rwd_scores = rwd_scores_orig[rwd_scores_IQR]

                # rwd_scores = loss.return_rwd_plot()
                mean_score = rwd_scores.mean()
                loss.rwd_affine_offset = mean_score.item()

                loss.min_r = rwd_scores.min()
                loss.max_r = rwd_scores.max()

                # Ziegler et al. (2019) reward standardisation: normalise the reward
                # model so its scores on samples from the *initial* (pre-tuned)
                # generator have mean 0 and variance 1. offset = mean (above),
                # scale = 1 / std. (Previously scale was 1/(max_r - min_r), a range
                # normalisation -- not the paper's variance normalisation.) min_r /
                # max_r are retained for the clamped_iqr tune_type and the histograms.
                rwd_std = rwd_scores.std()
                # Floor the std before inverting so a near-degenerate initial-reward
                # spread can't blow rwd_scale up (a tiny std -> huge scale -> the
                # standardised reward rockets past the clamp after ~1 tick; see the
                # sigma smoke where std~0.07 gave scale~13.9). Floor is a fraction of
                # |offset| so it adapts across reward dtypes (sigma ~0.3, dmap ~16).
                # rwd_std_floor_frac=0 disables it (pure Ziegler 1/std).
                std_floor = loss.hydra_cfg.get("rwd_std_floor_frac", 0.0) * abs(loss.rwd_affine_offset)
                rwd_std_eff = max(float(rwd_std), std_floor)
                rwd_scale = 1.0 / rwd_std_eff if rwd_std_eff > 1e-8 else 1.0
                # Absolute cap: a near-degenerate reward model (tiny std AND ~0 offset,
                # so the std_floor can't engage) would otherwise give a huge scale that
                # saturates the clamp instantly (e.g. sdmap smoke model -> scale ~87).
                rwd_scale_max = float(loss.hydra_cfg.get("rwd_scale_max", 0.0))
                if rwd_scale_max > 0.0:
                    rwd_scale = min(rwd_scale, rwd_scale_max)
                loss.rwd_scale = rwd_scale

                print("rwd affine offset: ", loss.rwd_affine_offset)
                print("loss lower Q range: ", loss.min_r)
                print("loss upper Q range: ", loss.max_r)

                print("rwd scale: ", loss.rwd_scale)

                print("min max rwd pre:")

                print(loss.min_r)
                print(loss.max_r)

                # Baseline computation above (offset / scale / hard_nose_depth) runs
                # whenever standardisation is on; only the histogram/CSV/wandb logging
                # is gated on plot_rwd_dist.
                if loss.hydra_cfg.plot_rwd_dist:
                    _save_reward_baseline_histogram(loss, rwd_scores, rwd_scores_orig, rwd_scores_IQR, seeds, run_dir, cur_tick)

            if loss.rwd_affine_offset is None and (loss.hydra_cfg.plot_rwd_dist or loss.hydra_cfg.normalise_scalar_rwd) and loss.hydra_cfg.init_seeds_first and cur_tick > 0:
                seeds = [i for i in range(loss.hydra_cfg.plot.minseed, loss.hydra_cfg.plot.maxseed)]  # GET INITIAL SEED IDX
                seeds_for_pairs = [i for i in range(loss.hydra_cfg.pair_seeds.minseed, loss.hydra_cfg.pair_seeds.maxseed)]  # GET COMPARISON SEED IDX

                # calculate the embeddings
                # loss.get_global_embeddings_eval_from_loss_G(seeds)

                loss.get_global_embeddings_eval_from_loss_G(seeds)

                all_dmaps = loss.returned_dmaps  # hard_nose_depth
                i = 10
                # stack and get min depth for middle region of image, ie the nose
                tvs = torch.vstack(loss.returned_dmaps)
                tvvs = tvs[:, :, 63 - i : 63 + i, 63 - i : 63 + i]
                nose_min = tvvs.min().item()

                loss.hydra_cfg.rlhf_tune_hpms.hard_nose_depth = nose_min

                loss.returned_global_features = torch.vstack(loss.returned_global_features)  # ,1)

                rwd_scores = loss.get_rwd_scores_eval_from_loss_G(seeds).detach().cpu().numpy().flatten()

                # -----

                # -----
                # filter it..

                rwd_scores_orig = pd.Series(rwd_scores)
                rwd_scores_IQR = filter_IR_series(rwd_scores_orig)
                rwd_scores = rwd_scores_orig[rwd_scores_IQR]

                # rwd_scores = loss.return_rwd_plot()
                mean_score = rwd_scores.mean()
                loss.rwd_affine_offset = mean_score.item()

                loss.min_r = rwd_scores.min()
                loss.max_r = rwd_scores.max()

                # Ziegler et al. (2019) reward standardisation: normalise the reward
                # model so its scores on samples from the *initial* (pre-tuned)
                # generator have mean 0 and variance 1. offset = mean (above),
                # scale = 1 / std. (Previously scale was 1/(max_r - min_r), a range
                # normalisation -- not the paper's variance normalisation.) min_r /
                # max_r are retained for the clamped_iqr tune_type and the histograms.
                rwd_std = rwd_scores.std()
                # Floor the std before inverting so a near-degenerate initial-reward
                # spread can't blow rwd_scale up (a tiny std -> huge scale -> the
                # standardised reward rockets past the clamp after ~1 tick; see the
                # sigma smoke where std~0.07 gave scale~13.9). Floor is a fraction of
                # |offset| so it adapts across reward dtypes (sigma ~0.3, dmap ~16).
                # rwd_std_floor_frac=0 disables it (pure Ziegler 1/std).
                std_floor = loss.hydra_cfg.get("rwd_std_floor_frac", 0.0) * abs(loss.rwd_affine_offset)
                rwd_std_eff = max(float(rwd_std), std_floor)
                rwd_scale = 1.0 / rwd_std_eff if rwd_std_eff > 1e-8 else 1.0
                # Absolute cap: a near-degenerate reward model (tiny std AND ~0 offset,
                # so the std_floor can't engage) would otherwise give a huge scale that
                # saturates the clamp instantly (e.g. sdmap smoke model -> scale ~87).
                rwd_scale_max = float(loss.hydra_cfg.get("rwd_scale_max", 0.0))
                if rwd_scale_max > 0.0:
                    rwd_scale = min(rwd_scale, rwd_scale_max)
                loss.rwd_scale = rwd_scale

                print("rwd affine offset: ", loss.rwd_affine_offset)
                print("loss lower Q range: ", loss.min_r)
                print("loss upper Q range: ", loss.max_r)

                print("rwd scale: ", loss.rwd_scale)

                print("min max rwd pre:")

                print(loss.min_r)
                print(loss.max_r)

                # delete loss.returned_dmaps to save memory (always; not just when plotting)
                loss.returned_dmaps.clear()

                # Baseline computation above (offset / scale / hard_nose_depth) runs
                # whenever standardisation is on; only the histogram/CSV/wandb logging
                # is gated on plot_rwd_dist.
                if loss.hydra_cfg.plot_rwd_dist:
                    _save_reward_baseline_histogram(loss, rwd_scores, rwd_scores_orig, rwd_scores_IQR, seeds, run_dir, cur_tick)

            # Depth Map Reglarisation

            # Fallback: if no baseline ran (init_seeds_first off, or neither plotting
            # nor standardisation requested) the reward is used RAW -> offset 0 /
            # scale 1. Loudly warn if standardisation was requested but did not engage.
            if loss.rwd_affine_offset is None:
                if loss.hydra_cfg.normalise_scalar_rwd:
                    print("WARNING: normalise_scalar_rwd=True but the reward baseline did not run (requires init_seeds_first=True); reward will be RAW / un-standardised.")
                loss.rwd_affine_offset = 0.0  # set dummy value
                loss.rwd_scale = 1.0

            # loss.old_G.load_state_dict(loss.G.state_dict())#.clone() #update state dict each epoch, see effect on mesh quality overall

            # inital image export

            # Execute training phases.
            for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
                # rewrite it so that it's like, batch_size=16 for example, and you split the inputs all_gen_Z, all_gen_c up by accum_steps

                # if not phase.name=='G_depth':
                #    continue #skip all but G_dept for experiemmnt. see what happens.
                accum_iter = loss.hydra_cfg.rlhf_tune_hpms.accum_steps

                phase.opt.zero_grad(set_to_none=True)

                if batch_idx % phase.interval != 0:
                    continue
                if phase.start_event is not None:
                    phase.start_event.record(torch.cuda.current_stream(device))

                # Accumulate gradients.

                phase.module.requires_grad_(True)

                for real_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c):
                    # Parcel the per-GPU batch into accum_iter micro-batches.
                    # torch.chunk clamps to the batch size, so tiny smoke
                    # batches (bsize < accum_iter) yield fewer, non-empty
                    # chunks instead of crashing.
                    real_img_chunk = real_img.chunk(accum_iter)
                    real_c_chunk = real_c.chunk(accum_iter)
                    gen_z_chunk = gen_z.chunk(accum_iter)
                    gen_c_chunk = gen_c.chunk(accum_iter)
                    for real_img_acm, real_c_acm, gen_z_acm, gen_c_acm in zip(real_img_chunk, real_c_chunk, gen_z_chunk, gen_c_chunk):
                        loss.accumulate_gradients(
                            phase=phase.name,
                            real_img=real_img_acm,
                            real_c=real_c_acm,
                            gen_z=gen_z_acm,
                            gen_c=gen_c_acm,
                            accum_grad_gain=1 / accum_iter,
                            cur_nimg=cur_nimg,
                        )
                phase.module.requires_grad_(False)

                with torch.autograd.profiler.record_function(phase.name + "_opt"):
                    params = [param for param in phase.module.parameters() if param.numel() > 0 and param.grad is not None]
                    if len(params) > 0:
                        flat = torch.cat([param.grad.flatten() for param in params])
                        if num_gpus > 1:
                            torch.distributed.all_reduce(flat)
                            flat /= num_gpus
                        misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                        grads = flat.split([param.numel() for param in params])
                        for param, grad in zip(params, grads):
                            param.grad = grad.reshape(param.shape)
                    phase.opt.step()
                    po = phase.opt.state_dict()
                    training_stats.report(phase.name + "_opt_lr", phase.opt.param_groups[0]["lr"])

                # Phase done.
                if phase.end_event is not None:
                    phase.end_event.record(torch.cuda.current_stream(device))

            # Update G_ema.
            with torch.autograd.profiler.record_function("Gema"):
                ema_nimg = ema_kimg * 1000
                if ema_rampup is not None:
                    ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
                ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_beta))
                for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                    b_ema.copy_(b)
                G_ema.neural_rendering_resolution = G.neural_rendering_resolution
                G_ema.rendering_kwargs = G.rendering_kwargs.copy()

            # # Update state.
            cur_nimg += batch_size
            batch_idx += 1

            # --- within-tick progress (resets each tick; a tick ends with the
            # eval/export/snapshot maintenance below). One tick = kimg_per_tick*1000
            # images, so N batches/tick = that / batch_size. ---
            _tick_nimg = kimg_per_tick * 1000
            _done_nimg = cur_nimg - tick_start_nimg
            _n = _done_nimg // batch_size
            _N = max(1, _tick_nimg // batch_size)
            _now = time.time()
            _sit = _now - getattr(loss, "_iter_last_t", _now)
            loss._iter_last_t = _now
            _eta_min = _sit * max(0, _N - _n) / 60.0
            print(f"[tick {cur_tick}] batch {_n}/{_N} ({100.0 * _done_nimg / _tick_nimg:4.1f}%) | {_sit:5.2f}s/batch | ETA tick {_eta_min:5.1f}m")

            # Execute ADA heuristic.
            if (ada_stats is not None) and (batch_idx % ada_interval == 0):
                ada_stats.update()
                adjust = np.sign(ada_stats["Loss/signs/real"] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
                augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

            # Perform maintenance tasks once per tick.
            done = cur_nimg >= total_kimg * 1000

            if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
                continue

            # Print status line, accumulating the same information in training_stats.
            tick_end_time = time.time()
            fields = []
            fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
            fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
            fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
            fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
            fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
            fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
            fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
            fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
            fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
            torch.cuda.reset_peak_memory_stats()
            fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
            training_stats.report0("Timing/total_hours", (tick_end_time - start_time) / (60 * 60))
            training_stats.report0("Timing/total_days", (tick_end_time - start_time) / (24 * 60 * 60))
            if rank == 0:
                print(" ".join(fields))

            # Check for abort.
            if (not done) and (abort_fn is not None) and abort_fn():
                done = True
                if rank == 0:
                    print()
                    print("Aborting...")
            # image_snapshot_ticks=int(loss_kwargs['rlhf_opts']['image_snapshot_ticks'])

            image_snapshot_ticks = 1
            # network_snapshot_ticks = loss.hydra_cfg.train_tick_stop-1
            # network_snapshot_ticks = 5
            EVALUATE_METRICS = loss.hydra_cfg.EVALUATE_METRICS
            # Save image snapshot.
            if loss.hydra_cfg.export_first_images:
                first_tick_export = 0
            else:
                first_tick_export = 1
            seedslist_visualisation = hydra_cfg.seedslist_visualisation

            # exporting images
            if (
                (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0)
                # and not EVALUATE_METRICS
            ):  # change from cur_tick % image_snapshot_ticks == 0
                # -------------------------------------------------------------------

                # IN THIS ONE WE ARE EXPORTING THE 10 ORIGINAL SEEDS AND VISUALISING THEM.

                # -------------------------------------------------------------------

                loss.G.requires_grad_(False)
                loss.G.eval()
                G.requires_grad_(False)
                G.eval()

                G_ema.requires_grad_(False)
                G_ema.eval()

                with torch.no_grad():
                    export_images(grid_z, grid_c, G_ema, grid_size, cur_nimg, loss, run_dir)
                    export_seedslist_visualisation(loss, cur_tick, first_tick_export, seedslist_visualisation, G_ema, run_dir, cur_nimg)

            # Save network snapshot.
            snapshot_pkl = None
            snapshot_data = None
            network_snapshot_ticks = loss.hydra_cfg.network_snapshot_ticks  # save every 5 ticks, roughly 1 hour, (nb usually set to 50 in orig eg3d paper)
            if ((network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0) and cur_tick >= 0) or cur_tick >= train_tick_stop:
                snapshot_data, snapshot_pkl = save_network_snapshot(training_set_kwargs, G, D, G_ema, augment_pipe, run_dir, cur_nimg, rank, num_gpus)

                if EVALUATE_METRICS:
                    if rank == 0:
                        print(run_dir)
                        print("Evaluating metrics...")  # fid5k_partial
                    for metric in metrics:
                        # for metric in ['kid50k_full','kid50k','fid50k']:
                        result_dict = metric_main.calc_metric(
                            metric=metric,
                            G=snapshot_data["G_ema"],
                            dataset_kwargs=training_set_kwargs,
                            num_gpus=num_gpus,
                            rank=rank,
                            device=device,
                        )
                        if rank == 0:
                            metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                        stats_metrics.update(result_dict.results)

                    out_save_fid = os.path.join(run_dir, "fid_kid_calc.json")
                    with open(out_save_fid, "w") as f:
                        json.dump(stats_metrics, f)

                    print(f"saved json dict to : {out_save_fid}")

                del snapshot_data  # conserve memory

            if loss.hydra_cfg.rlhf_tune_hpms.update_G_ema_every_tick:
                old_G_ema = copy.deepcopy(G_ema).eval().to("cpu")
                loss.old_G_ema = old_G_ema

            if loss.hydra_cfg.plot_rwd_dist and loss.hydra_cfg.init_seeds_first:
                plot_rwd_dist_top_bottom(loss, cur_tick, G_ema, run_dir, cur_nimg=cur_nimg)

                run_combined_epoch_analysis(run_dir)

            # Depth Map Reglarisation
            tick_start_nimg = cur_nimg
            tick_start_time = time.time()
            maintenance_time = tick_start_time - tick_end_time

            if cur_tick >= train_tick_stop:
                done = True  # early stop it

                snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
                for name, module in [
                    ("G", G),
                    ("D", D),
                    ("G_ema", G_ema),
                    ("augment_pipe", augment_pipe),
                ]:
                    if module is not None:
                        if num_gpus > 1:
                            misc.check_ddp_consistency(module, ignore_regex=r".*\.[^.]+_(avg|ema)")
                        module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                    snapshot_data[name] = module
                    del module  # conserve memory

                snapshot_pkl = os.path.join(run_dir, f"network-snapshot-{cur_nimg // 1000:06d}_LAST.pkl")
                if rank == 0:
                    with open(snapshot_pkl, "wb") as f:
                        pickle.dump(snapshot_data, f)

                # End-of-training mesh visualisation. This is heavy (mesh
                # extraction + pyvista render), so it is gated behind a flag.
                # Smoke runs set render_final_vis=false to stop cleanly at the
                # final tick without the expensive viz.
                if loss.hydra_cfg.get("render_final_vis", True):
                    from training.render_final_snapshot_vis import render_final_snapshot_vis

                    print("\n\nrendering img for vis...")
                    render_final_snapshot_vis(snapshot_pkl)
                    print("rndr complete.")

            if done:
                break

            # Update state.
            cur_tick += 1

    # output 100 meshes

    # /media/krillman/240GB_DATA/training_runs_2/00180-ffhq-FFHQ_512_4995-gpus1-batch4-gamma25/network-snapshot-002033.pkl
    # train_tick_stop

    if loss.hydra_cfg.get("eval_metrics_only", False):
        snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
        for name, module in [
            ("G", G),
            ("D", D),
            ("G_ema", G_ema),
            ("augment_pipe", augment_pipe),
        ]:
            if module is not None:
                if num_gpus > 1:
                    misc.check_ddp_consistency(module, ignore_regex=r".*\.[^.]+_(avg|ema)")
                module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
            snapshot_data[name] = module
            del module  # conserve memory

        snapshot_pkl = os.path.join(run_dir, f"network-snapshot-{cur_nimg // 1000:06d}.pkl")
        if rank == 0:
            with open(snapshot_pkl, "wb") as f:
                pickle.dump(snapshot_data, f)

        if rank == 0:
            print(run_dir)
            print("Evaluating metrics...")
        for metric in metrics:
            result_dict = metric_main.calc_metric(
                metric=metric,
                G=snapshot_data["G_ema"],
                dataset_kwargs=training_set_kwargs,
                num_gpus=num_gpus,
                rank=rank,
                device=device,
            )
            if rank == 0:
                metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
            stats_metrics.update(result_dict.results)

        out_save_fid = os.path.join(run_dir, "fid_kid_calc.json")
        with open(out_save_fid, "w") as f:
            json.dump(stats_metrics, f)

        print(f"saved json dict to : {out_save_fid}")
        if loss.hydra_cfg.get("output_mesh_images_only", False):
            dummy = 1
        else:
            print("exiting...")
            sys.exit()

    if loss.hydra_cfg.get("output_mesh_images_only", False):
        seeds_for_inference = [200000 + k for k in range(loss.hydra_cfg.n_output_for_preference_study)]

        # if cur_tick>0:# and hydra_cfg.export_first_images:
        print("pausing here")

        views = [-18, 18]

        rdir = loss.hydra_cfg.click_legacy_args.resume

        rdir = os.path.dirname(rdir)

        newdir = os.path.join(rdir, "tuned_meshes")

        os.makedirs(newdir, exist_ok=True)

        visualise_mesh_using_pyvista(
            seeds=seeds_for_inference,
            G=G_ema.cuda(),
            run_dir=newdir,
            shape_res=512,
            epoch=0,
            delete_ply=False,
            window_size=4096,
            cl_frac=1.0,
            views=views,
            specular=0.35,
            scale_factor=0.5,
            outputting_bigmesh=True,
        )  # cl frac increase=higher resolution and detail
        image_files = glob.glob(os.path.join(newdir, "seed_*_epoch_0_mesh.png"))

        # sort images by time created
        image_files.sort(key=os.path.getmtime)
        # stacked_image=stack_snapshot_images_fn(image_files)
        # stacked_im_fn=os.path.join(run_dir,f'stacked_im_cur_nimg_{cur_nimg//1000:06d}.jpg')
        # stacked_image.save(stacked_im_fn)
        # for i in image_files:
        #    os.remove(i)

        print("exiting...")
        sys.exit()

    # Done.
    if rank == 0:
        print()
        print("Exiting...")

        if _wandb_enabled(loss):
            wandb.finish()


# ----------------------------------------------------------------------------
