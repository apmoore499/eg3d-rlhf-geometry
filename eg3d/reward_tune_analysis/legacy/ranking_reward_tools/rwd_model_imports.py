import glob
import json
from collections import Counter

import autoroot  # noqa: F401

# checking all unique combinations of flattened depth maps
import numpy as np
import torch
from torch import Tensor

# --------------------------------------------------------------------------------------------
# Load necessary Pytorch packages
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import matplotlib
import torch
import torch.nn as nn
import torchvision

# plotting reward dist

matplotlib.use("Agg")

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from pandas_ods_reader import read_ods
from pyexcel import get_book
from tqdm import tqdm

import pickle

RLHF_DIR = "/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/000_RLHF_AM"
RLHF_DIR = RLHF_DIR.replace("##", "000")
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
REWARD_MODEL_TRAINING_DIR = PROJECT_ROOT / "reward_model_training"
PRECOMPUTED_DIR = REWARD_MODEL_TRAINING_DIR / "precomputed"


def _precomputed_path(filename):
    return str(PRECOMPUTED_DIR / filename)


import torch_geometric.transforms as geom_T

# from torch_geometric.datasets import TUDataset

face_pcd_transform = geom_T.Compose([geom_T.NormalizeScale(), geom_T.SamplePoints(2048, remove_faces=True)])


import math
import numbers
from itertools import repeat

import torch
from torch_geometric.data import Data

# pos=torch.randn(10,3).cuda()

# expecting points shape as N, dim

ffhq_rendering_options = {
    "depth_resolution": 48,  # number of uniform samples to take per ray.
    "depth_resolution_importance": 48,  # number of importance samples to take per ray.
    "ray_start": 2.25,  # near point along each ray to start taking samples.
    "ray_end": 3.3,  # far point along each ray to stop taking samples.
    "box_warp": 1,  # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
    "avg_camera_radius": 2.7,  # used only in the visualizer to specify camera orbit radius.
    "avg_camera_pivot": [
        0,
        0,
        0.2,
    ],  # used only in the visualizer to control center of camera rotation.
}


from create_binary_dset_torch_10_08_2023 import *
from rwd_model_definitions import *
from torch.utils.tensorboard import SummaryWriter


class dargs:
    def __init__(self):
        return None


def model_evaluate_with_plot(model_dir, reward_model, test_loader, train_loader, val_loader, dargs):
    state_dict = torch.load(os.path.join(model_dir, "model_state_dict_best.pth"))
    reward_model.load_state_dict(state_dict)
    reward_model.eval()
    print("here are your resulte")
    fig, normalisation_terms = evaluate_model(reward_model, test_loader, train_loader, val_loader, dargs)
    return fig, normalisation_terms, reward_model


def get_batch_format_func(model_class):
    if model_class == "rwd_model_stylecode":
        return predict_style_code_with_vals

    if model_class == "rwd_model_2d_landmarks_98":
        return predict_ldmk_2d_with_vals

    if model_class == "rwd_model_2d_landmarks_98_triple":
        return predict_ldmk_2d_triple_with_vals

    if model_class == "rl_decoder_three_inet":
        return predict_three_dmap_w_vals

    if model_class == "rwd_model_3dmap_vgg_minimal":
        return predict_three_dmap_w_vals_minimal_vgg_model

    if model_class == "rwd_model_3dmap_vggface2_minimal":
        return predict_three_dmap_w_vals_minimal_vggface2_model

    if model_class == "rwd_model_pointnet2":
        return predict_pcd

    if model_class == "rwd_model_pointnet2_gfeature":
        return predict_pcd

    if model_class == "rwd_model_pointnet2_global":
        return predict_pcd_from_pairs_split_posneg

    # if model_class=='rwd_model_pointnet2_global':
    #    return(predict_pcd_from_pairs_combined)


# for contrastive loss, no transforms applied...
class dset_pcd_for_closs(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, all_seeds, dtype, ddir_func, transforms=None):
        super().__init__()
        "Initialization"
        self.all_seeds_in_batch = all_seeds  # this is the list of ordered combo for the current batch. precomputed.
        self.dtype = dtype
        self.ddir_func = ddir_func
        # self.device=torch.device('cuda')
        self.transforms = None
        if transforms is not None:
            self.transforms = transforms
            # point_samples_per_pc_batch

        self.set_sorted_unique_seeds()
        self.n_point_samples_per_pcd_batch = 4096

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.all_seeds_in_batch)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        seed = self.all_seeds_in_batch[index]
        fn = create_pt_fn(ddir=self.ddir_func(s), ot=o, seed=seed)  # for s in seeds_in_batch]
        pcd = torch.load(fn, map_location=torch.device("cpu"))
        # pcd=downsample_pcd_points(data,n_points=1000000) #self.n_point_samples_per_pcd_batch) #for f in files] #random sample 2048
        return pcd
        # batch_len=len(ordered_batch[ordered_batch!=-1])
        # padded_vals_len=len(ordered_batch)-batch_len
        # seeds_in_batch=torch.tensor(ordered_batch[:batch_len].astype(int))#.unique()

        # ----------------------------------------------------------------------


def rescale_im_dmp_for_lmk(dmap):
    rmin = 2.25
    rmax = 3.3
    dm_min = -1.0
    dmap = (((dmap - rmin) / (rmax - rmin)) * 2) - 1
    dmap[dmap < dm_min] = dm_min
    dmap[dmap > 1.0] = 1.0
    return dmap


import cv2


def return_lmks_mask(s):
    fn_depth = create_pt_fn(ddir=ddir_func(s), ot="triple_dmap", seed=s)  # for s in [seed]][0]
    tdm = torch.load(fn_depth)[1].unsqueeze(0)
    tdm = torch.nn.functional.interpolate(tdm, size=(256, 256)).squeeze()
    lmks = assemble_single_lmks(ddir=ddir_func(s), seed=s)  # @256,256 resolution
    lmks = lmks.cpu().numpy().squeeze().astype(np.int32)
    dmp = tdm.squeeze(0)[:, :, None].expand(256, 256, 3).cpu().numpy()

    dmp = rescale_im_dmp_for_lmk(dmp)
    dmp = dmp / 2 + 0.5
    dmp = (dmp * 255).astype(np.uint8)
    ocvim = cv2.cvtColor(dmp, cv2.COLOR_RGB2BGR)
    radius = 9

    for l in lmks:
        x, y = l
        ocvim = cv2.circle(ocvim, (x, y), radius=radius, color=(0, 0, 255), thickness=-1)
    mask = cv2.cvtColor(ocvim, cv2.COLOR_BGR2RGB)

    retmask = np.empty_like(dmp).astype(np.bool_)
    retmask.fill(False)
    retmask[mask == [0, 0, 255]] = True

    return retmask


# converts the image to a point cloud given some depth values
def imd_to_xyz(image_depth, ray_origins, ray_directions, neural_rendering_resolution, radius_cutoff=10.0):
    final_dim = neural_rendering_resolution * neural_rendering_resolution
    imd_list = image_depth.reshape(final_dim)
    imd_greater = torch.where(imd_list <= radius_cutoff)
    final_dim = neural_rendering_resolution * neural_rendering_resolution
    imd = image_depth.reshape(1, final_dim).unsqueeze(2).expand(1, final_dim, 3)  # .cuda()
    retval = ray_origins + imd * ray_directions
    return (retval, imd_greater)


import sys

sys.path.append("/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/eg3d")
from training.volumetric_rendering.ray_sampler import RaySampler


class dset_single_stream_ordered_minimal(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, all_combined_rankings, dtype, ddir_func, for_CL=False, transforms=None):
        super().__init__()
        "Initialization"
        #!!!! MUST BE ORDERED OR METHOD WILL FAIL!!!!#
        self.all_combined_rankings_ordered = all_combined_rankings  # this is the list of ordered combo for the current batch. precomputed.
        self.dtype = dtype
        self.ddir_func = ddir_func
        # self.device=torch.device('cuda')
        self.transforms = None
        if transforms is not None:
            self.transforms = transforms
            # point_samples_per_pc_batch

        # self.set_sorted_unique_seeds()
        self.n_point_samples_per_pcd_batch = 8192

        self.ray_sampler_static = RaySampler()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.all_combined_rankings_ordered)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        ordered_batch = self.all_combined_rankings_ordered[index]
        batch_len = len(ordered_batch[ordered_batch != -1])
        padded_vals_len = len(ordered_batch) - batch_len
        seeds_in_batch = torch.tensor(ordered_batch[:batch_len].astype(int))  # .unique()

        # tts=torch.argsort(seeds_in_batch)
        o = self.dtype

        if o == "triple_rgb_lmks_98":
            files = [assemble_triple_lmks(ddir=self.ddir_func(s), seed=s) for s in seeds_in_batch]

        elif o == "canonical_rgb_lmks_98":
            files = [assemble_single_lmks(ddir=self.ddir_func(s), seed=s) for s in seeds_in_batch]

        elif o == "triple_rgb":
            files = [assemble_triple_rgb(ddir=self.ddir_func(s), seed=s) for s in seeds_in_batch]

        elif o == "canonical_rgb":
            files = [assemble_single_rgb(ddir=self.ddir_func(s), seed=s) for s in seeds_in_batch]

        elif "triple_dmap" in o:
            fns = [create_pt_fn(ddir=self.ddir_func(s), ot=o, seed=s) for s in seeds_in_batch]
            files = [torch.load(f) for f in fns]
            files = [upsample_normalise(f) for f in files]

        else:
            # fns=[create_pt_fn(ddir=self.ddir_func(s),ot=o,seed=s) for s in seeds_in_batch] #we want to crop landmarks

            files = [self.crop_pcd_to_lmks(s) for s in seeds_in_batch]
            # files=[torch.load(f,map_location=torch.device('cpu')) for f in fns]
            files = [downsample_pcd_points(f, n_points=self.n_point_samples_per_pcd_batch) for f in files]  # random sample 2048
            files = [center_points(f) for f in files]  # rescale normalise them...
            # files=[rotate_points_3d_random(f,degrees_yaw=0,degrees_pitch=0) for f in files]
            files = [jitter_points_uniform(f) for f in files]
            files = [center_points(f) for f in files]  # rescale normalise them...
            # files=[random_scale_points_along_axes(f) for f in files] #rescale normalise them...
            files = [mean_scale_pts(f) for f in files]  # rescale normalise them...
            # files=[random_translate_points(f) for f in files] #rescale normalise them...
            files = [f.permute(1, 0).unsqueeze(0) for f in files]
            # pcd=torch.load(fn,map_location=torch.device('cpu'))
        # pcd=downsample_pcd_points(pcd,n_points=4096)
        # pcd=center_points(pcd)
        # pcd=rotate_points_3d_random(pcd,degrees_yaw=60,degrees_pitch=0)
        # pcd=jitter_points_uniform(pcd)

        # pcd=random_translate_points(pcd)
        # pcd=random_scale_points_along_axes(pcd)
        # pcd=mean_scale_pts(pcd)

        extra_pad = torch.zeros_like(files[0])
        padlist = [extra_pad for k in range(padded_vals_len)]
        files += padlist

        return o, dict(files=files, batch_len=batch_len)

    def crop_pcd_to_lmks(self, seed):
        nrs = 256
        fn_depth = create_pt_fn(ddir=self.ddir_func(seed), ot="triple_dmap", seed=seed)

        tdm = torch.load(fn_depth, map_location=torch.device("cpu"))[1].unsqueeze(0)
        tdm = torch.nn.functional.interpolate(tdm, size=(256, 256)).squeeze(0, 1)

        tdmap_cams = torch.load(
            os.path.join(os.environ["STATIC_CONFIGS_DIR"], "triple_dmap_cameras.pt"),
            map_location=torch.device("cpu"),
        )
        canon_cam = tdmap_cams[1].unsqueeze(0)
        c = canon_cam  # .cuda()
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        ray_origins, ray_directions = self.ray_sampler_static(cam2world_matrix, intrinsics, nrs)
        dd, imd = imd_to_xyz(
            image_depth=tdm,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            neural_rendering_resolution=nrs,
            radius_cutoff=10.0,
        )
        retmask = return_lmks_mask(seed)
        retmask = retmask[:, :, 1].flatten()
        dd = dd[:, retmask, :].reshape(-1, 3)
        dd_idx = torch.randperm(dd.shape[0])[:8192]
        ptc = dd[dd_idx]

        return ptc

    def get_sorted_unique_seeds(self):
        all_batch_seeds = [self.all_combined_rankings_ordered[i] for i in range(len(self))]
        all_batch_seeds = np.concatenate(all_batch_seeds)
        all_unique_seeds = np.unique(all_batch_seeds).astype(np.int32)
        all_unique_seeds = all_unique_seeds[all_unique_seeds != -1]
        all_unique_seeds = np.sort(all_unique_seeds, axis=None)
        return all_unique_seeds

    # def set_sorted_unique_seeds(self):
    #     seeds=self.get_sorted_unique_seeds()
    #     self.

    def return_all_indiv_examples(self):
        o = self.dtype
        fns = [create_pt_fn(ddir=self.ddir_func(s), ot=o, seed=s) for s in self.get_sorted_unique_seeds()]
        files = [torch.load(f, map_location=torch.device("cpu")) for f in fns]

        files = [downsample_pcd_points(f, n_points=self.n_point_samples_per_pcd_batch) for f in files]  # random sample 2048
        files = [center_points(f) for f in files]  # rescale normalise them...
        files = [rotate_points_3d_random(f, degrees_yaw=0, degrees_pitch=0) for f in files]
        files = [jitter_points_uniform(f) for f in files]
        files = [center_points(f) for f in files]  # rescale normalise them...
        files = [random_scale_points_along_axes(f) for f in files]  # rescale normalise them...
        files = [mean_scale_pts(f) for f in files]  # rescale normalise them...
        files = [random_translate_points(f) for f in files]  # rescale normalise them...
        files = [f.permute(1, 0).unsqueeze(0) for f in files]

        return files

    def return_all_indiv_examples_notransform(self):
        o = self.dtype
        fns = [create_pt_fn(ddir=self.ddir_func(s), ot=o, seed=s) for s in self.get_sorted_unique_seeds()]
        files = [torch.load(f, map_location=torch.device("cpu")) for f in fns]

        files = [downsample_pcd_points(f, n_points=self.n_point_samples_per_pcd_batch) for f in files]  # random sample 2048
        files = [center_points(f) for f in files]  # rescale normalise them...
        # files=[rotate_points_3d_random(f,degrees_yaw=60,degrees_pitch=15) for f in files]
        # files=[jitter_points_uniform(f) for f in files]
        # files=[center_points(f) for f in files] #rescale normalise them...
        # files=[random_scale_points_along_axes(f) for f in files] #rescale normalise them...
        files = [mean_scale_pts(f) for f in files]  # rescale normalise them...
        # files=[random_translate_points(f) for f in files] #rescale normalise them...
        files = [f.permute(1, 0).unsqueeze(0) for f in files]

        return files

    def return_single_example_by_seed(self, seed):
        # assume seed is a singl enumber
        o = self.dtype
        fns = [create_pt_fn(ddir=self.ddir_func(s), ot=o, seed=s) for s in [seed]]
        files = [torch.load(f, map_location=torch.device("cpu")) for f in fns]
        files = [downsample_pcd_points(f, n_points=self.n_point_samples_per_pcd_batch) for f in files]  # random sample 2048
        files = [center_points(f) for f in files]  # rescale normalise them...
        files = [rotate_points_3d_random(f, degrees_yaw=0, degrees_pitch=0) for f in files]
        files = [jitter_points_uniform(f) for f in files]
        files = [center_points(f) for f in files]  # rescale normalise them...
        files = [random_scale_points_along_axes(f) for f in files]  # rescale normalise them...
        files = [mean_scale_pts(f) for f in files]  # rescale normalise them...
        files = [random_translate_points(f) for f in files]  # rescale normalise them...
        files = [f.permute(1, 0).unsqueeze(0) for f in files]
        # files=[f.permute(1,0).unsqueeze(0) for f in files]
        return files[0]

    def load_single_example_no_transform(self, seed):
        fns = create_pt_fn(ddir=self.ddir_func(seed), ot=self.dtype, seed=seed)
        file = torch.load(fns, map_location=torch.device("cpu"))
        return file


import torch


# allows to set transform and maintain to enable contrastive training
class ensemble_pointcloud_transforms(nn.Module):
    def __init__(self):
        super().__init__()

        self.translation_dist = 0.2
        self.random_scale_margins = 0.05
        self.degrees_pitch_range = 15
        self.degrees_yaw_range = 60
        self.jitter_range = 0.001
        self.n_points = 4096

        self.resample_transform_parameters()

    def reset_random_domains_for_train(self):
        self.translation_dist = 0.2
        self.random_scale_margins = 0.05
        self.degrees_pitch_range = 15
        self.degrees_yaw_range = 60
        self.jitter_range = 0.001
        # self.n_points=4096

        self.resample_transform_parameters()

        return self

    def set_no_random_for_validation(self):
        self.translation_dist = 0.0
        self.random_scale_margins = 0.00
        self.degrees_pitch_range = 0
        self.degrees_yaw_range = 0
        self.jitter_range = 0.0
        # self.n_points=4096

        self.resample_transform_parameters()

        return self

    def apply_transforms(self, in_pcd):
        # pcd has shape [B,N,3] where B is batch size, N is number of points in the training example ~30000 points usually

        out_pcd = self.downsample_pcd_points(in_pcd)
        out_pcd = self.center_points(out_pcd)
        out_pcd = self.rotate_points_3d_random(out_pcd)
        out_pcd = self.jitter_points_uniform(out_pcd)
        out_pcd = self.center_points(out_pcd)
        out_pcd = self.random_scale_points_along_axes(out_pcd)
        out_pcd = self.mean_scale_pts(out_pcd)
        out_pcd = self.random_translate_points(out_pcd)

        return out_pcd

        #     files=[downsample_pcd_points(f,n_points=self.n_point_samples_per_pcd_batch) for f in files] #random sample 2048
        # files=[center_points(f) for f in files] #rescale normalise them...
        # files=[rotate_points_3d_random(f,degrees_yaw=0,degrees_pitch=0) for f in files]
        # files=[jitter_points_uniform(f) for f in files]
        # files=[center_points(f) for f in files] #rescale normalise them...
        # files=[random_scale_points_along_axes(f) for f in files] #rescale normalise them...
        # files=[mean_scale_pts(f) for f in files] #rescale normalise them...
        # files=[random_translate_points(f) for f in files] #rescale normalise them...

    def resample_transform_parameters(self):
        self.translation_offset = torch.empty(1, 3).uniform_(-self.translation_dist, self.translation_dist)  # .expand(ttl.shape)
        self.scale_rndm = torch.empty(1, 3).uniform_(1 - self.random_scale_margins, 1 + self.random_scale_margins)
        # self.jitter_offset=torch.empty(1,3).uniform(-self.jitter_range,self.jitter_range)

        dim = 3

        ts = []
        for d in range(dim):
            ts.append(torch.empty((self.n_points,)).uniform_(-abs(self.jitter_range), abs(self.jitter_range)))

        self.jitter_offset = torch.stack(ts, dim=-1)

        # rotate first axis
        degrees_pitch = (-abs(self.degrees_pitch_range), abs(self.degrees_pitch_range))
        self.degrees_pitch = math.pi * random.uniform(*degrees_pitch) / 180.0

        # rotate second axis
        degrees = (-abs(self.degrees_yaw_range), abs(self.degrees_yaw_range))
        self.degrees_yaw = math.pi * random.uniform(*degrees) / 180.0
        # if fixed_degrees is not None:
        #    degree = math.pi * fixed_degrees[1]/180.0

        return self

    def mean_scale_pts(self, ttl):
        ttl_c = ttl - ttl.mean(dim=0, keepdim=True)
        scale = (1 / ttl_c.abs().max()) * 0.999999
        ttl_c = ttl_c * scale
        return ttl_c

    def center_points(self, ttl):
        ttl_c = ttl - ttl.mean(dim=0, keepdim=True)
        return ttl_c

    def downsample_pcd_points(self, ttl):
        perm = torch.randperm(ttl.size(0))
        idx = perm[: min(ttl.size(0), self.n_points)]
        samples = ttl[idx]
        return samples

    def random_translate_points(self, ttl, dist=0.2):
        translation = self.translation_offset.expand(ttl.shape)
        ttl_c = ttl + translation
        return ttl_c

    def random_scale_points_along_axes(self, ttl, margins=0.05):
        translation = self.scale_rndm.expand(ttl.shape)
        ttl_c = ttl * translation
        return ttl_c

    def jitter_points_uniform(self, ttl):
        # translation=self.translation_offset.expand(ttl.shape)
        ttl_c = ttl + self.jitter_offset.view_as(ttl)
        return ttl_c

    def rotate_points_3d_random(self, points, degrees_yaw=15, degrees_pitch=90, fixed_degrees=None):
        # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/random_rotate.html#RandomRotate
        # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/linear_transformation.html#LinearTransformation
        # rip from the pytorch_geometric code................
        pos = points

        pd = pos.get_device()
        if pd == -1:
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{pd}")
        orig_shape = pos.shape
        assert len(pos.shape) == 2, "error u need the 2 dim positions for rotation thing"

        if pos.shape[-1] != 3:
            pos = pos.permute(1, 0)

        # #rotate first axis
        # degrees = (-abs(degrees_pitch), abs(degrees_pitch))
        # degree = math.pi * random.uniform(*degrees) / 180.0

        # if fixed_degrees is not None:
        #     degree = math.pi * fixed_degrees[0]/180.0

        sin, cos = math.sin(self.degrees_pitch), math.cos(self.degrees_pitch)
        matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        matrix = torch.tensor(matrix, device=device)
        pos = pos @ matrix  # .to(pos.device, pos.dtype)

        # if fixed_degrees is not None:
        #     degree = math.pi * fixed_degrees[0]/180.0

        sin, cos = math.sin(self.degrees_yaw), math.cos(self.degrees_yaw)
        matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        matrix = torch.tensor(matrix, device=device)
        pos = pos @ matrix  # .to(pos.device, pos.dtype)

        # #rotate third axis
        # degrees = (-abs(degrees_range), abs(degrees_range))
        # degree = math.pi * random.uniform(*degrees) / 180.0
        # if fixed_degrees is not None:
        #     degree = math.pi * fixed_degrees[2]/180.0

        # sin, cos = math.sin(degree), math.cos(degree)
        # matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        # matrix = torch.tensor(matrix,device=device)
        # pos=pos @ matrix#.to(pos.device, pos.dtype)

        return pos.reshape(orig_shape)

    # rip from random jitter on the pytorch geom lib

    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/random_jitter.html

    # #expecting points shape as N, dim
    # def jitter_points_uniform(self,points,size=0.001):
    #     pos=points
    #     orig_shape=pos.shape

    #     assert len(pos.shape)==2,'error u need the 2 dim positions for rotation thing'
    #     if pos.shape[-1]!=3:
    #         pos=pos.permute(1,0)

    #     (n, dim), t = pos.size(), size
    #     if isinstance(t, numbers.Number):
    #         t = list(repeat(t, times=dim))
    #     assert len(t) == dim

    #     ts = []
    #     for d in range(dim):
    #         ts.append(torch.empty_like(pos[:,0]).uniform_(-abs(t[d]), abs(t[d])))

    #     pos = pos + torch.stack(ts, dim=-1)
    #     return(pos.reshape(orig_shape))


# -------------------------------------------------------------------------


# class dset_single_stream_ordered_minimal(torch.utils.data.Dataset):
#     'Characterizes a dataset for PyTorch'
#     def __init__(self, all_combined_rankings,dtype,ddir_func,for_CL=False,transforms=None):
#         super().__init__()
#         'Initialization'
#         #!!!! MUST BE ORDERED OR METHOD WILL FAIL!!!!#
#         self.all_combined_rankings_ordered = all_combined_rankings #this is the list of ordered combo for the current batch. precomputed.
#         self.dtype=dtype
#         self.ddir_func=ddir_func
#         #self.device=torch.device('cuda')
#         self.transforms=None
#         if transforms is not None:
#             self.transforms=transforms
#             #point_samples_per_pc_batch

#         self.set_sorted_unique_seeds()
#         self.n_point_samples_per_pcd_batch=2048

#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.all_combined_rankings_ordered)

#     def __getitem__(self, index):
#         'Generates one sample of data'
#         # Select sample
#         ordered_batch = self.all_combined_rankings_ordered[index]
#         batch_len=len(ordered_batch[ordered_batch!=-1])
#         padded_vals_len=len(ordered_batch)-batch_len
#         seeds_in_batch=torch.tensor(ordered_batch[:batch_len].astype(int))#.unique()

#         #tts=torch.argsort(seeds_in_batch)
#         o=self.dtype

#         if o=='triple_rgb_lmks_98':
#             files=[assemble_triple_lmks(ddir=self.ddir_func(s),seed=s) for s in seeds_in_batch]

#         elif o=='canonical_rgb_lmks_98':
#             files=[assemble_single_lmks(ddir=self.ddir_func(s),seed=s) for s in seeds_in_batch]

#         elif o=='triple_rgb':
#             files=[assemble_triple_rgb(ddir=self.ddir_func(s),seed=s) for s in seeds_in_batch]

#         elif o=='canonical_rgb':
#             files=[assemble_single_rgb(ddir=self.ddir_func(s),seed=s) for s in seeds_in_batch]

#         elif 'triple_dmap' in o:
#             fns=[create_pt_fn(ddir=self.ddir_func(s),ot=o,seed=s) for s in seeds_in_batch]
#             files=[torch.load(f) for f in fns]
#             files=[upsample_normalise(f) for f in files]

#         else:
#             fns=[create_pt_fn(ddir=self.ddir_func(s),ot=o,seed=s) for s in seeds_in_batch]
#             files=[torch.load(f,map_location=torch.device('cpu')) for f in fns]
#             files=[downsample_pcd_points(f,n_points=self.n_point_samples_per_pcd_batch) for f in files] #random sample 2048
#             files=[center_points(f) for f in files] #rescale normalise them...
#             files=[rotate_points_3d_random(f,degrees_yaw=0,degrees_pitch=0) for f in files]
#             files=[jitter_points_uniform(f) for f in files]
#             files=[center_points(f) for f in files] #rescale normalise them...
#             files=[random_scale_points_along_axes(f) for f in files] #rescale normalise them...
#             files=[mean_scale_pts(f) for f in files] #rescale normalise them...
#             files=[random_translate_points(f) for f in files] #rescale normalise them...
#             files=[f.permute(1,0).unsqueeze(0) for f in files]
#             #pcd=torch.load(fn,map_location=torch.device('cpu'))
# # pcd=downsample_pcd_points(pcd,n_points=4096)
# # pcd=center_points(pcd)
# # pcd=rotate_points_3d_random(pcd,degrees_yaw=60,degrees_pitch=0)
# # pcd=jitter_points_uniform(pcd)

# # pcd=random_translate_points(pcd)
# # pcd=random_scale_points_along_axes(pcd)
# # pcd=mean_scale_pts(pcd)

#         extra_pad=torch.zeros_like(files[0])
#         padlist=[extra_pad for k in range(padded_vals_len)]
#         files+=padlist

#         return o,dict(files=files,batch_len=batch_len)


#     def get_sorted_unique_seeds(self):
#         all_batch_seeds  = [self.all_combined_rankings_ordered[i] for i in range(len(self))]
#         all_batch_seeds = np.concatenate(all_batch_seeds)
#         all_unique_seeds = np.unique(all_batch_seeds).astype(np.int32)
#         all_unique_seeds = all_unique_seeds[all_unique_seeds!=-1]
#         all_unique_seeds=np.sort(all_unique_seeds, axis=None)
#         return(all_unique_seeds)

#     def return_all_indiv_examples(self):

#         o=self.dtype
#         fns=[create_pt_fn(ddir=self.ddir_func(s),ot=o,seed=s) for s in self.get_sorted_unique_seeds()]
#         files=[torch.load(f,map_location=torch.device('cpu')) for f in fns]

#         files=[downsample_pcd_points(f,n_points=self.n_point_samples_per_pcd_batch) for f in files] #random sample 2048
#         files=[center_points(f) for f in files] #rescale normalise them...
#         files=[rotate_points_3d_random(f,degrees_yaw=0,degrees_pitch=0) for f in files]
#         files=[jitter_points_uniform(f) for f in files]
#         files=[center_points(f) for f in files] #rescale normalise them...
#         files=[random_scale_points_along_axes(f) for f in files] #rescale normalise them...
#         files=[mean_scale_pts(f) for f in files] #rescale normalise them...
#         files=[random_translate_points(f) for f in files] #rescale normalise them...
#         files=[f.permute(1,0).unsqueeze(0) for f in files]

#         return(files)

#     def return_all_indiv_examples_notransform(self):

#         o=self.dtype
#         fns=[create_pt_fn(ddir=self.ddir_func(s),ot=o,seed=s) for s in self.get_sorted_unique_seeds()]
#         files=[torch.load(f,map_location=torch.device('cpu')) for f in fns]

#         files=[downsample_pcd_points(f,n_points=self.n_point_samples_per_pcd_batch) for f in files] #random sample 2048
#         files=[center_points(f) for f in files] #rescale normalise them...
#         #files=[rotate_points_3d_random(f,degrees_yaw=60,degrees_pitch=15) for f in files]
#         #files=[jitter_points_uniform(f) for f in files]
#         #files=[center_points(f) for f in files] #rescale normalise them...
#         #files=[random_scale_points_along_axes(f) for f in files] #rescale normalise them...
#         files=[mean_scale_pts(f) for f in files] #rescale normalise them...
#         #files=[random_translate_points(f) for f in files] #rescale normalise them...
#         files=[f.permute(1,0).unsqueeze(0) for f in files]

#         return(files)


#     def return_single_example_by_seed(self,seed):
#         #assume seed is a singl enumber
#         o=self.dtype
#         fns=[create_pt_fn(ddir=self.ddir_func(s),ot=o,seed=s) for s in [seed]]
#         files=[torch.load(f,map_location=torch.device('cpu')) for f in fns]
#         files=[downsample_pcd_points(f,n_points=self.n_point_samples_per_pcd_batch) for f in files] #random sample 2048
#         files=[center_points(f) for f in files] #rescale normalise them...
#         files=[rotate_points_3d_random(f,degrees_yaw=0,degrees_pitch=0) for f in files]
#         files=[jitter_points_uniform(f) for f in files]
#         files=[center_points(f) for f in files] #rescale normalise them...
#         files=[random_scale_points_along_axes(f) for f in files] #rescale normalise them...
#         files=[mean_scale_pts(f) for f in files] #rescale normalise them...
#         files=[random_translate_points(f) for f in files] #rescale normalise them...
#         files=[f.permute(1,0).unsqueeze(0) for f in files]
#         #files=[f.permute(1,0).unsqueeze(0) for f in files]
#         return(files[0])

#     def load_single_example_no_transform(self,seed):
#         fns=create_pt_fn(ddir=self.ddir_func(seed),ot=self.dtype,seed=seed)
#         file=torch.load(fns,map_location=torch.device('cpu'))
#         return(file)


# allows to set transform and maintain to enable contrastive training
class ensemble_pointcloud_transforms(nn.Module):
    def __init__(self):
        super().__init__()

        self.translation_dist = 0.2
        self.random_scale_margins = 0.05
        self.degrees_pitch_range = 15
        self.degrees_yaw_range = 60
        self.jitter_range = 0.001
        self.n_points = 4096

        self.resample_transform_parameters()

    def reset_random_domains_for_train(self):
        self.translation_dist = 0.2
        self.random_scale_margins = 0.05
        self.degrees_pitch_range = 15
        self.degrees_yaw_range = 60
        self.jitter_range = 0.001
        # self.n_points=4096

        self.resample_transform_parameters()

        return self

    def set_no_random_for_validation(self):
        self.translation_dist = 0.0
        self.random_scale_margins = 0.00
        self.degrees_pitch_range = 0
        self.degrees_yaw_range = 0
        self.jitter_range = 0.0
        # self.n_points=4096

        self.resample_transform_parameters()

        return self

    def apply_transforms(self, in_pcd):
        # pcd has shape [B,N,3] where B is batch size, N is number of points in the training example ~30000 points usually

        out_pcd = self.downsample_pcd_points(in_pcd)
        out_pcd = self.center_points(out_pcd)
        out_pcd = self.rotate_points_3d_random(out_pcd)
        out_pcd = self.jitter_points_uniform(out_pcd)
        out_pcd = self.center_points(out_pcd)
        out_pcd = self.random_scale_points_along_axes(out_pcd)
        out_pcd = self.mean_scale_pts(out_pcd)
        out_pcd = self.random_translate_points(out_pcd)

        return out_pcd

        #     files=[downsample_pcd_points(f,n_points=self.n_point_samples_per_pcd_batch) for f in files] #random sample 2048
        # files=[center_points(f) for f in files] #rescale normalise them...
        # files=[rotate_points_3d_random(f,degrees_yaw=0,degrees_pitch=0) for f in files]
        # files=[jitter_points_uniform(f) for f in files]
        # files=[center_points(f) for f in files] #rescale normalise them...
        # files=[random_scale_points_along_axes(f) for f in files] #rescale normalise them...
        # files=[mean_scale_pts(f) for f in files] #rescale normalise them...
        # files=[random_translate_points(f) for f in files] #rescale normalise them...

    def resample_transform_parameters(self):
        self.translation_offset = torch.empty(1, 3).uniform_(-self.translation_dist, self.translation_dist)  # .expand(ttl.shape)
        self.scale_rndm = torch.empty(1, 3).uniform_(1 - self.random_scale_margins, 1 + self.random_scale_margins)
        # self.jitter_offset=torch.empty(1,3).uniform(-self.jitter_range,self.jitter_range)

        dim = 3

        ts = []
        for d in range(dim):
            ts.append(torch.empty((self.n_points,)).uniform_(-abs(self.jitter_range), abs(self.jitter_range)))

        self.jitter_offset = torch.stack(ts, dim=-1)

        # rotate first axis
        degrees_pitch = (-abs(self.degrees_pitch_range), abs(self.degrees_pitch_range))
        self.degrees_pitch = math.pi * random.uniform(*degrees_pitch) / 180.0

        # rotate second axis
        degrees = (-abs(self.degrees_yaw_range), abs(self.degrees_yaw_range))
        self.degrees_yaw = math.pi * random.uniform(*degrees) / 180.0
        # if fixed_degrees is not None:
        #    degree = math.pi * fixed_degrees[1]/180.0

        return self

    def mean_scale_pts(self, ttl):
        ttl_c = ttl - ttl.mean(dim=0, keepdim=True)
        scale = (1 / ttl_c.abs().max()) * 0.999999
        ttl_c = ttl_c * scale
        return ttl_c

    def center_points(self, ttl):
        ttl_c = ttl - ttl.mean(dim=0, keepdim=True)
        return ttl_c

    def downsample_pcd_points(self, ttl):
        perm = torch.randperm(ttl.size(0))
        idx = perm[: min(ttl.size(0), self.n_points)]
        samples = ttl[idx]
        return samples

    def random_translate_points(self, ttl, dist=0.2):
        translation = self.translation_offset.expand(ttl.shape)
        ttl_c = ttl + translation
        return ttl_c

    def random_scale_points_along_axes(self, ttl, margins=0.05):
        translation = self.scale_rndm.expand(ttl.shape)
        ttl_c = ttl * translation
        return ttl_c

    def jitter_points_uniform(self, ttl):
        # translation=self.translation_offset.expand(ttl.shape)
        ttl_c = ttl + self.jitter_offset.view_as(ttl)
        return ttl_c

    def rotate_points_3d_random(self, points, degrees_yaw=15, degrees_pitch=90, fixed_degrees=None):
        # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/random_rotate.html#RandomRotate
        # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/linear_transformation.html#LinearTransformation
        # rip from the pytorch_geometric code................
        pos = points

        pd = pos.get_device()
        if pd == -1:
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{pd}")
        orig_shape = pos.shape
        assert len(pos.shape) == 2, "error u need the 2 dim positions for rotation thing"

        if pos.shape[-1] != 3:
            pos = pos.permute(1, 0)

        # #rotate first axis
        # degrees = (-abs(degrees_pitch), abs(degrees_pitch))
        # degree = math.pi * random.uniform(*degrees) / 180.0

        # if fixed_degrees is not None:
        #     degree = math.pi * fixed_degrees[0]/180.0

        sin, cos = math.sin(self.degrees_pitch), math.cos(self.degrees_pitch)
        matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        matrix = torch.tensor(matrix, device=device)
        pos = pos @ matrix  # .to(pos.device, pos.dtype)

        # if fixed_degrees is not None:
        #     degree = math.pi * fixed_degrees[0]/180.0

        sin, cos = math.sin(self.degrees_yaw), math.cos(self.degrees_yaw)
        matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        matrix = torch.tensor(matrix, device=device)
        pos = pos @ matrix  # .to(pos.device, pos.dtype)

        # #rotate third axis
        # degrees = (-abs(degrees_range), abs(degrees_range))
        # degree = math.pi * random.uniform(*degrees) / 180.0
        # if fixed_degrees is not None:
        #     degree = math.pi * fixed_degrees[2]/180.0

        # sin, cos = math.sin(degree), math.cos(degree)
        # matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        # matrix = torch.tensor(matrix,device=device)
        # pos=pos @ matrix#.to(pos.device, pos.dtype)

        return pos.reshape(orig_shape)

    # rip from random jitter on the pytorch geom lib

    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/random_jitter.html

    # #expecting points shape as N, dim
    # def jitter_points_uniform(self,points,size=0.001):
    #     pos=points
    #     orig_shape=pos.shape

    #     assert len(pos.shape)==2,'error u need the 2 dim positions for rotation thing'
    #     if pos.shape[-1]!=3:
    #         pos=pos.permute(1,0)

    #     (n, dim), t = pos.size(), size
    #     if isinstance(t, numbers.Number):
    #         t = list(repeat(t, times=dim))
    #     assert len(t) == dim

    #     ts = []
    #     for d in range(dim):
    #         ts.append(torch.empty_like(pos[:,0]).uniform_(-abs(t[d]), abs(t[d])))

    #     pos = pos + torch.stack(ts, dim=-1)
    #     return(pos.reshape(orig_shape))


def mean_scale_pts(ttl):
    ttl_c = ttl - ttl.mean(dim=0, keepdim=True)
    scale = (1 / ttl_c.abs().max()) * 0.999999
    ttl_c = ttl_c * scale
    return ttl_c


def center_points(ttl):
    ttl_c = ttl - ttl.mean(dim=0, keepdim=True)
    return ttl_c


def downsample_pcd_points(ttl, n_points=5000):
    perm = torch.randperm(ttl.size(0))
    idx = perm[: min(ttl.size(0), n_points)]
    samples = ttl[idx]
    return samples


def random_translate_points(ttl, dist=0.2):
    translation = torch.empty(1, 3).uniform_(-dist, dist).expand(ttl.shape)
    ttl_c = ttl + translation
    return ttl_c


def random_scale_points_along_axes(ttl, margins=0.05):
    translation = torch.empty(1, 3).uniform_(1 - margins, 1 + margins).expand(ttl.shape)
    ttl_c = ttl * translation
    return ttl_c


# def sample_5000_random(ttl): #upsampling as per facenet paper, using 5000 instead of 2048

#     perm = torch.randperm(ttl.size(0))
#     idx = perm[:5000]
#     samples = ttl[idx]

#     return(samples)


def rotate_points_3d_random(points, degrees_yaw=15, degrees_pitch=90, fixed_degrees=None):
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/random_rotate.html#RandomRotate
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/linear_transformation.html#LinearTransformation
    # rip from the pytorch_geometric code................
    pos = points

    pd = pos.get_device()
    if pd == -1:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{pd}")
    orig_shape = pos.shape
    assert len(pos.shape) == 2, "error u need the 2 dim positions for rotation thing"

    if pos.shape[-1] != 3:
        pos = pos.permute(1, 0)

    # rotate first axis
    degrees = (-abs(degrees_pitch), abs(degrees_pitch))
    degree = math.pi * random.uniform(*degrees) / 180.0

    if fixed_degrees is not None:
        degree = math.pi * fixed_degrees[0] / 180.0

    sin, cos = math.sin(degree), math.cos(degree)
    matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
    matrix = torch.tensor(matrix, device=device)
    pos = pos @ matrix  # .to(pos.device, pos.dtype)

    if fixed_degrees is not None:
        degree = math.pi * fixed_degrees[0] / 180.0

    # rotate second axis
    degrees = (-abs(degrees_yaw), abs(degrees_yaw))
    degree = math.pi * random.uniform(*degrees) / 180.0
    if fixed_degrees is not None:
        degree = math.pi * fixed_degrees[1] / 180.0

    sin, cos = math.sin(degree), math.cos(degree)
    matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
    matrix = torch.tensor(matrix, device=device)
    pos = pos @ matrix  # .to(pos.device, pos.dtype)

    # #rotate third axis
    # degrees = (-abs(degrees_range), abs(degrees_range))
    # degree = math.pi * random.uniform(*degrees) / 180.0
    # if fixed_degrees is not None:
    #     degree = math.pi * fixed_degrees[2]/180.0

    # sin, cos = math.sin(degree), math.cos(degree)
    # matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
    # matrix = torch.tensor(matrix,device=device)
    # pos=pos @ matrix#.to(pos.device, pos.dtype)

    return pos.reshape(orig_shape)


# rip from random jitter on the pytorch geom lib

# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/random_jitter.html


# expecting points shape as N, dim
def jitter_points_uniform(points, size=0.001):
    pos = points
    orig_shape = pos.shape

    assert len(pos.shape) == 2, "error u need the 2 dim positions for rotation thing"
    if pos.shape[-1] != 3:
        pos = pos.permute(1, 0)

    (n, dim), t = pos.size(), size
    if isinstance(t, numbers.Number):
        t = list(repeat(t, times=dim))
    assert len(t) == dim

    ts = []
    for d in range(dim):
        ts.append(torch.empty_like(pos[:, 0]).uniform_(-abs(t[d]), abs(t[d])))

    pos = pos + torch.stack(ts, dim=-1)
    return pos.reshape(orig_shape)


# import torch
# import torch.nn.functional as F
# from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Dropout, BatchNorm1d
# from torch_geometric.nn import  PointNetConv, fps, radius, knn
# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn.inits import reset
# from torch_geometric.utils.num_nodes import maybe_num_nodes
# from torch_geometric.data.data import Data
# from torch_scatter import scatter_add, scatter_max
# import torch_cluster


import torch

#import torch_cluster
from torch.nn import BatchNorm1d, Dropout, ReLU
from torch.nn import Linear as Lin
from torch.nn import Sequential as Seq
from torch_geometric.data.data import Data
from torch_geometric.nn import PointNetConv, fps, knn, radius
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils.num_nodes import maybe_num_nodes

#from torch_scatter import scatter_add, scatter_max


class PointNet2SAModule(torch.nn.Module):
    def __init__(self, sample_radio, radius, max_num_neighbors, mlp):
        super().__init__()
        self.sample_ratio = sample_radio
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors
        self.point_conv = PointNetConv(mlp)

    def forward(self, data):
        x, pos, batch = data

        # Sample
        idx = fps(pos, batch, ratio=self.sample_ratio)

        # Group(Build graph)
        row, col = radius(pos, pos[idx], self.radius, batch, batch[idx], max_num_neighbors=self.max_num_neighbors)
        edge_index = torch.stack([col, row], dim=0)

        # Apply pointnet
        x1 = self.point_conv(x, (pos, pos[idx]), edge_index)
        pos1, batch1 = pos[idx], batch[idx]

        return x1, pos1, batch1


class PointNet2GlobalSAModule(torch.nn.Module):
    """One group with all input points, can be viewed as a simple PointNet module.

    It also return the only one output point(set as origin point).
    """

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, data):
        x, pos, batch = data
        if x is not None:
            x = torch.cat([x, pos], dim=1)
        x1 = self.mlp(x)

        x1 = scatter_max(x1, batch, dim=0)[0]  # (batch_size, C1)

        batch_size = x1.shape[0]
        pos1 = x1.new_zeros((batch_size, 3))  # set the output point as origin
        batch1 = torch.arange(batch_size).to(batch.device, batch.dtype)

        return x1, pos1, batch1


class PointConvFP(MessagePassing):
    """Core layer of Feature propagtaion module."""

    def __init__(self, mlp=None):
        super().__init__()
        self.mlp = mlp
        self.aggr = "add"
        self.flow = "source_to_target"

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp)

    def forward(self, x, pos, edge_index):
        r"""
        Args:
            x (tuple), (tensor, tensor) or (tensor, NoneType)
            pos (tuple): The node position matrix. Either given as
                tensor for use in general message passing or as tuple for use
                in message passing in bipartite graphs.
            edge_index (LongTensor): The edge indices.
        """
        # Do not pass (tensor, None) directly into propagate(), sice it will check each item's size() inside.
        x_tmp = x[0] if x[1] is None else x
        aggr_out = self.propagate(edge_index, x=x_tmp, pos=pos)

        #
        i, j = (0, 1) if self.flow == "target_to_source" else (1, 0)
        x_target, pos_target = x[i], pos[i]

        add = (
            [
                pos_target,
            ]
            if x_target is None
            else [x_target, pos_target]
        )
        aggr_out = torch.cat([aggr_out, *add], dim=1)

        if self.mlp is not None:
            aggr_out = self.mlp(aggr_out)

        return aggr_out

    def message(self, x_j, pos_j, pos_i, edge_index):
        """
        x_j: (E, in_channels)
        pos_j: (E, 3)
        pos_i: (E, 3)
        """
        dist = (pos_j - pos_i).pow(2).sum(dim=1).pow(0.5)
        dist = torch.max(dist, torch.Tensor([1e-10]).to(dist.device, dist.dtype))
        weight = 1.0 / dist  # (E,)

        row, col = edge_index
        index = col
        num_nodes = maybe_num_nodes(index, None)
        wsum = scatter_add(weight, col, dim=0, dim_size=num_nodes)[index] + 1e-16  # (E,)
        weight /= wsum

        return weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class PointNet2FPModule(torch.nn.Module):
    def __init__(self, knn_num, mlp):
        super().__init__()
        self.knn_num = knn_num
        self.point_conv = PointConvFP(mlp)

    def forward(self, in_layer_data, skip_layer_data):
        in_x, in_pos, in_batch = in_layer_data
        skip_x, skip_pos, skip_batch = skip_layer_data

        row, col = knn(in_pos, skip_pos, self.knn_num, in_batch, skip_batch)
        edge_index = torch.stack([col, row], dim=0)

        x1 = self.point_conv((in_x, skip_x), (in_pos, skip_pos), edge_index)
        pos1, batch1 = skip_pos, skip_batch

        return x1, pos1, batch1


def make_mlp(in_channels, mlp_channels, batch_norm=True):
    assert len(mlp_channels) >= 1
    layers = []

    for c in mlp_channels:
        layers += [Lin(in_channels, c)]
        if batch_norm:
            layers += [BatchNorm1d(c)]
        layers += [ReLU()]

        in_channels = c

    return Seq(*layers)


class rwd_model_pointnet2_gfeature(torch.nn.Module):
    """
    ref:
        - https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_part_seg.py
        - https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet++.py
    """

    # but modified to go global vec for the faces....
    def __init__(self, **kwargs):
        super().__init__()
        self.num_classes = 2  # leave not used anyway
        self.model_class = "rwd_model_pointnet"
        # SA1
        sa1_sample_ratio = 0.5
        sa1_radius = 0.2
        sa1_max_num_neighbours = 64
        sa1_mlp = make_mlp(3, [64, 64, 128])
        self.sa1_module = PointNet2SAModule(sa1_sample_ratio, sa1_radius, sa1_max_num_neighbours, sa1_mlp)

        # SA2
        sa2_sample_ratio = 0.25
        sa2_radius = 0.4
        sa2_max_num_neighbours = 64
        sa2_mlp = make_mlp(128 + 3, [128, 128, 256])
        self.sa2_module = PointNet2SAModule(sa2_sample_ratio, sa2_radius, sa2_max_num_neighbours, sa2_mlp)

        # SA3
        sa3_mlp = make_mlp(256 + 3, [256, 512, 1024])
        self.sa3_module = PointNet2GlobalSAModule(sa3_mlp)

        self.rwd_fc1 = torch.nn.Linear(1024, 256)

        self.rwd_fc2 = torch.nn.Linear(256, 32)

        self.rwd_fc3 = torch.nn.Linear(32, 1)

        self.relu = torch.nn.functional.relu

        ##
        knn_num = 3

        # FP3, reverse of sa3
        fp3_knn_num = 1  # After global sa module, there is only one point in point cloud
        fp3_mlp = make_mlp(1024 + 256 + 3, [256, 256])
        self.fp3_module = PointNet2FPModule(fp3_knn_num, fp3_mlp)

        # FP2, reverse of sa2
        fp2_knn_num = knn_num
        fp2_mlp = make_mlp(256 + 128 + 3, [256, 128])
        self.fp2_module = PointNet2FPModule(fp2_knn_num, fp2_mlp)

        # FP1, reverse of sa1
        fp1_knn_num = knn_num
        fp1_mlp = make_mlp(128 + 3, [128, 128, 128])
        self.fp1_module = PointNet2FPModule(fp1_knn_num, fp1_mlp)

        self.fc1 = Lin(128, 128)
        self.dropout1 = Dropout(p=0.5)
        self.fc2 = Lin(128, self.num_classes)

        self.norm_offset = nn.Parameter(torch.tensor(0.0))
        self.norm_scale = nn.Parameter(torch.tensor(1.0))

        # to be used when we want to scale the reward
        self.affine_offset = nn.Parameter(torch.tensor(0.0))
        self.affine_scale = nn.Parameter(torch.tensor(1.0))

        self.affine_offset.requires_grad = False
        self.affine_scale.requires_grad = False

    def forward(self, data):
        """
        data: a batch of input, torch.Tensor or torch_geometric.data.Data type
            - torch.Tensor: (batch_size, 3, num_points), as common batch input

            - torch_geometric.data.Data, as torch_geometric batch input:
                data.x: (batch_size * ~num_points, C), batch nodes/points feature,
                    ~num_points means each sample can have different number of points/nodes

                data.pos: (batch_size * ~num_points, 3)

                data.batch: (batch_size * ~num_points,), a column vector of graph/pointcloud
                    idendifiers for all nodes of all graphs/pointclouds in the batch. See
                    pytorch_gemometric documentation for more information
        """
        dense_input = True if isinstance(data, torch.Tensor) else False

        if dense_input:
            # Convert to torch_geometric.data.Data type
            data = data.transpose(1, 2).contiguous()
            batch_size, N, _ = data.shape  # (batch_size, num_points, 3)
            pos = data.view(batch_size * N, -1)
            batch = torch.zeros((batch_size, N), device=pos.device, dtype=torch.long)
            for i in range(batch_size):
                batch[i] = i
            batch = batch.view(-1)

            data = Data()
            data.pos, data.batch = pos, batch

        if not hasattr(data, "x"):
            data.x = None
        data_in = data.x, data.pos, data.batch

        sa1_out = self.sa1_module(data_in)
        sa2_out = self.sa2_module(sa1_out)
        sa3_out = self.sa3_module(sa2_out)

        global_feature_vec = sa3_out[0]

        gf_norm = torch.norm(global_feature_vec, p=2, dim=1, keepdim=True)
        gf_norm = self.norm_scale * gf_norm + self.norm_offset  # will seek to maximise norm as reward val. norm ~ quality of data
        x = gf_norm

        # get rwd score

        # x=self.rwd_fc1(global_feature_vec)
        # x=self.relu(x)
        # x=self.rwd_fc2(x)
        # x=self.relu(x)
        # x=self.rwd_fc3(x)

        # return(global_feature_vec,x)
        return x

        # fp3_out = self.fp3_module(sa3_out, sa2_out)
        # fp2_out = self.fp2_module(fp3_out, sa1_out)
        # fp1_out = self.fp1_module(fp2_out, data_in)

        # fp1_out_x, fp1_out_pos, fp1_out_batch = fp1_out
        # x = self.fc2(self.dropout1(self.fc1(fp1_out_x)))
        # x = F.log_softmax(x, dim=-1)

        # print(fp1_out_x.shape)
        # print(fp1_out_batch.shape)
        # print(self.fc1)

        if dense_input:
            return x.view(batch_size, N, self.num_classes)

        else:
            return x, fp1_out_batch


class rwd_model_pointnet2(torch.nn.Module):
    """
    ref:
        - https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_part_seg.py
        - https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet++.py
    """

    # but modified to go global vec for the faces....
    def __init__(self, **kwargs):
        super().__init__()
        self.num_classes = 2  # leave not used anyway
        self.model_class = "rwd_model_pointnet"
        # SA1
        sa1_sample_ratio = 0.5
        sa1_radius = 0.2
        sa1_max_num_neighbours = 64
        sa1_mlp = make_mlp(3, [64, 64, 128])
        self.sa1_module = PointNet2SAModule(sa1_sample_ratio, sa1_radius, sa1_max_num_neighbours, sa1_mlp)

        # SA2
        sa2_sample_ratio = 0.25
        sa2_radius = 0.4
        sa2_max_num_neighbours = 64
        sa2_mlp = make_mlp(128 + 3, [128, 128, 256])
        self.sa2_module = PointNet2SAModule(sa2_sample_ratio, sa2_radius, sa2_max_num_neighbours, sa2_mlp)

        # SA3
        sa3_mlp = make_mlp(256 + 3, [256, 512, 1024])
        self.sa3_module = PointNet2GlobalSAModule(sa3_mlp)

        self.rwd_fc1 = torch.nn.Linear(1024, 256)

        self.rwd_fc2 = torch.nn.Linear(256, 32)

        self.rwd_fc3 = torch.nn.Linear(32, 1)

        self.relu = torch.nn.functional.relu

        ##
        knn_num = 3

        # FP3, reverse of sa3
        fp3_knn_num = 1  # After global sa module, there is only one point in point cloud
        fp3_mlp = make_mlp(1024 + 256 + 3, [256, 256])
        self.fp3_module = PointNet2FPModule(fp3_knn_num, fp3_mlp)

        # FP2, reverse of sa2
        fp2_knn_num = knn_num
        fp2_mlp = make_mlp(256 + 128 + 3, [256, 128])
        self.fp2_module = PointNet2FPModule(fp2_knn_num, fp2_mlp)

        # FP1, reverse of sa1
        fp1_knn_num = knn_num
        fp1_mlp = make_mlp(128 + 3, [128, 128, 128])
        self.fp1_module = PointNet2FPModule(fp1_knn_num, fp1_mlp)

        self.fc1 = Lin(128, 128)
        self.dropout1 = Dropout(p=0.5)
        self.fc2 = Lin(128, self.num_classes)

        # to be used when we want to scale the reward
        self.affine_offset = nn.Parameter(torch.tensor(0.0))
        self.affine_scale = nn.Parameter(torch.tensor(1.0))

        self.affine_offset.requires_grad = False
        self.affine_scale.requires_grad = False

    def forward(self, data):
        """
        data: a batch of input, torch.Tensor or torch_geometric.data.Data type
            - torch.Tensor: (batch_size, 3, num_points), as common batch input

            - torch_geometric.data.Data, as torch_geometric batch input:
                data.x: (batch_size * ~num_points, C), batch nodes/points feature,
                    ~num_points means each sample can have different number of points/nodes

                data.pos: (batch_size * ~num_points, 3)

                data.batch: (batch_size * ~num_points,), a column vector of graph/pointcloud
                    idendifiers for all nodes of all graphs/pointclouds in the batch. See
                    pytorch_gemometric documentation for more information
        """
        dense_input = True if isinstance(data, torch.Tensor) else False

        if dense_input:
            # Convert to torch_geometric.data.Data type
            data = data.transpose(1, 2).contiguous()
            batch_size, N, _ = data.shape  # (batch_size, num_points, 3)
            pos = data.view(batch_size * N, -1)
            batch = torch.zeros((batch_size, N), device=pos.device, dtype=torch.long)
            for i in range(batch_size):
                batch[i] = i
            batch = batch.view(-1)

            data = Data()
            data.pos, data.batch = pos, batch

        if not hasattr(data, "x"):
            data.x = None
        data_in = data.x, data.pos, data.batch

        sa1_out = self.sa1_module(data_in)
        sa2_out = self.sa2_module(sa1_out)
        sa3_out = self.sa3_module(sa2_out)

        global_feature_vec = sa3_out[0]

        # get rwd score

        x = self.rwd_fc1(global_feature_vec)
        x = self.relu(x)
        x = self.rwd_fc2(x)
        x = self.relu(x)
        x = self.rwd_fc3(x)

        # return(global_feature_vec,x)
        return x

        # fp3_out = self.fp3_module(sa3_out, sa2_out)
        # fp2_out = self.fp2_module(fp3_out, sa1_out)
        # fp1_out = self.fp1_module(fp2_out, data_in)

        # fp1_out_x, fp1_out_pos, fp1_out_batch = fp1_out
        # x = self.fc2(self.dropout1(self.fc1(fp1_out_x)))
        # x = F.log_softmax(x, dim=-1)

        # print(fp1_out_x.shape)
        # print(fp1_out_batch.shape)
        # print(self.fc1)

        if dense_input:
            return x.view(batch_size, N, self.num_classes)

        else:
            return x, fp1_out_batch


class rwd_model_pointnet2_global(torch.nn.Module):
    """
    ref:
        - https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_part_seg.py
        - https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet++.py
    """

    # but modified to go global vec for the faces....
    def __init__(self, **kwargs):
        super().__init__()
        self.num_classes = 2  # leave not used anyway
        self.model_class = "rwd_model_pointnet2_global"

        self.reward_model_type = "rwd_model_pointnet2_global"
        # SA1
        sa1_sample_ratio = 0.5
        sa1_radius = 0.2
        sa1_max_num_neighbours = 64
        sa1_mlp = make_mlp(3, [64, 64, 128])
        self.sa1_module = PointNet2SAModule(sa1_sample_ratio, sa1_radius, sa1_max_num_neighbours, sa1_mlp)

        # SA2
        sa2_sample_ratio = 0.25
        sa2_radius = 0.4
        sa2_max_num_neighbours = 64
        sa2_mlp = make_mlp(128 + 3, [128, 128, 256])
        self.sa2_module = PointNet2SAModule(sa2_sample_ratio, sa2_radius, sa2_max_num_neighbours, sa2_mlp)

        # # SA3
        # sa3_mlp = make_mlp(256+3, [256, 512, 1024])
        # self.sa3_module = PointNet2GlobalSAModule(sa3_mlp)

        # SA3
        sa3_mlp = make_mlp(256 + 3, [256, 512, 1024])
        self.sa3_module = PointNet2GlobalSAModule(sa3_mlp)

        self.projection_head_first = torch.nn.Linear(1024, 32)
        self.projection_head_second = torch.nn.Linear(32, 32)

        self.rwd_fc1 = torch.nn.Linear(1024, 256)

        self.rwd_fc2 = torch.nn.Linear(256, 32)

        self.rwd_fc3 = torch.nn.Linear(32, 1)

        self.relu = torch.nn.functional.relu

        self.sigmoid = torch.nn.functional.sigmoid

        self.softmax = torch.nn.functional.softmax

        ##
        knn_num = 3

        # FP3, reverse of sa3
        fp3_knn_num = 1  # After global sa module, there is only one point in point cloud
        fp3_mlp = make_mlp(1024 + 256 + 3, [256, 256])
        self.fp3_module = PointNet2FPModule(fp3_knn_num, fp3_mlp)

        # FP2, reverse of sa2
        fp2_knn_num = knn_num
        fp2_mlp = make_mlp(256 + 128 + 3, [256, 128])
        self.fp2_module = PointNet2FPModule(fp2_knn_num, fp2_mlp)

        # FP1, reverse of sa1
        fp1_knn_num = knn_num
        fp1_mlp = make_mlp(128 + 3, [128, 128, 128])
        self.fp1_module = PointNet2FPModule(fp1_knn_num, fp1_mlp)

        self.fc1 = Lin(128, 128)
        self.dropout1 = Dropout(p=0.5)
        self.fc2 = Lin(128, self.num_classes)

        # to be used when we want to scale the reward
        self.affine_offset = nn.Parameter(torch.tensor(0.0))
        self.affine_scale = nn.Parameter(torch.tensor(1.0))

        self.affine_offset.requires_grad = False
        self.affine_scale.requires_grad = False

        self.rwd_global_fc1_head1 = torch.nn.Linear(1024, 512)
        self.rwd_global_fc2_head1 = torch.nn.Linear(512, 256)

        self.rwd_global_fc1_head2 = torch.nn.Linear(1024, 512)
        self.rwd_global_fc2_head2 = torch.nn.Linear(512, 256)

        self.rwd_global_join_fc1 = torch.nn.Linear(512, 256)
        self.rwd_global_join_fc2 = torch.nn.Linear(256, 128)
        self.rwd_global_join_fc3 = torch.nn.Linear(128, 2)

        # self.fc1_bn = nn.BatchNorm1d(512) #best model may use instance norm not sure
        # self.fc2_bn = nn.BatchNorm1d(512)

        # self.rwd_global_fc1_head1=torch.nn.Linear(512,256)
        # self.rwd_global_fc2_head1=torch.nn.Linear(256,256)

        # self.rwd_global_fc1_head2=torch.nn.Linear(512,256)
        # self.rwd_global_fc2_head2=torch.nn.Linear(256,256)

        # self.rwd_global_join_fc1=torch.nn.Linear(512,256)
        # self.rwd_global_join_fc2=torch.nn.Linear(256,128)
        # self.rwd_global_join_fc3=torch.nn.Linear(128,2)

        # self.fc1_bn = nn.InstanceNorm1d(256)
        # self.fc2_bn = nn.InstanceNorm1d(256)

        # self.rwd_global_fc2=torch.nn.Linear(256,32)

        # self.rwd_global_fc3=torch.nn.Linear(32,1)

        self.relu = torch.nn.functional.relu

    def forward(self, data):
        """
        data: a batch of input, torch.Tensor or torch_geometric.data.Data type
            - torch.Tensor: (batch_size, 3, num_points), as common batch input

            - torch_geometric.data.Data, as torch_geometric batch input:
                data.x: (batch_size * ~num_points, C), batch nodes/points feature,
                    ~num_points means each sample can have different number of points/nodes

                data.pos: (batch_size * ~num_points, 3)

                data.batch: (batch_size * ~num_points,), a column vector of graph/pointcloud
                    idendifiers for all nodes of all graphs/pointclouds in the batch. See
                    pytorch_gemometric documentation for more information
        """
        dense_input = True if isinstance(data, torch.Tensor) else False

        if dense_input:
            # Convert to torch_geometric.data.Data type
            data = data.transpose(1, 2).contiguous()
            batch_size, N, _ = data.shape  # (batch_size, num_points, 3)
            pos = data.view(batch_size * N, -1)
            batch = torch.zeros((batch_size, N), device=pos.device, dtype=torch.long)
            for i in range(batch_size):
                batch[i] = i
            batch = batch.view(-1)

            data = Data()
            data.pos, data.batch = pos, batch

        if not hasattr(data, "x"):
            data.x = None
        data_in = data.x, data.pos, data.batch

        sa1_out = self.sa1_module(data_in)
        sa2_out = self.sa2_module(sa1_out)
        sa3_out = self.sa3_module(sa2_out)

        global_feature_vec = sa3_out[0]

        return global_feature_vec

    def projection_head(self, gv):
        x = self.projection_head_first(gv)
        x = self.relu(x)
        x = self.projection_head_second(x)
        return x

    def forward_from_cat_global_vectors(self, gv1, gv2, with_softmax=True):
        # maps 2 global vectors to probability that gv1 > gv2

        x1 = gv1

        x1 = self.rwd_global_fc1_head1(x1)
        x1 = self.relu(x1)
        # x1=self.fc1_bn(x1)

        x1 = self.rwd_global_fc2_head1(x1)
        x1 = self.relu(x1)  # intermediate rep...

        x2 = gv2

        x2 = self.rwd_global_fc1_head2(x2)
        x2 = self.relu(x2)
        # x2=self.fc2_bn(x2)

        x2 = self.rwd_global_fc2_head2(x2)
        x2 = self.relu(x2)  # intermediate rep...

        cat_rep = torch.cat([x1, x2], 1)

        x = self.rwd_global_join_fc1(cat_rep)
        x = self.relu(x)
        x = self.rwd_global_join_fc2(x)
        x = self.relu(x)
        x = self.rwd_global_join_fc3(x)

        if with_softmax:
            x = self.softmax(x)

        return x


# class dset_single_stream_ordered_minimal(torch.utils.data.Dataset):
#     'Characterizes a dataset for PyTorch'
#     def __init__(self, all_combined_rankings,dtype,ddir_func,transforms=None):
#         'Initialization'
#         #!!!! MUST BE ORDERED OR METHOD WILL FAIL!!!!#
#         self.all_combined_rankings_ordered = all_combined_rankings #this is the list of ordered combo for the current batch. precomputed.
#         self.dtype=dtype
#         self.ddir_func=ddir_func
#         #self.device=torch.device('cuda')
#         self.transforms=None
#         if transforms is not None:
#             self.transforms=transforms

#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.all_combined_rankings_ordered)

#     def __getitem__(self, index):
#         'Generates one sample of data'
#         # Select sample
#         ordered_batch = self.all_combined_rankings_ordered[index]

#         batch_len=len(ordered_batch[ordered_batch!=-1])

#         padded_vals_len=len(ordered_batch)-batch_len


#         seeds_in_batch=torch.tensor(ordered_batch[:batch_len].astype(int))#.unique()

#         #tts=torch.argsort(seeds_in_batch)
#         o=self.dtype

#         if o=='triple_rgb_lmks_98':
#             files=[assemble_triple_lmks(ddir=self.ddir_func(s),seed=s) for s in seeds_in_batch]

#         elif o=='canonical_rgb_lmks_98':
#             files=[assemble_single_lmks(ddir=self.ddir_func(s),seed=s) for s in seeds_in_batch]

#         elif o=='triple_rgb':
#             files=[assemble_triple_rgb(ddir=self.ddir_func(s),seed=s) for s in seeds_in_batch]

#         elif o=='canonical_rgb':
#             files=[assemble_single_rgb(ddir=self.ddir_func(s),seed=s) for s in seeds_in_batch]

#         elif 'triple_dmap' in o:
#             fns=[create_pt_fn(ddir=self.ddir_func(s),ot=o,seed=s) for s in seeds_in_batch]
#             #files=[torch.load(f,map_location=torch.device('cuda')) for f in fns]
#             files=[torch.load(f) for f in fns]#,map_location=torch.device('cuda')) for f in fns]


#         else:
#             fns=[create_pt_fn(ddir=self.ddir_func(s),ot=o,seed=s) for s in seeds_in_batch]
#            # files=[torch.load(f,map_location=torch.device('cuda')) for f in fns]
#             files=[torch.load(f) for f in fns]#,map_location=torch.device('cuda')) for f in fns]


#         if self.transforms is not None:
#             files=[self.transforms(f) for f in files]

#         extra_pad=torch.zeros_like(files[0])

#         padlist=[extra_pad for k in range(padded_vals_len)]

#         files+=padlist

#         #dict_rename={s.item():argsorted.item() for s,argsorted in zip(seeds_in_batch,tts)}

#         #batch=torch.tensor(batch.astype(int))
#         #for s in seeds_in_batch:
#         #    batch[batch==s]=dict_rename[s.item()]

#         #col1=batch[:,0]
#         #bc=[files[b].unsqueeze(0) for b in col1]
#         #col1=torch.cat(bc,0).unsqueeze(1)
#         #col2=batch[:,1]
#         #bc=[files[b].unsqueeze(0) for b in col2]
#         #col2=torch.cat(bc,0).unsqueeze(1)
#         #combined_cols=torch.cat([col1,col2],dim=1)

#         return o,dict(files=files,batch_len=batch_len)


# from here https://androidkt.com/pytorch-dataloader-set-pin_memory-to-true/


class SimpleCustomBatchDmap3:
    def __init__(self, batch):
        # transposed_data = list(zip(*data))
        i = 0
        # current_dtype_name=batch[0][i][0]
        batches = [b[i][1] for b in batch]
        self.file_batch = torch.cat(
            [torch.cat([bb.unsqueeze(0) for bb in b["files"]], dim=0).unsqueeze(0) for b in batches],
            0,
        )
        self.lens_batch = torch.cat([torch.tensor(b["batch_len"]).unsqueeze(0) for b in batches])

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.file_batch = self.file_batch.pin_memory()
        self.lens_batch = self.lens_batch.pin_memory()
        return self


class SimpleCustomBatchPointCloud:
    def __init__(self, batch):
        # transposed_data = list(zip(*data))
        i = 0
        # current_dtype_name=batch[0][i][0]

        # files=batch[0][1]['files']
        # lens=batch[0][1]['batch_len']
        batches = [b[i][1] for b in batch]
        self.file_batch = torch.cat(
            [torch.cat([bb.unsqueeze(0) for bb in b["files"]], dim=0).unsqueeze(0) for b in batches],
            0,
        )
        self.lens_batch = torch.cat([torch.tensor(b["batch_len"]).unsqueeze(0) for b in batches])

        # list_of_files=[f[:l] for f,l in zip(files,lens)]
        # return(batch)
        # not implemented....

        # transposed_data = list(zip(*data))
        # i=0
        # #current_dtype_name=batch[0][i][0]
        # batches=[b[i][1] for b in batch]
        # self.file_batch=torch.cat([torch.cat([bb.unsqueeze(0) for bb in b['files']],dim=0).unsqueeze(0) for b in batches],0)
        # self.lens_batch=torch.cat([torch.tensor(b['batch_len']).unsqueeze(0) for b in batches])

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.file_batch = self.file_batch.pin_memory()
        self.lens_batch = self.lens_batch.pin_memory()
        return self


def collate_wrapper_pcd(batch):
    return SimpleCustomBatchPointCloud(batch)


def collate_wrapper_dm3(batch):
    return SimpleCustomBatchDmap3(batch)


def collate_fn(batch):
    # batch_d=dict(batch)

    retdict = {}
    n_elements = len(batch[0])

    for i in range(n_elements):
        current_dtype_name = batch[0][i][0]
        batches = [b[i][1] for b in batch]
        file_batch = torch.cat(
            [torch.cat([bb.unsqueeze(0) for bb in b["files"]], dim=0).unsqueeze(0) for b in batches],
            0,
        )
        lens_batch = torch.cat([torch.tensor(b["batch_len"]).unsqueeze(0) for b in batches])
        retdict[current_dtype_name] = dict(files=file_batch, lens=lens_batch)

    # print('pausing here')
    return retdict


class RescaleIm(torch.nn.Module):
    def forward(self, dmap):  # we assume inputs are always structured like this
        rmin = 2.25
        rmax = 3.3
        dm_min = -1.0
        dmap = (((dmap - rmin) / (rmax - rmin)) * 2) - 1
        dmap[dmap < dm_min] = dm_min
        dmap[dmap > 1.0] = 1.0
        return dmap
        # Do some transformations. Here, we're just passing though the input
        # return img, bboxes, label


# dxm=
class ReScaleDmap(nn.Module):
    def __init__(self, rescale_size=160, mode="bicubic", align_corners=False):
        super().__init__()
        # self.interp = interpolate
        self.rescale_size = rescale_size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        B, C, T, W, H = x.size()
        x = x.reshape(B * C, T, W, H)
        x = F.interpolate(x, size=(self.rescale_size, self.rescale_size), mode=self.mode)
        x = x.reshape(B, C, T, W, H)
        return x


from torchvision.transforms import v2

dmap_transforms_facenet_160 = torch.nn.Sequential(
    # v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
    # v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
    # ...
    RescaleIm(),
    ReScaleDmap(),
    # v2.Resize(size=(160, 160), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),  # Or Resize(antialias=True)
    # v2.Lambda(lambd=rescale_im)
    # v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
)

# seed_ids? not need just use entier_dict_of_fn

dmt_jit = torch.jit.script(dmap_transforms_facenet_160)


# for contrastive loss, no transforms applied...
class dset_pcd_for_closs(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, all_seeds, dtype, ddir_func, transforms=None):
        super().__init__()
        "Initialization"
        self.all_seeds_in_batch = all_seeds  # this is the list of ordered combo for the current batch. precomputed.
        self.dtype = dtype
        self.ddir_func = ddir_func
        # self.device=torch.device('cuda')
        self.transforms = None
        if transforms is not None:
            self.transforms = transforms
            # point_samples_per_pc_batch

        self.n_points = 4096

        self.xcut = 0.32
        self.ycut = 0.3

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.all_seeds_in_batch)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        seed = self.all_seeds_in_batch[index]
        fn = create_pt_fn(ddir=self.ddir_func(seed), ot="pcd_as_pt", seed=seed)  # for s in seeds_in_batch]
        ptc = torch.load(fn, map_location=torch.device("cpu"))

        ptm = ptc - ptc.mean(0)
        pty = torch.linalg.norm(ptm[:, 0].unsqueeze(0), ord=2, axis=0)
        ptx = torch.linalg.norm(ptm[:, 1].unsqueeze(0), ord=2, axis=0)
        sel_np = torch.logical_and(ptx < self.xcut, pty < self.ycut)
        # center it

        # crops randomly

        # return....
        # dset_pcd_for_closs

        # ptc=ptc[torch.linalg.norm(ptc-ptc.mean(0),ord=2,axis=0)<self.xcut]
        # pcd=ptc[torch.linalg.norm(ptc-ptc.mean(1),ord=2,axis=0)<self.ycut]

        pcd = ptc[sel_np]
        pcd = self.downsample_pcd_points(pcd)  # self.n_point_samples_per_pcd_batch) #for f in files] #random sample 2048
        return pcd
        # batch_len=len(ordered_batch[ordered_batch!=-1])
        # padded_vals_len=len(ordered_batch)-batch_len
        # seeds_in_batch=torch.tensor(ordered_batch[:batch_len].astype(int))#.unique()

    def downsample_pcd_points(self, ttl):
        perm = torch.randperm(ttl.size(0))
        idx = perm[: min(ttl.size(0), self.n_points)]
        samples = ttl[idx]
        return samples


def train_contrastive_save_rwd_model(dset_dict_dirs, model_name, model_class, plot_dists_as_train, n_epochs=20, **model_kwargs):
    reward_model, da = new_reward_model(model_name, model_class, **model_kwargs)
    da.model_kwargs = model_kwargs["model_kwargs"]
    os.makedirs(da.model_dir, exist_ok=True)

    model_kwargs_fn = os.path.join(da.model_dir, "model_kwargs.json")
    with open(model_kwargs_fn, "w") as f:
        json.dump(model_kwargs, f)
    print("model kwargs init saved")

    dset_dict = {}
    optimizer = get_optimiser(reward_model, LR=da.model_kwargs["LR"])

    # using_old_dloader=False
    # if using_old_dloader:
    #     for dname in dset_dict_dirs.keys():
    #         dset_dict[dname]=torch.load(dset_dict_dirs[dname],map_location=torch.device('cuda'))

    for dname in dset_dict_dirs.keys():
        dset_dict[dname] = torch.load(dset_dict_dirs[dname], map_location=torch.device("cuda"))

    train_loader = dset_dict["train"]
    val_loader = dset_dict["val"]
    # test_loader=dset_dict['test']

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if device == torch.device("cuda:0"):
        torch.backends.cudnn.benchmark = True

    params = {
        "batch_size": model_kwargs["model_kwargs"]["batch_size"],
        "shuffle": True,
        "num_workers": 8,
        "pin_memory": True,
    }

    params_gpu = {"batch_size": 64, "shuffle": True, "num_workers": 0, "pin_memory": False}

    sel_params = params

    if model_kwargs["model_kwargs"]["using_precomputed_dataset"]:
        sel_params = params_gpu
    dset_dict["sel_params"] = sel_params

    # max_epochs = 100

    # Datasets
    # partition = # IDs
    # labels = # Labels

    # for d in dset_dict[dname].dsets:
    #     if d.dtype=='pcd_as_pt':
    #         d.n_point_samples_per_pcd_batch=int(model_kwargs['model_kwargs']['n_point_samples_per_pcd_batch'])
    #     #n_point_samples_per_pcd_batch

    # test_set = Dataset(partition['validation'], labels)
    # test_loader = torch.utils.data.DataLoader(test_dset, **sel_params)

    # print('pausing here')
    # import open3d as o3d

    # for p in tqdm(val_loader):
    #     current_batch=p
    #     pcds=[p.file_batch[0][k] for k in range(len(p.file_batch[0]))]

    #     seed=val_dset.dsets[0].get_sorted_unique_seeds()
    #     example=val_dset.dsets[0].load_single_example_no_transform(seed[0])
    #     retval=o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(example.cpu().numpy()))

    #     retval=o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(example.unsqueeze(0).cpu().numpy().reshape(-1,3)))
    #     retval1=o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcds[2].unsqueeze(0).cpu().numpy().reshape(-1,3)))

    #     #coords=retval.create_coordinate_frame()
    #     o3d.visualization.draw_geometries([retval])#,coords])

    # debug visualistaion......................

    # SEED=25401

    # pt_pcd_fn=f'/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/000_RLHF_AM/rlhf_meshes/rlhf_meshes_ffhq512-128_const_noise_t1_augment/pcd_as_pt_s_{SEED}.pt'

    # ptc=torch.load(pt_pcd_fn).cpu().detach().numpy()

    # retval=o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(ptc))
    # o3d.visualization.draw_geometries([retval])

    print("create new rwd mode")

    print("train it now")
    # return_all_indiv_examples

    # reward_model.da=da # here so we can read in args later on...

    # batch_format_func=get_batch_format_func(model_class)

    # da.batch_format_func=batch_format_func

    if model_class == "rwd_model_3dmap_vgg_minimal":
        da.vgg19_to_4096_model = vgg19_to_4096()
        da.vgg19_to_4096_model = da.vgg19_to_4096_model.cuda()
        # train_model_dmap_3(da,reward_model,train_loader,val_loader,test_loader,n_epochs=n_epochs)
        # return(None)
    elif model_class == "rwd_model_3dmap_vggface2_minimal":
        vggf = vggface2_to_512().to(device)
        da.vggface2_to_512_model = vggf

    # train_model_stylecode(da,optimizer,reward_model,train_loader,val_loader,n_epochs=n_epochs)

    # train_rwd_model_from_scratch(   da,optimizer,reward_model,train_loader,val_loader,test_loader,plot_dists_as_train=plot_dists_as_train,
    #                                n_epochs=n_epochs)

    # train_rwd_model_from_scratch_with_multi_loader(   da,optimizer,reward_model,train_loader,val_loader,test_loader,plot_dists_as_train=plot_dists_as_train,
    #                                 n_epochs=n_epochs,dset_dict=dset_dict)

    train_contrastive_reward_model(
        da,
        optimizer,
        reward_model,
        dset_dict,
        plot_dists_as_train=plot_dists_as_train,
        n_epochs=n_epochs,
    )

    # train_paired_rwd_model_from_scratch_with_multi_loader(   da,optimizer,reward_model,train_loader,val_loader,test_loader,plot_dists_as_train=plot_dists_as_train,
    #                             n_epochs=n_epochs,dset_dict=dset_dict)

    print("evaluating rwd mod")

    state_dict = torch.load(os.path.join(da.model_dir, "model_state_dict_best.pth"))
    reward_model.load_state_dict(state_dict)
    reward_model.eval()
    save_optimal_rwd_model(reward_model, da)


def train_evaluate_save_rwd_model(
    dset_dict,
    model_name,
    model_class,
    plot_dists_as_train,
    n_epochs=20,
    reload_chkpt=True,
    **model_kwargs,
):
    reward_model, da = new_reward_model(model_name, model_class, **model_kwargs)
    da.model_kwargs = model_kwargs["model_kwargs"]
    os.makedirs(da.model_dir, exist_ok=True)

    da.current_epoch = 0

    if reload_chkpt:
        condition = "latest"
        pkl_save_name = os.path.join(da.model_dir, f"chkpt_{condition}.pkl")
        # pkl_save_name=os.path.join(da.model_dir,f'{da.MODEL_NAME}_f{condition}.pkl')

        if os.path.exists(pkl_save_name):
            reward_model, da = load_rwd_mdl_from_pkl(pkl_dir=pkl_save_name)

    else:
        model_kwargs_fn = os.path.join(da.model_dir, "model_kwargs.json")
        with open(model_kwargs_fn, "w") as f:
            json.dump(model_kwargs, f)
        print("model kwargs init saved")

    # dset_dict={}
    optimizer = get_optimiser(reward_model, LR=da.model_kwargs["LR"])

    using_old_dloader = False
    if using_old_dloader:
        for dname in dset_dict_dirs.keys():
            dset_dict[dname] = torch.load(dset_dict_dirs[dname], map_location=torch.device("cuda"))

        train_loader = dset_dict["train"]
        val_loader = dset_dict["val"]
        test_loader = dset_dict["test"]

    else:
        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True

        if da.model_class == "dm3":
            collate_fn = collate_wrapper_dm3

        elif da.model_class == "rwd_model_pointnet2":
            collate_fn = collate_wrapper_pcd

        elif da.model_class == "rwd_model_pointnet2_global":
            collate_fn = collate_wrapper_pcd

        elif da.model_class == "rwd_model_pointnet2_gfeature":
            collate_fn = collate_wrapper_pcd
        # Parameters
        params = {
            "batch_size": model_kwargs["model_kwargs"]["batch_size"],
            "shuffle": True,
            "num_workers": 8,
            "pin_memory": True,
            "collate_fn": collate_fn,
        }

        params_gpu = {"batch_size": 64, "shuffle": True, "num_workers": 0, "pin_memory": False}

        sel_params = params

        if model_kwargs["model_kwargs"]["using_precomputed_dataset"]:
            sel_params = params_gpu

        # max_epochs = 100

        # Datasets
        # partition = # IDs
        # labels = # Labels

        for dname in dset_dict.keys():
            dset_dict[dname] = torch.load(dset_dict[dname], map_location=torch.device("cuda"))

            for d in dset_dict[dname].dsets:
                if d.dtype == "pcd_as_pt":
                    d.n_point_samples_per_pcd_batch = int(model_kwargs["model_kwargs"]["n_point_samples_per_pcd_batch"])
                # n_point_samples_per_pcd_batch

        train_dset = dset_dict["train"]  # .to_iterable_dataset()#.to(device)
        val_dset = dset_dict["val"]  # .to_iterable_dataset()#.to(device)
        test_dset = dset_dict["test"]  # .to_iterable_dataset()#.to(device)

        # Generators
        # train_loader = Dataset(partition['train'], labels)
        train_loader = torch.utils.data.DataLoader(train_dset, **sel_params)

        sel_params["batch_size"] = 8
        sel_params["shuffle"] = False  # not set for case where we have 97% acc for test...
        # validation_set = Dataset(partition['validation'], labels)
        val_loader = torch.utils.data.DataLoader(val_dset, **sel_params)

        # test_set = Dataset(partition['validation'], labels)
        test_loader = torch.utils.data.DataLoader(test_dset, **sel_params)

    print("create new rwd mode")

    print("train it now")
    # return_all_indiv_examples

    # reward_model.da=da # here so we can read in args later on...

    batch_format_func = get_batch_format_func(model_class)

    da.batch_format_func = batch_format_func

    if model_class == "rwd_model_3dmap_vgg_minimal":
        da.vgg19_to_4096_model = vgg19_to_4096()
        da.vgg19_to_4096_model = da.vgg19_to_4096_model.cuda()
        # train_model_dmap_3(da,reward_model,train_loader,val_loader,test_loader,n_epochs=n_epochs)
        # return(None)
    elif model_class == "rwd_model_3dmap_vggface2_minimal":
        vggf = vggface2_to_512().to(device)
        da.vggface2_to_512_model = vggf

    # train_model_stylecode(da,optimizer,reward_model,train_loader,val_loader,n_epochs=n_epochs)

    # train_rwd_model_from_scratch(   da,optimizer,reward_model,train_loader,val_loader,test_loader,plot_dists_as_train=plot_dists_as_train,
    #                                n_epochs=n_epochs)

    # train_rwd_model_from_scratch_with_multi_loader(   da,optimizer,reward_model,train_loader,val_loader,test_loader,plot_dists_as_train=plot_dists_as_train,
    #                                 n_epochs=n_epochs,dset_dict=dset_dict)

    train_paired_rwd_model_from_scratch_with_multi_loader(
        da,
        optimizer,
        reward_model,
        train_loader,
        val_loader,
        test_loader,
        plot_dists_as_train=plot_dists_as_train,
        n_epochs=n_epochs,
        dset_dict=dset_dict,
    )

    print("evaluating rwd mod")

    state_dict = torch.load(os.path.join(da.model_dir, "model_state_dict_best.pth"))
    reward_model.load_state_dict(state_dict)
    reward_model.eval()
    save_optimal_rwd_model(reward_model, da)


def save_optimal_rwd_model(optimal_rwd_model, da):
    optimal_rwd_models_dir = os.path.join(da.RLHF_DIR, "optimal_reward_models")
    os.makedirs(optimal_rwd_models_dir, exist_ok=True)
    bundle_pkl = {}
    bundle_pkl["m_init_params"] = da.m_init_params
    bundle_pkl["model_state_dict"] = optimal_rwd_model.state_dict()
    bundle_pkl["reward_model_type"] = optimal_rwd_model.reward_model_type

    pkl_save_name = os.path.join(optimal_rwd_models_dir, f"{da.MODEL_NAME}.pkl")

    with open(pkl_save_name, "wb") as f:
        pickle.dump(bundle_pkl, f)

    print("saved optimal rwd_model")


def predict_style_code_with_vals(classifier, batch, da):
    stylecode = batch[3]
    pos_style = stylecode[:, 0, 0, :].unsqueeze(0)
    neg_style = stylecode[:, 1, 0, :].unsqueeze(0)
    zvals = -1
    cpred_winning = classifier.forward(pos_style)
    cpred_losing = classifier.forward(neg_style)
    diff = cpred_winning - cpred_losing
    loss = -torch.log(torch.sigmoid(diff))
    # sums=cpred_winning+cpred_losing
    indiv = dict(cp_w=cpred_winning, cp_l=cpred_losing)
    return (loss.mean(), indiv)


def predict_ldmk_2d_with_vals(classifier, batch, da):
    ldmk_2d = batch[4]
    pos_style = ldmk_2d[:, 0, :, :].reshape(-1, 2 * 98)
    neg_style = ldmk_2d[:, 1, :, :].reshape(-1, 2 * 98)
    zvals = -1
    cpred_winning = classifier.forward(pos_style)
    cpred_losing = classifier.forward(neg_style)
    diff = cpred_winning - cpred_losing
    loss = -torch.log(torch.sigmoid(diff))
    # sums=cpred_winning+cpred_losing
    indiv = dict(cp_w=cpred_winning, cp_l=cpred_losing)
    return (loss.mean(), indiv)


def predict_ldmk_2d_triple_with_vals(classifier, batch, da):
    ldmk_2d_triple = batch[5]
    pos_style = ldmk_2d_triple[:, 0, :, :, :]  # .reshape(-1,2*98*3)
    neg_style = ldmk_2d_triple[:, 1, :, :, :]  # .reshape(-1,2*98*3)
    zvals = -1
    cpred_winning = classifier.forward(pos_style)
    cpred_losing = classifier.forward(neg_style)
    diff = cpred_winning - cpred_losing
    loss = -torch.log(torch.sigmoid(diff))
    # sums=cpred_winning+cpred_losing
    indiv = dict(cp_w=cpred_winning, cp_l=cpred_losing)
    return (loss.mean(), indiv)


def predict_three_dmap_w_vals(classifier, batch, da):
    X_dmap = batch[0]
    # X_z=batch[1]
    scores = batch[1]
    ids = batch[2]
    pos_dmap = X_dmap[
        :,
        0,
    ]
    neg_dmap = X_dmap[
        :,
        1,
    ]
    cpred_winning = classifier.forward(pos_dmap)
    cpred_losing = classifier.forward(neg_dmap)
    diff = cpred_winning - cpred_losing
    loss = -torch.log(torch.sigmoid(diff))
    sums = cpred_winning + cpred_losing
    indiv = dict(cp_w=cpred_winning, cp_l=cpred_losing)
    return (loss.mean(), indiv)


def predict_three_dmap_w_vals_minimal_vgg_model(classifier, batch, da):
    X_dmap = batch[0]

    pos_dmap = X_dmap[
        :,
        0,
    ]
    neg_dmap = X_dmap[
        :,
        1,
    ]

    pos_dmap = da.vgg19_to_4096_model.forward(pos_dmap)
    neg_dmap = da.vgg19_to_4096_model.forward(neg_dmap)
    cpred_winning = classifier.forward(pos_dmap)
    cpred_losing = classifier.forward(neg_dmap)
    diff = cpred_winning - cpred_losing
    loss = -torch.log(torch.sigmoid(diff))
    indiv = dict(cp_w=cpred_winning, cp_l=cpred_losing)
    return (loss.mean(), indiv)


import torchvision.transforms as T


def get_transform_rc() -> callable:
    return torch.nn.Sequential(T.RandomCrop(160))


def get_transform_rs_to_160() -> callable:
    return torch.nn.Sequential(T.Resize(size=(160, 160)))


# precalc for depth map...

means = [0.3674, 0.1205, 0.3750]
stds = [0.5739, 0.3659, 0.5731]


def get_transform_rs_to_160_norm() -> callable:
    return torch.nn.Sequential(
        T.Resize(size=(160, 160)),
        T.Normalize(mean=means, std=stds),
    )


rcrop_160 = get_transform_rc()

rs_160 = get_transform_rs_to_160()

rs_160_norm = get_transform_rs_to_160_norm()

import itertools


def predict_three_dmap_w_vals_minimal_vggface2_model(classifier, batch, da):
    # X_dmap=batch['triple_dmap']['files'].squeeze(3).cuda()
    # Lengths=batch['triple_dmap']['lens']

    # X_dmap=batch['triple_dmap'].file_batch.to('cuda:0', non_blocking=True)
    # Lengths=batch['triple_dmap'].lens_batch.to('cuda:0', non_blocking=True)

    if da.model_kwargs["using_precomputed_dataset"] == True:
        X_dmap = batch[0].squeeze(3).to(torch.float32)
        Lengths = batch[1].to(torch.uint8)

        dmaps_computed = [da.vggface2_to_512_model.forward(rs_160_norm(xd[:L].view(-1, 3, 256, 256)).view(-1, 3, 160, 160)) for xd, L in zip(X_dmap, Lengths)]

    else:
        X_dmap = batch.file_batch.to("cuda:0", non_blocking=True)
        Lengths = batch.lens_batch.to("cuda:0", non_blocking=True)
        dmaps_computed = [da.vggface2_to_512_model.forward(xd[:L]) for xd, L in zip(X_dmap, Lengths)]

    cpreds = [classifier.forward(dmc) for dmc in dmaps_computed]
    ordered_combos = [list(itertools.combinations(range(L), 2)) for L in Lengths]
    intermediate_losses = [[-torch.log(torch.sigmoid(cp[o[0]] - cp[o[1]])) for o in oc] for oc, cp in zip(ordered_combos, cpreds)]

    im_val = [[torch.cat([cp[o[0]].unsqueeze(0), cp[o[1]].unsqueeze(0)]).flatten() for o in oc] for oc, cp in zip(ordered_combos, cpreds)]
    iml_stacked = torch.vstack([torch.vstack(il) for il in im_val])

    im_losses = [torch.mean(torch.cat(im)).unsqueeze(0) for im in intermediate_losses]
    entire_loss = torch.cat(im_losses).mean()
    loss = entire_loss
    indiv = dict(pairwise_comp=iml_stacked, rwd_vals=cpreds)

    return (loss, indiv)


def predict_pcd(classifier, batch, da):
    X_dmap = batch.file_batch.to("cuda:0", non_blocking=True).squeeze(2)
    Lengths = batch.lens_batch.to("cuda:0", non_blocking=True)
    cpreds = [classifier.forward(xd[:L]) for xd, L in zip(X_dmap, Lengths)]
    ordered_combos = [list(itertools.combinations(range(L), 2)) for L in Lengths]
    intermediate_losses = [[-torch.log(torch.sigmoid(cp[o[0]] - cp[o[1]])) for o in oc] for oc, cp in zip(ordered_combos, cpreds)]
    im_val = [[torch.cat([cp[o[0]].unsqueeze(0), cp[o[1]].unsqueeze(0)]).flatten() for o in oc] for oc, cp in zip(ordered_combos, cpreds)]
    iml_stacked = torch.vstack([torch.vstack(il) for il in im_val])
    im_losses = [torch.mean(torch.cat(im)).unsqueeze(0) for im in intermediate_losses]
    entire_loss = torch.cat(im_losses).mean()
    loss = entire_loss
    indiv = dict(pairwise_comp=iml_stacked, rwd_vals=cpreds)

    return (loss, indiv)


def reorder_pair_by_idx(pair, idx):
    out_tuple = (pair[idx[0]], pair[idx[1]])
    return out_tuple


ce_loss = nn.CrossEntropyLoss()
bce_w_logits_loss = nn.BCEWithLogitsLoss()


def get_rand_reordered_pair(p=0.5):
    sel_p = np.random.rand()
    in_tuple = (0, 1)
    if sel_p > p:
        out_tuple = (in_tuple[1], in_tuple[0])
    else:
        out_tuple = in_tuple
    return out_tuple


mse_loss = torch.nn.MSELoss()


# tested doesn't work
def predict_pcd_from_pairs_parallel(classifier, batch, da):
    X_dmap = batch.file_batch.to("cuda:0", non_blocking=True).squeeze(2)
    Lengths = batch.lens_batch.to("cuda:0", non_blocking=True)

    data = [xd[:L] for xd, L in zip(X_dmap, Lengths)]

    new_idxes = [len(d) for d in data]

    data_for_pred = torch.cat(data)

    cc = classifier.forward(data_for_pred)  # single forward pass

    cpreds = list(torch.split(cc, new_idxes))

    # cpreds=[classifier.forward(xd[:L]) for xd,L in zip(X_dmap,Lengths)] #single pass per each batch of ordered meshes
    ordered_combos = [list(itertools.combinations(range(L), 2)) for L in Lengths]
    binary_idx_order_rand = [[get_rand_reordered_pair(p=1) for t in oc] for oc in ordered_combos]
    ordered_combos_rand = [[reorder_pair_by_idx(o, p) for o, p in zip(oc, pc)] for oc, pc in zip(ordered_combos, binary_idx_order_rand)]
    # cn=[torch.norm(cp,2,1) for cp in cpreds]
    # [[torch.norm(cp[o[0]],2,1)-torch.norm(cp[o[1]],2,1)) for o in oc] for oc,cp in zip(ordered_combos_rand,cn)]
    # normdiff=torch.hstack([torch.hstack([(cp[o[0]]-cp[o[1]]) for o in oc]) for oc,cp in zip(ordered_combos_rand,cn)])
    # nd=torch.clamp(normdiff,max=0)

    # loss=torch.nn.L1Loss()

    # target=torch.zeros_like(nd)

    # meandiff_loss = loss(nd, target)
    batches = torch.cat([torch.cat([torch.cat((cp[o[0]].unsqueeze(0), cp[o[1]].unsqueeze(0)), 1) for o in oc]) for oc, cp in zip(ordered_combos_rand, cpreds)])
    batches_rev = torch.cat([torch.cat([torch.cat((cp[o[1]].unsqueeze(0), cp[o[0]].unsqueeze(0)), 1) for o in oc]) for oc, cp in zip(ordered_combos_rand, cpreds)])

    targets = torch.hstack([torch.tensor([0 for w in bi]).long().cuda() for bi in binary_idx_order_rand])
    targets_rev = torch.hstack([torch.tensor([1 for w in bi]).long().cuda() for bi in binary_idx_order_rand])

    # batches=torch.cat([torch.cat(b) for b in batches])

    batches = classifier.forward_from_cat_global_vectors(batches[:, :1024], batches[:, 1024:], with_softmax=False)
    batches_rev = classifier.forward_from_cat_global_vectors(batches_rev[:, :1024], batches_rev[:, 1024:], with_softmax=False)
    # targets=[torch.tensor([0 for w in bi]).long().cuda() for bi in binary_idx_order_rand]

    ce_batches_loss = ce_loss(batches, targets).unsqueeze(0)  # for i,t in zip(batches,targets)]
    ce_batches_loss_rev = ce_loss(batches_rev, targets_rev).unsqueeze(0)  # for i,t in zip(batches,targets)]

    # ce_batches_loss_rev=[ce_loss(i,t).unsqueeze(0) for i,t in zip(i_ss_rev,targets_rev)]

    # batches_rev=[classifier.forward_from_cat_global_vectors(b[:,:1024],b[:,1024:],with_softmax=False) for b in batches_rev]
    # targets_rev=[torch.tensor([1 for w in bi]).long().cuda() for bi in binary_idx_order_rand]

    # ce_batches_loss_rev=[ce_loss(i,t).unsqueeze(0) for i,t in zip(batches_rev,targets_rev)]

    # ce_batches_loss_rev=[ce_loss(i,t).unsqueeze(0) for i,t in zip(i_ss_rev,targets_rev)]

    # batches_rev= torch.cat([torch.cat(b) for b in batches_rev])
    # batch_combined=torch.cat([batches,batches_rev])

    # intermediate_softmax=        [[classifier.forward_from_cat_global_vectors(cp[o[0]].unsqueeze(0),cp[o[1]].unsqueeze(0),with_softmax=False) for o in oc] for oc,cp in zip(ordered_combos_rand,cpreds)]
    # intermediate_softmax_reverse=[[classifier.forward_from_cat_global_vectors(cp[o[1]].unsqueeze(0),cp[o[0]].unsqueeze(0),with_softmax=False) for o in oc] for oc,cp in zip(ordered_combos_rand,cpreds)]

    # targets_hstack=torch.hstack([torch.hstack(targets),torch.hstack(targets_rev)])

    # preds=classifier.forward_from_cat_global_vectors(batch_combined[:,:1024],batch_combined[:,1024:],with_softmax=False)

    # i_ss=[torch.cat(i) for i in intermediate_softmax]
    # i_ss_rev=[torch.cat(i) for i in intermediate_softmax_reverse]

    # entire_loss=ce_loss(preds,targets_hstack)
    # ci_0=[i_ss[0],targets[0],i_ss_rev[0],targets_rev[0]]
    # n_in_batch=torch.hstack(targets).shape[0]
    # ce_batches_loss_rev=[ce_loss(i,t).unsqueeze(0) for i,t in zip(i_ss_rev,targets_rev)]
    # ce_batches_loss=[ce_loss(i,t).unsqueeze(0) for i,t in zip(i_ss,targets)]

    entire_loss = (ce_batches_loss + ce_batches_loss_rev) / 2  # + meandiff_loss

    # ii=torch.vstack(i_s)
    # tt=torch.hstack(ts)
    # entire_loss=ce_batches_loss=ce_loss(ii,tt)#.unsqueeze(0) for i,t in zip(i_ss,targets)]

    # symmetry_pred_loss=[mse_loss(i,i_rev).unsqueeze(0) for i,i_rev in zip(i_ss,i_ss_rev)]
    # entire_loss=torch.cat(ce_batches_loss).mean() + torch.cat(symmetry_pred_loss).mean()
    # batches_rev=torch.cat(batches_rev,0)
    # batches=torch.cat(batches,0)
    return_for_logits = torch.cat((batches_rev[:, 1:], batches_rev[:, :1]), 1)

    return_for_logits = torch.cat((batches, return_for_logits), 0)

    # cpreds=torch.zeros([1,1])
    # pairwise_comp=cpreds
    # iml_stacked=pairwise_comp
    global_feature_preds = [cp.clone().detach().cpu() for cp in cpreds]

    indiv = dict(pairwise_comp=-1, global_feature_preds=global_feature_preds, logits=return_for_logits)

    loss = entire_loss
    return (loss, indiv)


# negative within each batch
def predict_pcd_from_pairs_combined(classifier, batch, da):
    X_dmap = batch.file_batch.to("cuda:0", non_blocking=True).squeeze(2)
    Lengths = batch.lens_batch.to("cuda:0", non_blocking=True)
    cpreds = [classifier.forward(xd[:L]) for xd, L in zip(X_dmap, Lengths)]  # single pass per each batch of ordered meshes
    ordered_combos = [list(itertools.combinations(range(L), 2)) for L in Lengths]
    binary_idx_order_rand = [[get_rand_reordered_pair(p=1) for t in oc] for oc in ordered_combos]
    ordered_combos_rand = [[reorder_pair_by_idx(o, p) for o, p in zip(oc, pc)] for oc, pc in zip(ordered_combos, binary_idx_order_rand)]
    # cn=[torch.norm(cp,2,1) for cp in cpreds]
    # [[torch.norm(cp[o[0]],2,1)-torch.norm(cp[o[1]],2,1)) for o in oc] for oc,cp in zip(ordered_combos_rand,cn)]
    # normdiff=torch.hstack([torch.hstack([(cp[o[0]]-cp[o[1]]) for o in oc]) for oc,cp in zip(ordered_combos_rand,cn)])
    # nd=torch.clamp(normdiff,max=0)

    # loss=torch.nn.L1Loss()
    Lengths_np = Lengths.detach().cpu().numpy()
    sum_in_batch = np.sum([get_ncomb2(l) for l in Lengths_np])

    # target=torch.zeros_like(nd)

    # meandiff_loss = loss(nd, target)
    batches = [torch.cat([torch.cat((cp[o[0]].unsqueeze(0), cp[o[1]].unsqueeze(0)), 1) for o in oc]) for oc, cp in zip(ordered_combos_rand, cpreds)]
    batches_rev = [torch.cat([torch.cat((cp[o[1]].unsqueeze(0), cp[o[0]].unsqueeze(0)), 1) for o in oc]) for oc, cp in zip(ordered_combos_rand, cpreds)]

    targets = [torch.tensor([0 for w in bi]).long().cuda() for bi in binary_idx_order_rand]
    targets_rev = [torch.tensor([1 for w in bi]).long().cuda() for bi in binary_idx_order_rand]

    batches_with_negative = [torch.cat((b, b_r)) for b, b_r in zip(batches, batches_rev)]
    targets_with_negative = [torch.hstack((t, t_r)) for t, t_r in zip(targets, targets_rev)]

    ce_batches_loss = [ce_loss(i, t).unsqueeze(0) for i, t in zip(batches_with_negative, targets_with_negative)]

    n_in_batch = torch.hstack(targets).shape[0]

    entire_loss = torch.cat(ce_batches_loss).mean()

    batches_rev = torch.cat(batches_rev, 0)
    batches = torch.cat(batches, 0)
    return_for_logits = torch.cat((batches_rev[:, 1:], batches_rev[:, :1]), 1)
    return_for_logits = torch.cat((batches, return_for_logits), 0)

    global_feature_preds = [cp.clone().detach().cpu() for cp in cpreds]

    indiv = dict(pairwise_comp=-1, global_feature_preds=global_feature_preds, logits=return_for_logits)

    loss = entire_loss
    return (loss, indiv)


def predict_pcd_from_pairs_split_posneg(classifier, batch, da):
    X_dmap = batch.file_batch.to("cuda:0", non_blocking=True).squeeze(2)
    Lengths = batch.lens_batch.to("cuda:0", non_blocking=True)
    cpreds = [classifier.forward(xd[:L]) for xd, L in zip(X_dmap, Lengths)]  # single pass per each batch of ordered meshes
    ordered_combos = [list(itertools.combinations(range(L), 2)) for L in Lengths]
    binary_idx_order_rand = [[get_rand_reordered_pair(p=1) for t in oc] for oc in ordered_combos]
    ordered_combos_rand = [[reorder_pair_by_idx(o, p) for o, p in zip(oc, pc)] for oc, pc in zip(ordered_combos, binary_idx_order_rand)]
    cn = [torch.norm(cp, 2, 1) for cp in cpreds]

    Lengths_np = Lengths.detach().cpu().numpy()
    sum_in_batch = np.sum([get_ncomb2(l) for l in Lengths_np])

    batches = [torch.cat([torch.cat((cp[o[0]].unsqueeze(0), cp[o[1]].unsqueeze(0)), 1) for o in oc]) for oc, cp in zip(ordered_combos_rand, cpreds)]
    batches_rev = [torch.cat([torch.cat((cp[o[1]].unsqueeze(0), cp[o[0]].unsqueeze(0)), 1) for o in oc]) for oc, cp in zip(ordered_combos_rand, cpreds)]

    batches = [classifier.forward_from_cat_global_vectors(b[:, :1024], b[:, 1024:], with_softmax=False) for b in batches]
    targets = [torch.tensor([0 for w in bi]).long().cuda() for bi in binary_idx_order_rand]
    ce_batches_loss = [ce_loss(i, t).unsqueeze(0) for i, t in zip(batches, targets)]

    batches_rev = [classifier.forward_from_cat_global_vectors(b[:, :1024], b[:, 1024:], with_softmax=False) for b in batches_rev]
    targets_rev = [torch.tensor([1 for w in bi]).long().cuda() for bi in binary_idx_order_rand]
    ce_batches_loss_rev = [ce_loss(i, t).unsqueeze(0) for i, t in zip(batches_rev, targets_rev)]

    n_in_batch = torch.hstack(targets).shape[0]
    entire_loss = (torch.cat(ce_batches_loss).mean() + torch.cat(ce_batches_loss_rev).mean()) / 2  # + meandiff_loss

    batches_rev = torch.cat(batches_rev, 0)
    batches = torch.cat(batches, 0)
    return_for_logits = torch.cat((batches_rev[:, 1:], batches_rev[:, :1]), 1)
    return_for_logits = torch.cat((batches, return_for_logits), 0)

    global_feature_preds = [cp.clone().detach().cpu() for cp in cpreds]

    indiv = dict(pairwise_comp=-1, global_feature_preds=global_feature_preds, logits=return_for_logits)

    loss = entire_loss
    return (loss, indiv)


def predict_pcd_from_pairs_scaled(classifier, batch, da):
    X_dmap = batch.file_batch.to("cuda:0", non_blocking=True).squeeze(2)
    Lengths = batch.lens_batch.to("cuda:0", non_blocking=True)
    cpreds = [classifier.forward(xd[:L]) for xd, L in zip(X_dmap, Lengths)]  # single pass per each batch of ordered meshes
    ordered_combos = [list(itertools.combinations(range(L), 2)) for L in Lengths]
    binary_idx_order_rand = [[get_rand_reordered_pair(p=1) for t in oc] for oc in ordered_combos]
    ordered_combos_rand = [[reorder_pair_by_idx(o, p) for o, p in zip(oc, pc)] for oc, pc in zip(ordered_combos, binary_idx_order_rand)]
    cn = [torch.norm(cp, 2, 1) for cp in cpreds]
    # [[torch.norm(cp[o[0]],2,1)-torch.norm(cp[o[1]],2,1)) for o in oc] for oc,cp in zip(ordered_combos_rand,cn)]
    # normdiff=torch.hstack([torch.hstack([(cp[o[0]]-cp[o[1]]) for o in oc]) for oc,cp in zip(ordered_combos_rand,cn)])
    # nd=torch.clamp(normdiff,max=0)

    # loss=torch.nn.L1Loss()
    Lengths_np = Lengths.detach().cpu().numpy()
    sum_in_batch = np.sum([get_ncomb2(l) for l in Lengths_np])

    # target=torch.zeros_like(nd)

    # meandiff_loss = loss(nd, target)
    batches = [torch.cat([torch.cat((cp[o[0]].unsqueeze(0), cp[o[1]].unsqueeze(0)), 1) for o in oc]) for oc, cp in zip(ordered_combos_rand, cpreds)]
    batches_rev = [torch.cat([torch.cat((cp[o[1]].unsqueeze(0), cp[o[0]].unsqueeze(0)), 1) for o in oc]) for oc, cp in zip(ordered_combos_rand, cpreds)]

    # batches=torch.cat([torch.cat(b) for b in batches])

    batches = [classifier.forward_from_cat_global_vectors(b[:, :1024], b[:, 1024:], with_softmax=False) for b in batches]
    targets = [torch.tensor([0 for w in bi]).long().cuda() for bi in binary_idx_order_rand]

    ce_batches_loss = [ce_loss(i, t).unsqueeze(0) * get_ncomb2(L) / sum_in_batch for i, t, L in zip(batches, targets, Lengths_np)]

    # ce_batches_loss_rev=[ce_loss(i,t).unsqueeze(0) for i,t in zip(i_ss_rev,targets_rev)]

    batches_rev = [classifier.forward_from_cat_global_vectors(b[:, :1024], b[:, 1024:], with_softmax=False) for b in batches_rev]
    targets_rev = [torch.tensor([1 for w in bi]).long().cuda() for bi in binary_idx_order_rand]

    ce_batches_loss_rev = [ce_loss(i, t).unsqueeze(0) * get_ncomb2(L) / sum_in_batch for i, t, L in zip(batches_rev, targets_rev, Lengths_np)]

    # ce_batches_loss_rev=[ce_loss(i,t).unsqueeze(0) for i,t in zip(i_ss_rev,targets_rev)]

    # batches_rev= torch.cat([torch.cat(b) for b in batches_rev])
    # batch_combined=torch.cat([batches,batches_rev])

    # intermediate_softmax=        [[classifier.forward_from_cat_global_vectors(cp[o[0]].unsqueeze(0),cp[o[1]].unsqueeze(0),with_softmax=False) for o in oc] for oc,cp in zip(ordered_combos_rand,cpreds)]
    # intermediate_softmax_reverse=[[classifier.forward_from_cat_global_vectors(cp[o[1]].unsqueeze(0),cp[o[0]].unsqueeze(0),with_softmax=False) for o in oc] for oc,cp in zip(ordered_combos_rand,cpreds)]

    # targets_hstack=torch.hstack([torch.hstack(targets),torch.hstack(targets_rev)])

    # preds=classifier.forward_from_cat_global_vectors(batch_combined[:,:1024],batch_combined[:,1024:],with_softmax=False)

    # i_ss=[torch.cat(i) for i in intermediate_softmax]
    # i_ss_rev=[torch.cat(i) for i in intermediate_softmax_reverse]

    # entire_loss=ce_loss(preds,targets_hstack)
    # ci_0=[i_ss[0],targets[0],i_ss_rev[0],targets_rev[0]]
    n_in_batch = torch.hstack(targets).shape[0]
    # ce_batches_loss_rev=[ce_loss(i,t).unsqueeze(0) for i,t in zip(i_ss_rev,targets_rev)]
    # ce_batches_loss=[ce_loss(i,t).unsqueeze(0) for i,t in zip(i_ss,targets)]

    entire_loss = (torch.cat(ce_batches_loss).mean() + torch.cat(ce_batches_loss_rev).mean()) / 2  # + meandiff_loss

    # ii=torch.vstack(i_s)
    # tt=torch.hstack(ts)
    # entire_loss=ce_batches_loss=ce_loss(ii,tt)#.unsqueeze(0) for i,t in zip(i_ss,targets)]

    # symmetry_pred_loss=[mse_loss(i,i_rev).unsqueeze(0) for i,i_rev in zip(i_ss,i_ss_rev)]
    # entire_loss=torch.cat(ce_batches_loss).mean() + torch.cat(symmetry_pred_loss).mean()
    batches_rev = torch.cat(batches_rev, 0)
    batches = torch.cat(batches, 0)
    return_for_logits = torch.cat((batches_rev[:, 1:], batches_rev[:, :1]), 1)

    return_for_logits = torch.cat((batches, return_for_logits), 0)

    # cpreds=torch.zeros([1,1])
    # pairwise_comp=cpreds
    # iml_stacked=pairwise_comp
    global_feature_preds = [cp.clone().detach().cpu() for cp in cpreds]

    indiv = dict(pairwise_comp=-1, global_feature_preds=global_feature_preds, logits=return_for_logits)

    loss = entire_loss
    return (loss, indiv)


def reorder_tuple_pair(in_tuple, p=0.5):
    sel_p = np.random.rand()
    if sel_p > p:
        out_tuple = (in_tuple[1], in_tuple[0])
    else:
        out_tuple = in_tuple
    return out_tuple


def get_rand_reordered_pair(p=0.5):
    sel_p = np.random.rand()
    in_tuple = (0, 1)
    if sel_p > p:
        out_tuple = (in_tuple[1], in_tuple[0])
    else:
        out_tuple = in_tuple
    return out_tuple


# rwd_model_pointnet2_global


def new_reward_model(model_name, model_class, model_kwargs):
    da = dargs()
    da.nrs = 128
    da.MODEL_NAME = model_name
    da.RLHF_DIR = RLHF_DIR
    da.model_class = model_class
    da.interpolation_mode = model_kwargs["da_interpolation_mode"]
    m_init_params = {}
    m_init_params["MODEL_CLASS"] = model_class

    model_dir = os.path.join(da.RLHF_DIR, "rlhf_reward_models", da.MODEL_NAME)
    da.model_dir = model_dir

    m_init_params.update(model_kwargs)

    da.m_init_params = m_init_params

    reward_model = eval(m_init_params["MODEL_CLASS"])(**m_init_params)  # this is reward model....
    reward_model = reward_model.cuda()
    reward_model.eval()
    reward_model.cuda()

    da.vgg19_to_4096_model = None  # only used for minimal vgg 3dmap model
    da.vggface2_to_512_model = None  # only used for minimal facenet 3dmap model

    return (reward_model, da)


def plot_rwd_dists_pcd(reward_model, da):
    mname_str = da.MODEL_NAME

    # reward_model,da=load_rwd_mdl(f'/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/000_RLHF_AM/optimal_reward_models/{mname_str}.pkl')

    if not hasattr(da, "interpolation_mode"):
        da.interpolation_mode = "nearest"  # quick hack
    # if 'interpolation_mode' not in a:
    #    print('hello')

    #    mname_str=['rwd_model_3dmap_facenet512_new_data2_10112_w_goodmesh_bilinear_epo_30_bs_256_hls_256_nh_4'][0]
    # state_dict=torch.load(os.path.join(f'/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/000_RLHF_AM/rlhf_reward_models/{mname_str}/model_state_epo_19.pth'))
    # reward_model.load_state_dict(state_dict)
    # reward_model.eval()

    print("affine scale (std)")
    print(reward_model.affine_scale)

    # set affine scale to 1.0

    reward_model.affine_scale = torch.nn.Parameter(torch.tensor(1.0))

    print("affine scale (std) AFTER set = 1.0")
    print(reward_model.affine_scale)

    fn_list_dict = {
        "rndm_truncate_2.0": _precomputed_path("pretrained_pkl_pcd_pts_1000_128_rndm_noise_t2p0_pcd_pts_condensed.pt"),
        "const_truncate0.25": _precomputed_path("pretrained_pkl_pcd_pts_1000_128_const_noise_t025_pcd_pts_condensed.pt"),
        "rwds_unseen": _precomputed_path("pretrained_pkl_pcd_pts_unseen_200k_to_300k_128_pcd_pts_condensed.pt"),
        "seen": _precomputed_path("pretrained_pkl_pcd_pts_1000_128_const_noise_pcd_pts_condensed.pt"),
    }

    def rescale_im(dmap):
        rmin = 2.25
        rmax = 3.3
        dm_min = -1.0
        dmap = (((dmap - rmin) / (rmax - rmin)) * 2) - 1
        dmap[dmap < dm_min] = dm_min
        dmap[dmap > 1.0] = 1.0
        return dmap

    means = [0.3674, 0.1205, 0.3750]
    stds = [0.5739, 0.3659, 0.5731]

    from torchvision.transforms import v2

    dmap_transforms_facenet_160 = v2.Compose(
        [
            # v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            # v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
            # ...
            v2.Resize(
                size=(160, 160),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),  # Or Resize(antialias=True)
            v2.Lambda(lambd=rescale_im),
            v2.Normalize(mean=means, std=stds),
            # v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    reward_model.affine_offset = torch.nn.Parameter(torch.tensor(0.0))
    # ie if the affine offset not set properly...
    if reward_model.affine_offset == 0.0:
        # getting normalisation terms
        rescale_size = 160
        # Legacy example path before refactor:
        # pt_fn = _precomputed_path("pretrained_pkl_dmaps_1000_128_const_noise_dmaps_condensed.pt")
        pt_fn = fn_list_dict["rwds_unseen"]
        combined_tensor = torch.load(pt_fn, map_location=torch.device("cuda"))
        # norm_min=-1.0
        # dmap=normalise_dmap_vals(combined_tensor,rendering_options=ffhq_rendering_options,min=norm_min,check_lims=True)
        # dxm=F.interpolate(dmap,size=(rescale_size,rescale_size),mode=da.interpolation_mode)
        dx_tensor = TensorDataset(combined_tensor)
        loader = DataLoader(dx_tensor, batch_size=64, drop_last=False)
        rwds = np.array([])
        with torch.no_grad():
            for d in loader:
                if type(d) == list and len(d) == 1:
                    d = d[0]
                # dmap=da.vggface2_to_512_model(dmap_transforms_facenet_160(d[0]))
                rwd = reward_model(d).detach().cpu().numpy()
                rwds = np.append(rwds, rwd.flatten())

        mean = np.mean(rwds.flatten())
        std = np.std(rwds.flatten())
        normalisation_terms = dict(mean=mean, std=std)
        # set the norm terms

        reward_model.affine_offset = torch.nn.Parameter(torch.tensor(normalisation_terms["mean"]))
        # optimal_rwd_model.affine_scale=torch.nn.Parameter(torch.tensor(normalisation_terms['std']))

        print("afine offset (mean)")
        print(reward_model.affine_offset)

        # optimal_rwd_model.affine_offset=torch.nn.Parameter(torch.tensor(normalisation_terms['mean']))

    # import torch

    # # dmaps_rndm_t2p0 = torch.load(_precomputed_path("pretrained_pkl_dmaps_1000_128_rndm_noise_t2p0.pt"))
    # # dmaps_const_t025 = torch.load(_precomputed_path("pretrained_pkl_dmaps_1000_128_const_noise_t025.pt"))

    # # fn = _precomputed_path("pretrained_pkl_dmaps_unseen_200k_to_300k_128.pt")

    # fn_list = [
    #     _precomputed_path("pretrained_pkl_dmaps_1000_128_rndm_noise_t2p0.pt"),
    #     _precomputed_path("pretrained_pkl_dmaps_1000_128_const_noise_t025.pt"),
    #     _precomputed_path("pretrained_pkl_dmaps_unseen_200k_to_300k_128.pt"),
    #     _precomputed_path("pretrained_pkl_dmaps_1000_128_const_noise.pt"),
    # ]

    # for fn in fn_list:

    #     sdict=torch.load(fn)

    #     dmaps=[sdict[s]['dmap'] for s in sdict.keys()]

    #     dmp_cuda=torch.tensor(dmaps,device=torch.device('cuda'))

    #     torch.save(obj=dmp_cuda.view(-1,3,128,128),f=fn.replace('.pt','_dmaps_condensed.pt'))

    #     print(f'done fn {fn}')

    # def get_rwds_vggface2(reward_model,seeds_dict_unseen,da,check_lims=True):
    #     rwds_unseen=[]

    #     for s in seeds_dict_unseen.keys():
    #         dmap=seeds_dict_unseen[s]['dmap']
    #         dmap=torch.tensor(dmap).to(device)
    #         norm_min=-1.0
    #         dmap=normalise_dmap_vals(dmap,rendering_options=ffhq_rendering_options,min=norm_min,check_lims=check_lims)
    #         dmap=rescale_dmap_single(dmap,160,mode=da.interpolation_mode)
    #         dmap=da.vggface2_to_512_model(dmap)
    #         rwd=reward_model(dmap,).item()

    #         rwds_unseen.append(rwd)

    #         if len(rwds_unseen)%100==0:
    #             print(len(rwds_unseen))

    #     return rwds_unseen

    # def normalise_dmap_vals_plots(dmap,rendering_options,min=0.0,check_lims=True):
    #     rmin=rendering_options['ray_start']
    #     rmax=rendering_options['ray_end']
    #     if min==0.0:
    #         dmap=((dmap-rmin)/(rmax-rmin))
    #         dm_min=0.0
    #     elif min==-1.0:
    #         dmap=(((dmap-rmin)/(rmax-rmin))*2)-1
    #         dm_min=-1.0

    #     dmap[dmap<dm_min]=dm_min
    #     dmap[dmap>1.0]=1.0
    #     #dmap.clamp_(min=dm_min,max=1.0) #clamp between vals

    #     #if check_lims:
    #     #    assert dmap.max()<=1.0
    #     #    assert dmap.min()>=dm_min
    #     return(dmap)

    named_rwds = {}

    rescale_size = 160

    for k in fn_list_dict.keys():
        pt_fn = fn_list_dict[k]
        combined_tensor = torch.load(pt_fn, map_location=torch.device("cuda"))
        # norm_min=-1.0
        # dmap=normalise_dmap_vals_plots(combined_tensor,rendering_options=ffhq_rendering_options,min=norm_min,check_lims=True) #try _plots without @torch.compile call
        # dxm=F.interpolate(dmap,size=(rescale_size,rescale_size),mode=da.interpolation_mode)
        dx_tensor = TensorDataset(combined_tensor)
        loader = DataLoader(dx_tensor, batch_size=64, drop_last=False)

        rwds = np.array([])
        with torch.no_grad():
            for d in loader:
                if type(d) == list and len(d) == 1:
                    d = d[0]
                # dmap=da.vggface2_to_512_model(dmap_transforms_facenet_160(d[0]))
                rwd = (
                    reward_model(
                        d,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                rwds = np.append(rwds, rwd.flatten())

        named_rwds[k] = rwds

    gaussian_samples = get_gaussian_mean_rwds(named_rwds["seen"])

    named_rwds["gaussian_samples"] = gaussian_samples

    # dict_fn = _precomputed_path("pretrained_pkl_dmaps_1000_128_const_noise.pt")
    # seeds_dict_orig = torch.load(dict_fn)

    # rwds_seen=get_rwds_vggface2(reward_model,seeds_dict_orig,da)

    # dmaps_rndm_t2p0 = torch.load(_precomputed_path("pretrained_pkl_dmaps_1000_128_rndm_noise_t2p0.pt"))
    # dmaps_const_t025 = torch.load(_precomputed_path("pretrained_pkl_dmaps_1000_128_const_noise_t025.pt"))
    # seeds_dict_unseen = torch.load(_precomputed_path("pretrained_pkl_dmaps_unseen_200k_to_300k_128.pt"))

    model_dir = os.path.join("/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/000_RLHF_AM/rlhf_reward_models", mname_str)

    # #reward_model.affine_scale=nn.Parameter(torch.tensor(1.0))
    # #reward_model.affine_offset=nn.Parameter(torch.tensor(0.0))

    # rwds_unseen=get_rwds_vggface2(reward_model,seeds_dict_unseen,da)
    # rwds_t025=get_rwds_vggface2(reward_model,dmaps_const_t025,da)
    # dmaps_rndm_t2p0=get_rwds_vggface2(reward_model,dmaps_rndm_t2p0,da)

    # names=['seen','gaussian_samples','const_truncate0.25','rndm_truncate_2.0','rwds_unseen']

    rwds_dfs = []

    # rwds=[rwds_seen,gaussian_samples,rwds_t025,dmaps_rndm_t2p0,rwds_unseen]

    # names=['seen','gaussian_samples','const_truncate0.25','rndm_truncate_2.0','rwds_unseen']

    for r in named_rwds.keys():
        rwds_dfs.append(rwd_to_df(named_rwds[r], r))

    # for r,n in zip(rwds,names):
    #     rwds_dfs.append(rwd_to_df(r,n))

    rwds_df = pd.concat(rwds_dfs)

    rwds_df.columns = ["rwd", "condition"]

    current_rwd_ims = glob.glob(os.path.join(model_dir, "rwds_seen_unseen*.png"))
    next_idx = len(current_rwd_ims)

    rwds_df.to_csv("rwds_df.csv")

    def normalize(arr, t_min, t_max):
        norm_arr = []
        diff = t_max - t_min
        diff_arr = max(arr) - min(arr)
        for i in arr:
            temp = (((i - min(arr)) * diff) / diff_arr) + t_min
            norm_arr.append(temp)
        return norm_arr

    def kdeplot(data, **kwargs):
        ax = sns.kdeplot(data, **kwargs)
        if "fill" in kwargs.keys() and kwargs["fill"] == True:
            path = ax.collections[0].get_paths()
            ys = normalize(path[0].vertices[:, 1], 0, 1)
            path[0].vertices[:, 1] = ys
        else:
            line = ax.lines[0]
            line.set_ydata(normalize(line.get_ydata(), 0, 1))
        ax.set_ylim(0, 1.05)
        ax.autoscale_view()

    # g.map(kdeplot, "x",bw_adjust=.5, clip_on=False,
    #      fill=True, alpha=1, linewidth=1.5)
    # g.map(kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.5)

    g = sns.FacetGrid(rwds_df, col="condition", height=2.5, col_wrap=3)

    fn_sun = os.path.join(model_dir, f"rwds_seen_unseen_kde_custom_{next_idx}.png")

    g.map(kdeplot, "rwd")
    g.savefig(fn_sun)
    plt.close(g.fig)

    # with pd.option_context('mode.use_inf_as_na', True):
    #     g = sns.displot(rwds_df, x="rwd", hue="condition", kind="kde", fill=True)
    #     fn_all=os.path.join(model_dir,f'rwds_all_{next_idx}.png')
    #     g.savefig(fn_all)
    #     plt.close(g.fig)

    # with pd.option_context('mode.use_inf_as_na', True):
    #     g = sns.displot(rwds_df, x="rwd", hue="condition", kind="hist", fill=True)
    #     fn_all=os.path.join(model_dir,f'rwds_all_hist_{next_idx}.png')
    #     g.savefig(fn_all)

    #     plt.close(g.fig)

    for cond in [
        "seen",
        "gaussian_samples",
        "const_truncate0.25",
        "rndm_truncate_2.0",
        "rwds_unseen",
    ]:
        rwds_df_sub = rwds_df[rwds_df.condition == cond]
        mean_of_it = rwds_df_sub.rwd.values.mean()
        mean_of_it = f"{mean_of_it:.4f}".replace(".", "_")
        print(f"cond: {cond} mean rwd: {mean_of_it}")
        with pd.option_context("mode.use_inf_as_na", True):
            fig = sns.displot(rwds_df_sub, x="rwd", hue="condition", kind="hist", fill=True)
            fn_all = os.path.join(model_dir, f"rwds_{cond}_hist_{next_idx}_mean_{mean_of_it}.png")
            fig.savefig(fn_all)
            plt.close(fig.fig)

        with pd.option_context("mode.use_inf_as_na", True):
            fig = sns.displot(rwds_df_sub, x="rwd", hue="condition", kind="kde", fill=True)
            fn_all = os.path.join(model_dir, f"rwds_{cond}_kde_{next_idx}_mean_{mean_of_it}.png")
            fig.savefig(fn_all)
            plt.close(fig.fig)

    reward_model.affine_offset = torch.nn.Parameter(torch.tensor(0.0))

    return


def plot_rwd_dists_xtra(reward_model, da):
    mname_str = da.MODEL_NAME

    # reward_model,da=load_rwd_mdl(f'/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/000_RLHF_AM/optimal_reward_models/{mname_str}.pkl')

    if not hasattr(da, "interpolation_mode"):
        da.interpolation_mode = "nearest"  # quick hack
    # if 'interpolation_mode' not in a:
    #    print('hello')

    #    mname_str=['rwd_model_3dmap_facenet512_new_data2_10112_w_goodmesh_bilinear_epo_30_bs_256_hls_256_nh_4'][0]
    # state_dict=torch.load(os.path.join(f'/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/000_RLHF_AM/rlhf_reward_models/{mname_str}/model_state_epo_19.pth'))
    # reward_model.load_state_dict(state_dict)
    # reward_model.eval()

    print("affine scale (std)")
    print(reward_model.affine_scale)

    # set affine scale to 1.0

    reward_model.affine_scale = torch.nn.Parameter(torch.tensor(1.0))

    print("affine scale (std) AFTER set = 1.0")
    print(reward_model.affine_scale)

    def rescale_im(dmap):
        rmin = 2.25
        rmax = 3.3
        dm_min = -1.0
        dmap = (((dmap - rmin) / (rmax - rmin)) * 2) - 1
        dmap[dmap < dm_min] = dm_min
        dmap[dmap > 1.0] = 1.0
        return dmap

    means = [0.3674, 0.1205, 0.3750]
    stds = [0.5739, 0.3659, 0.5731]

    from torchvision.transforms import v2

    dmap_transforms_facenet_160 = v2.Compose(
        [
            # v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            # v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
            # ...
            v2.Resize(
                size=(160, 160),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),  # Or Resize(antialias=True)
            v2.Lambda(lambd=rescale_im),
            v2.Normalize(mean=means, std=stds),
            # v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    reward_model.affine_offset = torch.nn.Parameter(torch.tensor(0.0))
    # ie if the affine offset not set properly...
    if reward_model.affine_offset == 0.0:
        # getting normalisation terms
        rescale_size = 160
        # Legacy example path before refactor:
        # pt_fn = _precomputed_path("pretrained_pkl_dmaps_1000_128_const_noise_dmaps_condensed.pt")
        pt_fn = _precomputed_path("pretrained_pkl_dmaps_1000_128_const_noise_dmaps_condensed.pt")
        combined_tensor = torch.load(pt_fn, map_location=torch.device("cuda"))
        # norm_min=-1.0
        # dmap=normalise_dmap_vals(combined_tensor,rendering_options=ffhq_rendering_options,min=norm_min,check_lims=True)
        # dxm=F.interpolate(dmap,size=(rescale_size,rescale_size),mode=da.interpolation_mode)
        dx_tensor = TensorDataset(combined_tensor)
        loader = DataLoader(dx_tensor, batch_size=64, drop_last=False)
        rwds = np.array([])
        with torch.no_grad():
            for d in loader:
                dmap = da.vggface2_to_512_model(dmap_transforms_facenet_160(d[0]))
                rwd = (
                    reward_model(
                        dmap,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                rwds = np.append(rwds, rwd.flatten())

        mean = np.mean(rwds.flatten())
        std = np.std(rwds.flatten())
        normalisation_terms = dict(mean=mean, std=std)
        # set the norm terms

        reward_model.affine_offset = torch.nn.Parameter(torch.tensor(normalisation_terms["mean"]))
        # optimal_rwd_model.affine_scale=torch.nn.Parameter(torch.tensor(normalisation_terms['std']))

        print("afine offset (mean)")
        print(reward_model.affine_offset)

        # optimal_rwd_model.affine_offset=torch.nn.Parameter(torch.tensor(normalisation_terms['mean']))

    fn_list = [
        _precomputed_path("pretrained_pkl_dmaps_1000_128_rndm_noise_t2p0.pt"),
        _precomputed_path("pretrained_pkl_dmaps_1000_128_const_noise_t025.pt"),
        _precomputed_path("pretrained_pkl_dmaps_unseen_200k_to_300k_128.pt"),
        _precomputed_path("pretrained_pkl_dmaps_1000_128_const_noise.pt"),
    ]

    fn_list_dict = {
        "rndm_truncate_2.0": _precomputed_path("pretrained_pkl_dmaps_1000_128_rndm_noise_t2p0_dmaps_condensed.pt"),
        "const_truncate0.25": _precomputed_path("pretrained_pkl_dmaps_1000_128_const_noise_t025_dmaps_condensed.pt"),
        "rwds_unseen": _precomputed_path("pretrained_pkl_dmaps_unseen_200k_to_300k_128_dmaps_condensed.pt"),
        "seen": _precomputed_path("pretrained_pkl_dmaps_1000_128_const_noise_dmaps_condensed.pt"),
    }

    named_rwds = {}

    rescale_size = 160

    for k in fn_list_dict.keys():
        pt_fn = fn_list_dict[k]
        combined_tensor = torch.load(pt_fn, map_location=torch.device("cuda"))
        # norm_min=-1.0
        # dmap=normalise_dmap_vals_plots(combined_tensor,rendering_options=ffhq_rendering_options,min=norm_min,check_lims=True) #try _plots without @torch.compile call
        # dxm=F.interpolate(dmap,size=(rescale_size,rescale_size),mode=da.interpolation_mode)
        dx_tensor = TensorDataset(combined_tensor)
        loader = DataLoader(dx_tensor, batch_size=64, drop_last=False)

        rwds = np.array([])
        with torch.no_grad():
            for d in loader:
                dmap = da.vggface2_to_512_model(dmap_transforms_facenet_160(d[0]))
                rwd = (
                    reward_model(
                        dmap,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                rwds = np.append(rwds, rwd.flatten())

        named_rwds[k] = rwds

    gaussian_samples = get_gaussian_mean_rwds(named_rwds["seen"])

    named_rwds["gaussian_samples"] = gaussian_samples
    model_dir = os.path.join("/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/000_RLHF_AM/rlhf_reward_models", mname_str)

    rwds_dfs = []

    for r in named_rwds.keys():
        rwds_dfs.append(rwd_to_df(named_rwds[r], r))

    rwds_df = pd.concat(rwds_dfs)
    rwds_df.columns = ["rwd", "condition"]
    current_rwd_ims = glob.glob(os.path.join(model_dir, "rwds_seen_unseen*.png"))
    next_idx = len(current_rwd_ims)
    rwds_df.to_csv("rwds_df.csv")

    def normalize(arr, t_min, t_max):
        norm_arr = []
        diff = t_max - t_min
        diff_arr = max(arr) - min(arr)
        for i in arr:
            temp = (((i - min(arr)) * diff) / diff_arr) + t_min
            norm_arr.append(temp)
        return norm_arr

    def kdeplot(data, **kwargs):
        ax = sns.kdeplot(data, **kwargs)
        if "fill" in kwargs.keys() and kwargs["fill"] == True:
            path = ax.collections[0].get_paths()
            ys = normalize(path[0].vertices[:, 1], 0, 1)
            path[0].vertices[:, 1] = ys
        else:
            line = ax.lines[0]
            line.set_ydata(normalize(line.get_ydata(), 0, 1))
        ax.set_ylim(0, 1.05)
        ax.autoscale_view()

    g = sns.FacetGrid(rwds_df, col="condition", height=2.5, col_wrap=3)
    fn_sun = os.path.join(model_dir, f"rwds_seen_unseen_kde_custom_{next_idx}.png")

    g.map(kdeplot, "rwd")
    g.savefig(fn_sun)
    plt.close(g.fig)

    # with pd.option_context('mode.use_inf_as_na', True):
    #     g = sns.displot(rwds_df, x="rwd", hue="condition", kind="kde", fill=True)
    #     fn_all=os.path.join(model_dir,f'rwds_all_{next_idx}.png')
    #     g.savefig(fn_all)
    #     plt.close(g.fig)

    # with pd.option_context('mode.use_inf_as_na', True):
    #     g = sns.displot(rwds_df, x="rwd", hue="condition", kind="hist", fill=True)
    #     fn_all=os.path.join(model_dir,f'rwds_all_hist_{next_idx}.png')
    #     g.savefig(fn_all)

    #     plt.close(g.fig)

    for cond in [
        "seen",
        "gaussian_samples",
        "const_truncate0.25",
        "rndm_truncate_2.0",
        "rwds_unseen",
    ]:
        rwds_df_sub = rwds_df[rwds_df.condition == cond]
        mean_of_it = rwds_df_sub.rwd.values.mean()
        mean_of_it = f"{mean_of_it:.4f}".replace(".", "_")
        print(f"cond: {cond} mean rwd: {mean_of_it}")
        with pd.option_context("mode.use_inf_as_na", True):
            fig = sns.displot(rwds_df_sub, x="rwd", hue="condition", kind="hist", fill=True)
            fn_all = os.path.join(model_dir, f"rwds_{cond}_hist_{next_idx}_mean_{mean_of_it}.png")
            fig.savefig(fn_all)
            plt.close(fig.fig)

        with pd.option_context("mode.use_inf_as_na", True):
            fig = sns.displot(rwds_df_sub, x="rwd", hue="condition", kind="kde", fill=True)
            fn_all = os.path.join(model_dir, f"rwds_{cond}_kde_{next_idx}_mean_{mean_of_it}.png")
            fig.savefig(fn_all)
            plt.close(fig.fig)

    reward_model.affine_offset = torch.nn.Parameter(torch.tensor(0.0))

    return


def train_rwd_model_from_scratch_with_multi_loader(
    da,
    optimizer,
    reward_model,
    train_loader,
    val_loader,
    test_loader,
    plot_dists_as_train=False,
    n_epochs=20,
    dset_dict=None,
):
    model_dir = da.model_dir
    # create a summary writer object for train_log
    train_writer = SummaryWriter(model_dir)

    running_val_losses = []
    estop_threshold = 20
    estop_counter = 0
    for epoch in range(n_epochs):
        # print(f'epoch: {epoch}')
        reward_model.train()
        total_loss = 0
        train_losses_total = 0
        train_pred = []
        val_pred = []
        test_pred = []
        for train_batch in tqdm(train_loader):
            optimizer.zero_grad()
            # train_batch.cuda()
            loss, vals = da.batch_format_func(reward_model, train_batch, da)
            loss.backward()
            optimizer.step()
            train_losses_total += loss
            train_pred.append(vals)  # .detach().cpu())

        reward_model.eval()
        vlosses = []
        with torch.no_grad():
            for batch in tqdm(val_loader):
                vl, vals = da.batch_format_func(reward_model, batch, da)
                vlosses.append(vl)
                val_pred.append(vals)  # .detach().cpu())

        tlosses = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                vl, vals = da.batch_format_func(reward_model, batch, da)
                tlosses.append(vl)
                test_pred.append(vals)  # .detach().cpu())

        val_loss = torch.mean(torch.stack(vlosses))

        print(f"epoch: {epoch}\tval_loss: {val_loss:.6f}")  # \t\tval_sym: {val_sym:.3f}\tval_mag: {val_mag:.3f}\tval_abs: {int(val_abs)}\tval_sixnorm: {val_sixnorm:.3f}')

        train_writer.add_scalar("vloss_epo", val_loss.item(), epoch)
        train_writer.add_scalar("train_loss_total_epo", train_losses_total, epoch)

        running_val_losses.append(val_loss)

        if val_loss == min(running_val_losses):
            torch.save(reward_model.state_dict(), os.path.join(model_dir, "model_state_dict_best.pth"))  # save model

        torch.save(reward_model.state_dict(), os.path.join(model_dir, f"model_state_epo_{epoch}.pth"))

        if epoch > 10:
            if running_val_losses[-1] > running_val_losses[-2]:
                estop_counter += 1
            else:
                estop_counter = 0
            if estop_counter > estop_threshold:
                print("early stopping")
                break

        # val_pred
        # fig,normalisation_terms,optimal_rwd_model=model_evaluate_with_plot(da.model_dir,reward_model,test_loader,train_loader,val_loader,da)
        # evaluate_model_return_precomputed_pc_correct
        if plot_dists_as_train:
            pred_pairs = {
                "train": convert_scores_to_df(train_pred),
                "val": convert_scores_to_df(val_pred),
                "test": convert_scores_to_df(test_pred),
            }

            indiv_scores = {
                "train": retrieve_flat_rwd_scores(train_pred),
                "val": retrieve_flat_rwd_scores(val_pred),
                "test": retrieve_flat_rwd_scores(test_pred),
            }

            fig, pcorrect = evaluate_model_return_precomputed_pc_correct(pred_pairs=pred_pairs, indiv_scores=indiv_scores, dargs=da)
            # fig,nterms,pcorrect=evaluate_model_return_precomputed_pc_correct(reward_model,train_pred,val_pred,test_pred,da)
            fig.savefig(
                os.path.join(da.model_dir, f"reward_model_epoch_{epoch}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            fig.clf()
            fig.clear()
            plt.close(fig)
            plt.close("all")

            tmp_eval_file = os.path.join(da.model_dir, "tmp_eval_file.txt")

            with open(tmp_eval_file, "a") as f:
                print(f"epoch: {epoch}\tval_loss: {val_loss:.6f}", file=f)  # \t\tval_sym: {val_sym:.3f}\tval_mag: {val_mag:.3f}\tval_abs: {int(val_abs)}\tval_sixnorm: {val_sixnorm:.3f}')

            os.rename(
                dst=os.path.join(da.model_dir, f"reward_model_epoch_{epoch}.txt"),
                src=tmp_eval_file,
            )

            for k in pcorrect.keys():
                train_writer.add_scalar(k, pcorrect[k], epoch)
        plotting_extra_rwd_dist = False
        # if da.model_class=='dmap3':
        if plotting_extra_rwd_dist:
            if da.model_class == "rwd_model_pointnet2":
                plot_rwd_dists_pcd(reward_model, da)
            else:
                # depth maps but others not implementd
                plot_rwd_dists_xtra(reward_model, da)

        # get individual scores

        dtype = ["val", "test", "train"]

        for dt in dtype:
            dsd = dset_dict[dt].dsets[0]
            seeds = dsd.get_sorted_unique_seeds()

            pt_data_dict = {k: dsd.return_single_example_by_seed(k) for k in seeds}
            ptd = list(pt_data_dict.items())

            keys = [p[0] for p in ptd]
            vals = [p[1] for p in ptd]

            vals = torch.cat(vals).squeeze(1).cuda()

            tload = torch.utils.data.DataLoader(vals, batch_size=64, drop_last=False, shuffle=False)

            gfv = []
            for tbatch in tqdm(tload):
                with torch.no_grad():
                    rwd = reward_model.forward(tbatch)
                    gfv.append(rwd)

            rwds = torch.cat(gfv)

            ttk = torch.tensor(keys).view(rwds.shape).cuda()
            ordered_seeds = torch.cat((ttk, rwds), 1)

            ordered_seeds_pd = pd.DataFrame(ordered_seeds.cpu().numpy())

            ordered_seeds_pd.columns = ["seed", "rwd"]

            ordered_seeds_pd = ordered_seeds_pd.sort_values("rwd", ascending=False).astype({"seed": "int32"})

            ordered_seeds_pd.to_csv(os.path.join(da.model_dir, f"{dt}_type_df_results_epoch_{epoch}.csv"))

            # read the joined_all dataframe from a csv file
            joined_all = pd.read_csv(os.path.join(da.model_dir, f"{dt}_type_df_results_epoch_{epoch}.csv"))

            # sort the dataframe by the mean column in descending order
            sorted_joined_all = joined_all.sort_values(by="rwd", ascending=False)

            # extract the top 5 seeds and loss values
            top_5_seeds = sorted_joined_all["seed"].head(5).tolist()
            top_5_losses = sorted_joined_all["rwd"].head(5).tolist()

            seedmeshes = [os.path.join(ddir_func(s), f"mesh_cat_s_{s}.jpg") for s in top_5_seeds]
            pics = [os.path.join(ddir_func(s), f"triple_rgb_s_{s}_1.jpg") for s in top_5_seeds]
            overall_images = plot_overall_images(seedmeshes, pics)
            grid_img = torchvision.utils.make_grid(overall_images, nrow=5)
            out_fn_t = os.path.join(da.model_dir, f"{dt}_ top_5_{epoch}.jpg")
            PIL.Image.fromarray(grid_img.numpy().transpose(1, 2, 0)).save(out_fn_t)

            top_5_seeds = sorted_joined_all["seed"].tail(5).tolist()
            top_5_losses = sorted_joined_all["rwd"].tail(5).tolist()

            # print("Top 5 seeds:", top_5_seeds)
            # print("Top 5 losses:", top_5_losses)
            seedmeshes = [os.path.join(ddir_func(s), f"mesh_cat_s_{s}.jpg") for s in top_5_seeds]
            pics = [os.path.join(ddir_func(s), f"triple_rgb_s_{s}_1.jpg") for s in top_5_seeds]
            overall_images = plot_overall_images(seedmeshes, pics)
            grid_img = torchvision.utils.make_grid(overall_images, nrow=5)
            out_fn_b = os.path.join(da.model_dir, f"{dt}_bottom_5_{epoch}.jpg")
            PIL.Image.fromarray(grid_img.numpy().transpose(1, 2, 0)).save(out_fn_b)

            # get top five which are not from the good mesh category...
            joined_all = pd.read_csv(os.path.join(da.model_dir, f"{dt}_type_df_results_epoch_{epoch}.csv"))

            # sort the dataframe by the mean column in descending order
            sorted_joined_all = joined_all.sort_values(by="rwd", ascending=False)
            sorted_joined_all = sorted_joined_all[sorted_joined_all.seed < 100000]  # 100000 is starting seed for the good meshes
            # extract the top 5 seeds and loss values
            top_5_seeds = sorted_joined_all["seed"].head(5).tolist()
            top_5_losses = sorted_joined_all["rwd"].head(5).tolist()

            # print("Top 5 seeds:", top_5_seeds)
            # print("Top 5 losses:", top_5_losses)
            seedmeshes = [os.path.join(ddir_func(s), f"mesh_cat_s_{s}.jpg") for s in top_5_seeds]
            pics = [os.path.join(ddir_func(s), f"triple_rgb_s_{s}_1.jpg") for s in top_5_seeds]
            overall_images = plot_overall_images(seedmeshes, pics)
            grid_img = torchvision.utils.make_grid(overall_images, nrow=5)
            out_fn_ng = os.path.join(da.model_dir, f"{dt}_top_5_nogood_{epoch}.jpg")
            PIL.Image.fromarray(grid_img.numpy().transpose(1, 2, 0)).save(out_fn_ng)

            tope = PIL.Image.open(out_fn_t)
            bote = PIL.Image.open(out_fn_ng)

            w1, h1 = tope.size
            w2, h2 = bote.size

            new_img = Image.new("RGB", (max(w1, w2), h1 + h2), (255, 255, 255))
            new_img.paste(tope, (0, 0))
            new_img.paste(bote, (0, h1))

            # out_fn=os.path.join(da.model_dir,f'{dt}_top_bottom_5_{epoch}.jpg')

            tope = new_img  # .save(out_fn)

            bote = PIL.Image.open(out_fn_b)

            w1, h1 = tope.size
            w2, h2 = bote.size

            new_img = Image.new("RGB", (max(w1, w2), h1 + h2), (255, 255, 255))
            new_img.paste(tope, (0, 0))
            new_img.paste(bote, (0, h1))

            out_fn = os.path.join(da.model_dir, f"{dt}_top_bottom_5_{epoch}.jpg")

            new_img.save(out_fn)

            os.remove(out_fn_b)
            os.remove(out_fn_t)
            os.remove(out_fn_ng)

        # current_rwd_ims=glob.glob(os.path.join(da.model_dir,'reward_model_evaluation*.png'))
        # next_idx=len(current_rwd_ims)
        # save ut

    train_writer.close()



# sort joined_all by column mean, descending
import PIL
from PIL import Image

# train_paired_rwd_model_from_scratch_with_multi_loader


def get_ncomb2(k):
    if k == 2:
        return 2
    if k == 3:
        return 3
    if k == 4:
        return 6
    if k == 5:
        return 10
    if k == 6:
        return 15
    if k == 7:
        return 21


# def train_contrastive_reward_model(da,optimizer,reward_model,train_loader,val_loader,dset_dict,plot_dists_as_train=False,n_epochs=20):


def train_contrastive_reward_model(da, optimizer, reward_model, dset_dict, plot_dists_as_train=False, n_epochs=20):
    model_dir = da.model_dir

    model_kwargs = da.model_kwargs
    # create a summary writer object for train_log
    train_writer = SummaryWriter(model_dir)
    contrastive_loss = ContrastiveLoss()
    transform_first = ensemble_pointcloud_transforms()
    transform_second = ensemble_pointcloud_transforms()

    running_val_losses = []
    estop_threshold = 20
    estop_counter = 0

    train_dset = dset_dict["train"]  # .to_iterable_dataset()#.to(device)
    val_dset = dset_dict["val"]  # .to_iterable_dataset()#.to(device)
    # test_dset=dset_dict['test']#.to_iterable_dataset()#.to(device)
    sel_params = dset_dict["sel_params"]

    # sel_params = {'batch_size': model_kwargs['batch_size'],
    #         'shuffle': True,
    #         'num_workers': 8,
    #         'pin_memory':True}

    for epoch in range(n_epochs):
        # print(f'epoch: {epoch}')
        reward_model.train()
        total_loss = 0
        train_losses_total = 0
        train_pred = []
        val_pred = []
        test_pred = []

        xcut = 0.1 + np.random.uniform() * 0.4
        ycut = 0.1 + np.random.uniform() * 0.4

        train_dset.xcut = xcut
        train_dset.ycut = ycut
        val_dset.xcut = xcut
        val_dset.ycut = ycut

        # bs_ratio=1
        bs_ratio = 2

        ss = np.random.randint(3)

        if epoch == 0:
            ss = 0

        if ss == 0:
            train_dset.n_points = 2048
            val_dset.n_points = 2048
            sel_params["batch_size"] = 16 * bs_ratio
        if ss == 1:
            train_dset.n_points = 4096
            val_dset.n_points = 4096
            sel_params["batch_size"] = 8 * bs_ratio

        if ss == 2:
            train_dset.n_points = 8092
            val_dset.n_points = 8092
            sel_params["batch_size"] = 4 * bs_ratio

        # Generators
        # train_loader = Dataset(partition['train'], labels)
        train_loader = torch.utils.data.DataLoader(train_dset, **sel_params)

        # sel_params['batch_size']=8
        # sel_params['shuffle']=False #not set for case where we have 97% acc for test...
        # validation_set = Dataset(partition['validation'], labels)

        sel_params["batch_size"] = sel_params["batch_size"] * 8
        val_loader = torch.utils.data.DataLoader(val_dset, **sel_params)

        transform_first.n_points = train_dset.n_points
        transform_second.n_points = train_dset.n_points

        transform_first.reset_random_domains_for_train()
        transform_second.reset_random_domains_for_train()

        transform_first.degrees_pitch_range = int(min(10, epoch * 3))
        transform_first.degrees_yaw_range = int(min(60, epoch * 3))
        transform_second.degrees_pitch_range = int(min(10, epoch * 3))
        transform_second.degrees_yaw_range = int(min(60, epoch * 3))

        train_writer.add_scalar("n_points", train_dset.n_points, epoch)
        train_writer.add_scalar("batch_size", sel_params["batch_size"], epoch)
        train_writer.add_scalar("xcut", xcut, epoch)
        train_writer.add_scalar("ycut", ycut, epoch)

        for tk, train_batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            batch_c1 = torch.cat([transform_first.apply_transforms(t).permute(1, 0)[None, ...] for t in train_batch]).cuda()
            batch_c2 = torch.cat([transform_second.apply_transforms(t).permute(1, 0)[None, ...] for t in train_batch]).cuda()

            global_t1 = reward_model.forward(batch_c1)
            global_t2 = reward_model.forward(batch_c2)

            projection_t1 = reward_model.projection_head(global_t1)
            projection_t2 = reward_model.projection_head(global_t2)

            con_loss = contrastive_loss(projection_t1, projection_t2)
            # transform1=#

            # loss,vals=da.batch_format_func(reward_model,train_batch,da)
            con_loss.backward()
            optimizer.step()
            train_losses_total += con_loss
            # train_pred.append(vals)#.detach().cpu())
            # optimizer.param_groups[0]["lr"]=old_lr

            if tk % 100 == 0:
                print(f"train loss: {con_loss:.3f}")

            transform_first.resample_transform_parameters()
            transform_second.resample_transform_parameters()

        reward_model.eval()

        transform_first.set_no_random_for_validation()
        transform_second.set_no_random_for_validation()

        vlosses = []
        with torch.no_grad():
            for tk, batch in enumerate(tqdm(val_loader)):
                batch_c1 = torch.cat([transform_first.apply_transforms(t).permute(1, 0)[None, ...] for t in batch]).cuda()
                batch_c2 = torch.cat([transform_second.apply_transforms(t).permute(1, 0)[None, ...] for t in batch]).cuda()

                global_t1 = reward_model.forward(batch_c1)
                global_t2 = reward_model.forward(batch_c2)

                projection_t1 = reward_model.projection_head(global_t1)
                projection_t2 = reward_model.projection_head(global_t2)

                con_loss = contrastive_loss(projection_t1, projection_t2)
                # transform1=#

                # loss,vals=da.batch_format_func(reward_model,train_batch,da)
                # con_loss.backward()
                # optimizer.step()
                # vl,vals=da.batch_format_func(reward_model,batch,da)
                vlosses.append(con_loss)

                if tk % 100 == 0:
                    print(f"val loss: {con_loss:.3f}")

                # val_pred.append(vals)#.detach().cpu())
        # vp=[v['logits'] for v in val_pred]
        # vsv=torch.cat(vp)
        ##val_correct=torch.sum(vsv[:,0]>vsv[:,1])/vsv.shape[0]
        # print(f'validation correct: {val_correct.item():.4f}')

        # train_writer.add_scalar('vcorrect_epo', val_correct.item(), epoch)

        # tlosses=[]
        # with torch.no_grad():
        #    for batch in tqdm(test_loader):
        #        vl,vals=da.batch_format_func(reward_model,batch,da)
        #        tlosses.append(vl)
        #        test_pred.append(vals)#.detach().cpu())
        # vp=[v['logits'] for v in val_pred]
        # vsv=torch.cat(vp)
        # t_correct=torch.sum(vsv[:,0]>vsv[:,1])/vsv.shape[0]
        # print(f'test correct: {t_correct.item():.4f}')
        # train_writer.add_scalar('tcorrect_epo', t_correct.item(), epoch)

        val_loss = torch.mean(torch.stack(vlosses))

        print(f"epoch: {epoch}\tval_loss: {val_loss:.6f}")  # \t\tval_sym: {val_sym:.3f}\tval_mag: {val_mag:.3f}\tval_abs: {int(val_abs)}\tval_sixnorm: {val_sixnorm:.3f}')

        train_writer.add_scalar("vloss_epo", val_loss.item(), epoch)
        train_writer.add_scalar("train_loss_total_epo", train_losses_total, epoch)

        running_val_losses.append(val_loss)

        if val_loss == min(running_val_losses):
            torch.save(reward_model.state_dict(), os.path.join(model_dir, "model_state_dict_best.pth"))  # save model

        if epoch % 50 == 0:
            torch.save(reward_model.state_dict(), os.path.join(model_dir, f"model_state_epo_{epoch}.pth"))

        # if epoch>10:
        #     if running_val_losses[-1]>running_val_losses[-2]:
        #         estop_counter+=1
        #     else:
        #         estop_counter=0
        #     if estop_counter>estop_threshold:
        #         print('early stopping')
        #         break

        # val_pred
        # fig,normalisation_terms,optimal_rwd_model=model_evaluate_with_plot(da.model_dir,reward_model,test_loader,train_loader,val_loader,da)
        # evaluate_model_return_precomputed_pc_correct
    #     if plot_dists_as_train:

    #         sel_keys=['test','val']

    #         #if epoch%50==0:
    #         #    sel_keys.append('train')

    #         #for k in dset_dict.keys():
    #         for k in sel_keys:
    #             import pandas as pd
    #             all_indiv_e=dset_dict[k].dsets[0].return_all_indiv_examples()
    #             #all_indiv_e=[a.permute(1,0).unsqueeze(0) for a in all_indiv_e]

    #             all_indiv_e=torch.cat(all_indiv_e,0).cuda()

    #             tload=torch.utils.data.DataLoader(all_indiv_e,batch_size=64,drop_last=False,shuffle=False)

    #             gfv=[]
    #             for tbatch in tqdm(tload):
    #                 with torch.no_grad():
    #                     global_feature_vectors=reward_model.forward(tbatch)
    #                     gfv.append(global_feature_vectors)
    #             global_feature_vectors=torch.cat(gfv,0)
    #             tep=global_feature_vectors#torch.cat([torch.cat(tp['global_feature_preds'],0) for tp in test_pred])

    #             idx=torch.arange(tep.shape[0])

    #             wins=[]
    #             pwins=[]

    #             with torch.no_grad():
    #                 for i in idx:
    #                     idxe=[k for k in idx if k!=i]
    #                     lll=reward_model.forward_from_cat_global_vectors(tep[i][None,...].repeat(len(idxe),1),tep[idxe])
    #                     lll_r=reward_model.forward_from_cat_global_vectors(tep[idxe],tep[i][None,...].repeat(len(idxe),1))
    #                     win=torch.logical_and(lll[:,0] > 0.5,lll_r[:,1] > 0.5)
    #                     pwin=lll[:,0] * lll_r[:,1]

    #                     wins.append(win)
    #                     pwins.append(pwin)

    #             n_wins=[torch.sum(w) for w in wins]
    #             p_wins=[torch.mean(p) for p in pwins]

    #             n_wins=torch.hstack(n_wins).cpu().numpy()
    #             p_wins=torch.hstack(p_wins).cpu().numpy()

    #             all_unique_seeds = dset_dict[k].dsets[0].get_sorted_unique_seeds()
    #             pdr=pd.DataFrame([all_unique_seeds,n_wins,p_wins]).transpose()

    #             pdr.columns=['seed','total_g_05','mean']
    #             #mean_df.columns=['mean']
    #             #tot_df.columns=['total_g_05']
    #             #seed_df.columns=['seed']

    #             #joined_all=mean_df.join((tot_df,seed_df))
    #             norms=torch.norm(global_feature_vectors,dim=1).detach()
    #             #seeds=dset_dict[k].dsets[0].get_sorted_unique_seeds()
    #             dd={s:n.item() for s,n in zip(all_unique_seeds,norms)}
    #             pds=pd.DataFrame.from_dict(dd,orient='index').reset_index()
    #             pds.columns=['seed','l2norm_global_embedding']
    #             joined_all=pdr.merge(pds,on='seed')

    #             joined_all.to_csv(os.path.join(da.model_dir,f'{k}_df_results_epoch_{epoch}.csv'))

    #             type_of_comparison='perm'
    #             # mean_prob_success=[]
    #             # sum_g_05=[]

    #             # #shuffle torch.randperm(tep.shape[0])
    #             # idx = torch.randperm(tep.shape[0]).numpy()
    #             # comparison_results_p={k:[] for k in idx}

    #             # for type_of_comparison in ['perm','combo']:

    #             #     if type_of_comparison=='perm':
    #             #         perms = itertools.permutations(idx, r=2)

    #             #     else:
    #             #         perms = itertools.combinations(idx,r=2)

    #             #     #perms = itertools.permutations(idx, r=2)

    #             #     ppp=[p for p in perms] #581406
    #             #     ppp=np.array(ppp)

    #             #     fi=[tep[f].unsqueeze(0) for f in ppp[:,0]]
    #             #     si=[tep[f].unsqueeze(0) for f in ppp[:,1]]

    #             #     fi=torch.cat(fi,0)
    #             #     si=torch.cat(si,0)

    #             #     # fi=[tep[761].unsqueeze(0) for f in range(761)]
    #             #     # si=[tep[f].unsqueeze(0) for f in range(761)]

    #             #     # fi=torch.cat(fi,0)
    #             #     # si=torch.cat(si,0)

    #             #     with torch.no_grad():
    #             #         rls=reward_model.forward_from_cat_global_vectors(fi,si)

    #             #     for i in idx:
    #             #         current_results=rls[torch.tensor(ppp)==i].flatten().cpu().numpy()
    #             #         comparison_results_p[i]=current_results

    #             #     THRESH=0.5
    #             #     sum_g_50_pc={i:(comparison_results_p[i]>THRESH).sum() for i in comparison_results_p.keys()}
    #             #     sum_g_50_pc_vals=list(sum_g_50_pc.values())
    #             #     mean_score={i:(comparison_results_p[i]).mean() for i in comparison_results_p.keys()}
    #             #     mean_score_vals=list(mean_score.values())

    #             #    # all_batch_seeds=dset_dict[k].dsets[0].all_combined_rankings_ordered
    #             #    #all_batch_seeds = np.concatenate(all_batch_seeds)
    #             #     #all_unique_seeds = np.unique(all_batch_seeds).astype(np.int32)
    #             #     all_unique_seeds = dset_dict[k].dsets[0].get_sorted_unique_seeds()
    #             #     import pandas as pd

    #             #     mean_df=pd.DataFrame.from_dict(mean_score,orient='index')
    #             #     tot_df=pd.DataFrame.from_dict(sum_g_50_pc,orient='index')

    #             #     aus={i:all_unique_seeds[i] for i in idx}

    #             #     seed_df=pd.DataFrame.from_dict(aus,orient='index')

    #             #     mean_df.columns=['mean']
    #             #     tot_df.columns=['total_g_05']
    #             #     seed_df.columns=['seed']

    #             #     joined_all=mean_df.join((tot_df,seed_df))
    #             #     norms=torch.norm(global_feature_vectors,dim=1).detach()
    #             #     seeds=dset_dict[k].dsets[0].get_sorted_unique_seeds()
    #             #     dd={s:n.item() for s,n in zip(seeds,norms)}
    #             #     pds=pd.DataFrame.from_dict(dd,orient='index').reset_index()
    #             #     pds.columns=['seed','l2norm_global_embedding']

    #             #     joined_all=joined_all.merge(pds,on='seed')
    #             #     joined_all.to_csv(os.path.join(da.model_dir,f'{k}_type_{type_of_comparison}_df_results_epoch_{epoch}.csv'))

    #             N_MESHES=10
    #             # read the joined_all dataframe from a csv file
    #             joined_all = pd.read_csv(os.path.join(da.model_dir,f'{k}_df_results_epoch_{epoch}.csv')).astype({'seed': 'int32'})

    #             # sort the dataframe by the mean column in descending order
    #             sorted_joined_all = joined_all.sort_values(by='mean', ascending=False)

    #             # extract the top 5 seeds and loss values
    #             top_5_seeds = sorted_joined_all['seed'].head(N_MESHES).tolist()
    #             top_5_losses = sorted_joined_all['mean'].head(N_MESHES).tolist()

    #             seedmeshes=[os.path.join(ddir_func(s),f'mesh_cat_s_{s}.jpg') for s in top_5_seeds]
    #             pics=[os.path.join(ddir_func(s),f'triple_rgb_s_{s}_1.jpg') for s in top_5_seeds]
    #             overall_images=plot_overall_images(seedmeshes,pics)
    #             grid_img = torchvision.utils.make_grid(overall_images, nrow=N_MESHES)
    #             out_fn_t=os.path.join(da.model_dir,f'{k}_type_{type_of_comparison}_ top_5_{epoch}.jpg')
    #             PIL.Image.fromarray(grid_img.numpy().transpose(1,2,0)).save(out_fn_t)

    #             top_5_seeds = sorted_joined_all['seed'].tail(N_MESHES).tolist()
    #             top_5_losses = sorted_joined_all['mean'].tail(N_MESHES).tolist()

    #             #print("Top 5 seeds:", top_5_seeds)
    #             #print("Top 5 losses:", top_5_losses)
    #             seedmeshes=[os.path.join(ddir_func(s),f'mesh_cat_s_{s}.jpg') for s in top_5_seeds]
    #             pics=[os.path.join(ddir_func(s),f'triple_rgb_s_{s}_1.jpg') for s in top_5_seeds]
    #             overall_images=plot_overall_images(seedmeshes,pics)
    #             grid_img = torchvision.utils.make_grid(overall_images, nrow=N_MESHES)
    #             out_fn_b=os.path.join(da.model_dir,f'{k}_type_{type_of_comparison}_ bottom_5_{epoch}.jpg')
    #             PIL.Image.fromarray(grid_img.numpy().transpose(1,2,0)).save(out_fn_b)

    # #get top five which are not from the good mesh category...
    #             #joined_all = pd.read_csv(os.path.join(da.model_dir,f'{dt}_type_df_results_epoch_{epoch}.csv'))

    #             # sort the dataframe by the mean column in descending order
    #             #sorted_joined_all = joined_all.sort_values(by='rwd', ascending=False)
    #             sorted_joined_all=sorted_joined_all[sorted_joined_all.seed<100000].sort_values(by='mean', ascending=False) #100000 is starting seed for the good meshes
    #             # extract the top 5 seeds and loss values
    #             top_5_seeds = sorted_joined_all['seed'].head(N_MESHES).tolist()
    #             top_5_losses = sorted_joined_all['mean'].head(N_MESHES).tolist()

    #             #print("Top 5 seeds:", top_5_seeds)
    #             #print("Top 5 losses:", top_5_losses)
    #             seedmeshes=[os.path.join(ddir_func(s),f'mesh_cat_s_{s}.jpg') for s in top_5_seeds]
    #             pics=[os.path.join(ddir_func(s),f'triple_rgb_s_{s}_1.jpg') for s in top_5_seeds]
    #             overall_images=plot_overall_images(seedmeshes,pics)
    #             grid_img = torchvision.utils.make_grid(overall_images, nrow=N_MESHES)
    #             out_fn_ng=os.path.join(da.model_dir,f'{k}_type_{type_of_comparison}_nogood_bottom_5_{epoch}.jpg')
    #             PIL.Image.fromarray(grid_img.numpy().transpose(1,2,0)).save(out_fn_ng)

    #             tope=PIL.Image.open(out_fn_t)
    #             bote=PIL.Image.open(out_fn_ng)

    #             w1, h1 = tope.size
    #             w2, h2 = bote.size

    #             new_img = Image.new('RGB', (max(w1, w2), h1 + h2), (255, 255, 255))
    #             new_img.paste(tope, (0, 0))
    #             new_img.paste(bote, (0, h1))

    #             #out_fn=os.path.join(da.model_dir,f'{dt}_top_bottom_5_{epoch}.jpg')

    #             tope=new_img#.save(out_fn)

    #             bote=PIL.Image.open(out_fn_b)

    #             w1, h1 = tope.size
    #             w2, h2 = bote.size

    #             new_img = Image.new('RGB', (max(w1, w2), h1 + h2), (255, 255, 255))
    #             new_img.paste(tope, (0, 0))
    #             new_img.paste(bote, (0, h1))

    #             out_fn=os.path.join(da.model_dir,f'{k}_type_{type_of_comparison}_top_bottom_5_{epoch}.jpg')

    #             img=new_img
    #             img.resize((int(img.size[0]*0.4),int(img.size[1]*0.4))).save(out_fn)

    #             os.remove(out_fn_b)
    #             os.remove(out_fn_t)
    #             os.remove(out_fn_ng)

    #             # # read the joined_all dataframe from a csv file
    #             # joined_all = pd.read_csv(os.path.join(da.model_dir,f'{k}_type_{type_of_comparison}_df_results_epoch_{epoch}.csv'))

    #             # # sort the dataframe by the mean column in descending order
    #             # sorted_joined_all = joined_all.sort_values(by='mean', ascending=False)

    #             # # extract the top 5 seeds and loss values
    #             # top_5_seeds = sorted_joined_all['seed'].head(5).tolist()
    #             # top_5_losses = sorted_joined_all['mean'].head(5).tolist()

    #             # seedmeshes=[os.path.join(ddir_func(s),f'mesh_cat_s_{s}.jpg') for s in top_5_seeds]
    #             # pics=[os.path.join(ddir_func(s),f'triple_rgb_s_{s}_1.jpg') for s in top_5_seeds]
    #             # overall_images=plot_overall_images(seedmeshes,pics)
    #             # grid_img = torchvision.utils.make_grid(overall_images, nrow=5)
    #             # out_fn_t=os.path.join(da.model_dir,f'{k}_type_{type_of_comparison}_ top_5_{epoch}.jpg')
    #             # PIL.Image.fromarray(grid_img.numpy().transpose(1,2,0)).save(out_fn_t)

    #             # top_5_seeds = sorted_joined_all['seed'].tail(5).tolist()
    #             # top_5_losses = sorted_joined_all['mean'].tail(5).tolist()

    #             # #print("Top 5 seeds:", top_5_seeds)
    #             # #print("Top 5 losses:", top_5_losses)
    #             # seedmeshes=[os.path.join(ddir_func(s),f'mesh_cat_s_{s}.jpg') for s in top_5_seeds]
    #             # pics=[os.path.join(ddir_func(s),f'triple_rgb_s_{s}_1.jpg') for s in top_5_seeds]
    #             # overall_images=plot_overall_images(seedmeshes,pics)
    #             # grid_img = torchvision.utils.make_grid(overall_images, nrow=5)
    #             # out_fn_b=os.path.join(da.model_dir,f'{k}_type_{type_of_comparison}_ bottom_5_{epoch}.jpg')
    #             # PIL.Image.fromarray(grid_img.numpy().transpose(1,2,0)).save(out_fn_b)

    #             # tope=PIL.Image.open(out_fn_t)
    #             # bote=PIL.Image.open(out_fn_b)

    #             # w1, h1 = tope.size
    #             # w2, h2 = bote.size

    #             # new_img = Image.new('RGB', (max(w1, w2), h1 + h2), (255, 255, 255))
    #             # new_img.paste(tope, (0, 0))
    #             # new_img.paste(bote, (0, h1))

    #             # out_fn=os.path.join(da.model_dir,f'{k}_type_{type_of_comparison}_ top_bottom_5_{epoch}.jpg')

    #             # new_img.save(out_fn)

    #             # os.remove(out_fn_b)
    #             # os.remove(out_fn_t)
    #             # torchvision.utils.save_image(grid_img, str(REWARD_MODEL_TRAINING_DIR / "tester.jpg"))

    #             # for i in tqdm(idx):
    #             #     new_pair=torch.vstack([torch.hstack([i,i_comp]) for i_comp in idx[idx!=i]])
    #             #     permutation_idx=torch.randint(2,(len(new_pair),))
    #             #     permutation_idx=torch.nn.functional.one_hot(permutation_idx)

    #             #     pi_0=permutation_idx[:,0]
    #             #     pi_1=permutation_idx[:,1]

    #             #     first_idx=[np[pi].item() for np,pi in zip(new_pair,pi_0)]
    #             #     second_idx=[np[pi].item() for np,pi in zip(new_pair,pi_1)]

    #             #     fi=[tep[f].unsqueeze(0) for f in first_idx]
    #             #     si=[tep[f].unsqueeze(0) for f in second_idx]

    #             #     fi=torch.cat(fi,0).cuda()
    #             #     si=torch.cat(si,0).cuda()

    #             #     with torch.no_grad():
    #             #         rls=reward_model.forward_from_cat_global_vectors(fi,si)
    #             #         winning_preds=torch.masked_select(rls.cpu(),permutation_idx.to(torch.bool)).detach().flatten().cpu().numpy().mean()

    #             #         mean_prob_success.append(winning_preds)

    #             #         indiv_prob=torch.masked_select(rls.cpu(),permutation_idx.to(torch.bool)).detach().flatten().cpu().numpy()

    #             #         success_prob=indiv_prob[indiv_prob>0.5]

    #             #         sum_g_05.append(len(success_prob))
    #             mean_score_vals=pdr['mean'].values
    #             sum_g_50_pc_vals=pdr.total_g_05.values
    #             plt.clf()
    #             plt.hist(mean_score_vals,bins=100)
    #             plt.title('hist: mean_p_win_over_test_set_all_comparisons')
    #             plt.xlabel('mean probabiilltiy success')
    #             plt.ylabel('n')
    #             plt.savefig(os.path.join(da.model_dir,f'{k}_type_{type_of_comparison}_mean_p_epoch{epoch}.png'))

    #             plt.clf()
    #             plt.hist(sum_g_50_pc_vals,bins=100)
    #             plt.title('hist: sum_cases_p_greater_05_for_each_example')
    #             plt.xlabel('sum of successful comparisons')
    #             plt.ylabel('n')
    #             plt.savefig(os.path.join(da.model_dir,f'{k}_type_{type_of_comparison}_sum_wins_epoch{epoch}.png'))
    #             #tp=[tp['global_feature_preds'] for tp in train_pred]
    #             #vp=[tp['global_feature_preds'] for tp in val_pred]

    # dtype=['val','test','train']

    # for dt in dtype:
    #     dsd=dset_dict[dt].dsets[0]
    #     seeds=dsd.get_sorted_unique_seeds()

    #     pt_data_dict={k:dsd.return_single_example_by_seed(k) for k in seeds}
    #     ptd=list(pt_data_dict.items())

    #     keys=[p[0] for p in ptd]
    #     vals=[p[1] for p in ptd]

    #     vals=torch.cat(vals).squeeze(1).cuda()

    #     tload=torch.utils.data.DataLoader(vals,batch_size=64,drop_last=False,shuffle=False)

    #     gfv=[]
    #     for tbatch in tqdm(tload):
    #         with torch.no_grad():
    #             rwd=reward_model.forward(tbatch)
    #             gfv.append(rwd)

    #     rwds=torch.cat(gfv)

    #             ttk=torch.tensor(keys).view(rwds.shape).cuda()
    #             ordered_seeds=torch.cat((ttk,rwds),1)

    #             ordered_seeds_pd=pd.DataFrame(ordered_seeds.cpu().numpy())

    #             ordered_seeds_pd.columns=['seed','rwd']

    #             ordered_seeds_pd=ordered_seeds_pd.sort_values('rwd',ascending=False).astype({'seed': 'int32'})

    #             ordered_seeds_pd.to_csv(os.path.join(da.model_dir,f'{dt}_type_df_results_epoch_{epoch}.csv'))

    #             # read the joined_all dataframe from a csv file
    #             joined_all = pd.read_csv(os.path.join(da.model_dir,f'{dt}_type_df_results_epoch_{epoch}.csv'))

    #             # sort the dataframe by the mean column in descending order
    #             sorted_joined_all = joined_all.sort_values(by='rwd', ascending=False)

    #             # extract the top 5 seeds and loss values
    #             top_5_seeds = sorted_joined_all['seed'].head(5).tolist()
    #             top_5_losses = sorted_joined_all['rwd'].head(5).tolist()

    #             seedmeshes=[os.path.join(ddir_func(s),f'mesh_cat_s_{s}.jpg') for s in top_5_seeds]
    #             pics=[os.path.join(ddir_func(s),f'triple_rgb_s_{s}_1.jpg') for s in top_5_seeds]
    #             overall_images=plot_overall_images(seedmeshes,pics)
    #             grid_img = torchvision.utils.make_grid(overall_images, nrow=5)
    #             out_fn_t=os.path.join(da.model_dir,f'{dt}_ top_5_{epoch}.jpg')
    #             PIL.Image.fromarray(grid_img.numpy().transpose(1,2,0)).save(out_fn_t)

    #             top_5_seeds = sorted_joined_all['seed'].tail(5).tolist()
    #             top_5_losses = sorted_joined_all['rwd'].tail(5).tolist()

    #             #print("Top 5 seeds:", top_5_seeds)
    #             #print("Top 5 losses:", top_5_losses)
    #             seedmeshes=[os.path.join(ddir_func(s),f'mesh_cat_s_{s}.jpg') for s in top_5_seeds]
    #             pics=[os.path.join(ddir_func(s),f'triple_rgb_s_{s}_1.jpg') for s in top_5_seeds]
    #             overall_images=plot_overall_images(seedmeshes,pics)
    #             grid_img = torchvision.utils.make_grid(overall_images, nrow=5)
    #             out_fn_b=os.path.join(da.model_dir,f'{dt}_bottom_5_{epoch}.jpg')
    #             PIL.Image.fromarray(grid_img.numpy().transpose(1,2,0)).save(out_fn_b)

    # #get top five which are not from the good mesh category...
    #             joined_all = pd.read_csv(os.path.join(da.model_dir,f'{dt}_type_df_results_epoch_{epoch}.csv'))

    #             # sort the dataframe by the mean column in descending order
    #             sorted_joined_all = joined_all.sort_values(by='rwd', ascending=False)
    #             sorted_joined_all=sorted_joined_all[sorted_joined_all.seed<100000] #100000 is starting seed for the good meshes
    #             # extract the top 5 seeds and loss values
    #             top_5_seeds = sorted_joined_all['seed'].head(5).tolist()
    #             top_5_losses = sorted_joined_all['rwd'].head(5).tolist()

    #             #print("Top 5 seeds:", top_5_seeds)
    #             #print("Top 5 losses:", top_5_losses)
    #             seedmeshes=[os.path.join(ddir_func(s),f'mesh_cat_s_{s}.jpg') for s in top_5_seeds]
    #             pics=[os.path.join(ddir_func(s),f'triple_rgb_s_{s}_1.jpg') for s in top_5_seeds]
    #             overall_images=plot_overall_images(seedmeshes,pics)
    #             grid_img = torchvision.utils.make_grid(overall_images, nrow=5)
    #             out_fn_ng=os.path.join(da.model_dir,f'{dt}_top_5_nogood_{epoch}.jpg')
    #             PIL.Image.fromarray(grid_img.numpy().transpose(1,2,0)).save(out_fn_ng)

    #             tope=PIL.Image.open(out_fn_t)
    #             bote=PIL.Image.open(out_fn_ng)

    #             w1, h1 = tope.size
    #             w2, h2 = bote.size

    #             new_img = Image.new('RGB', (max(w1, w2), h1 + h2), (255, 255, 255))
    #             new_img.paste(tope, (0, 0))
    #             new_img.paste(bote, (0, h1))

    #             #out_fn=os.path.join(da.model_dir,f'{dt}_top_bottom_5_{epoch}.jpg')

    #             tope=new_img#.save(out_fn)

    #             bote=PIL.Image.open(out_fn_b)

    #             w1, h1 = tope.size
    #             w2, h2 = bote.size

    #             new_img = Image.new('RGB', (max(w1, w2), h1 + h2), (255, 255, 255))
    #             new_img.paste(tope, (0, 0))
    #             new_img.paste(bote, (0, h1))

    #             out_fn=os.path.join(da.model_dir,f'{dt}_top_bottom_5_{epoch}.jpg')

    #             new_img.save(out_fn)

    #             os.remove(out_fn_b)
    #             os.remove(out_fn_t)
    #             os.remove(out_fn_ng)

    # for all global vector, we must:
    # get pair of this global vector w.r.t all other vector, will be of size N*(N-1)...
    # randomly suffle this global pairs
    # get prediction of probability win based on reward_model.predict from global...
    # if p>0.5, count as a win (maybe)
    # get mean p?

    #     tp=train_pred[0]

    #     # pred_pairs={'train':convert_scores_to_df(train_pred),
    #     #             'val':convert_scores_to_df(val_pred),
    #     #             'test':convert_scores_to_df(test_pred)}

    #     indiv_scores={'train':retrieve_flat_rwd_scores(train_pred),
    #                 'val':retrieve_flat_rwd_scores(val_pred),
    #                 'test':retrieve_flat_rwd_scores(test_pred)}

    #     fig,pcorrect=evaluate_model_return_precomputed_pc_correct(pred_pairs=pred_pairs,indiv_scores=indiv_scores,dargs=da)
    #     #fig,nterms,pcorrect=evaluate_model_return_precomputed_pc_correct(reward_model,train_pred,val_pred,test_pred,da)
    #     fig.savefig(os.path.join(da.model_dir,f'reward_model_epoch_{epoch}.png'),dpi=300,bbox_inches='tight')
    #     fig.clf()
    #     fig.clear()
    #     plt.close(fig)
    #     plt.close('all')

    #     tmp_eval_file=os.path.join(da.model_dir,'tmp_eval_file.txt')

    #     with open(tmp_eval_file,'a') as f:
    #         print(f'epoch: {epoch}\tval_loss: {val_loss:.6f}',file=f)#\t\tval_sym: {val_sym:.3f}\tval_mag: {val_mag:.3f}\tval_abs: {int(val_abs)}\tval_sixnorm: {val_sixnorm:.3f}')

    #     os.rename(dst=os.path.join(da.model_dir,f'reward_model_epoch_{epoch}.txt'),src=tmp_eval_file)

    #     for k in pcorrect.keys():
    #         train_writer.add_scalar(k, pcorrect[k], epoch)
    # plotting_extra_rwd_dist=False
    # #if da.model_class=='dmap3':
    # if plotting_extra_rwd_dist:
    #     if da.model_class=='rwd_model_pointnet2':
    #         plot_rwd_dists_pcd(reward_model,da)
    #     else:
    #         #depth maps but others not implementd
    #         plot_rwd_dists_xtra(reward_model,da)

    # current_rwd_ims=glob.glob(os.path.join(da.model_dir,'reward_model_evaluation*.png'))
    # next_idx=len(current_rwd_ims)
    # save ut

    train_writer.close()


def train_paired_rwd_model_from_scratch_with_multi_loader(
    da,
    optimizer,
    reward_model,
    train_loader,
    val_loader,
    test_loader,
    dset_dict,
    plot_dists_as_train=False,
    n_epochs=20,
):
    if not hasattr(da, "current_epoch"):
        da.current_epoch = 0

    list_of_training_attr = [
        "running_train_losses",
        "running_val_losses",
        "running_test_correct",
        "running_val_correct",
        "running_train_correct",
    ]

    for l in list_of_training_attr:
        if not hasattr(da, l):
            exec(f"da.{l}=[]")

    model_dir = da.model_dir
    # create a summary writer object for train_log
    train_writer = SummaryWriter(model_dir)

    # running_val_losses=[]
    estop_threshold = 20
    estop_counter = 0
    while da.current_epoch < n_epochs:
        # for epoch in range(n_epochs):
        # print(f'epoch: {epoch}')
        reward_model.train()
        total_loss = 0
        train_losses_total = 0
        train_pred = []
        val_pred = []
        test_pred = []
        trainlosses = []
        for train_batch in tqdm(train_loader):
            optimizer.zero_grad()
            # train_batch.cuda()
            # old_lr=optimizer.param_groups[0]["lr"]
            # blen_indiv=train_batch.lens_batch
            # n_comparison=np.sum([get_ncomb2(t) for t in blen_indiv])
            # optimizer.param_groups[0]["lr"]=old_lr*n_comparison/8
            loss, vals = da.batch_format_func(reward_model, train_batch, da)
            loss.backward()
            optimizer.step()
            train_losses_total += loss
            train_pred.append(vals)  # .detach().cpu())
            trainlosses.append(loss)
            # optimizer.param_groups[0]["lr"]=old_lr
        vp = [v["logits"] for v in train_pred]
        vsv = torch.cat(vp)
        train_correct = torch.sum(vsv[:, 0] > vsv[:, 1]) / vsv.shape[0]
        print(f"train correct: {train_correct.item():.4f}")

        reward_model.eval()
        vlosses = []
        with torch.no_grad():
            for batch in tqdm(val_loader):
                vl, vals = da.batch_format_func(reward_model, batch, da)
                vlosses.append(vl)
                val_pred.append(vals)  # .detach().cpu())
        vp = [v["logits"] for v in val_pred]
        vsv = torch.cat(vp)
        val_correct = torch.sum(vsv[:, 0] > vsv[:, 1]) / vsv.shape[0]
        print(f"validation correct: {val_correct.item():.4f}")

        tlosses = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                vl, vals = da.batch_format_func(reward_model, batch, da)
                tlosses.append(vl)
                test_pred.append(vals)  # .detach().cpu())
        vp = [v["logits"] for v in val_pred]
        vsv = torch.cat(vp)
        test_correct = torch.sum(vsv[:, 0] > vsv[:, 1]) / vsv.shape[0]
        print(f"test correct: {test_correct.item():.4f}")

        train_writer.add_scalar("valcorrect_epo", val_correct.item(), da.current_epoch)
        train_writer.add_scalar("testcorrect_epo", test_correct.item(), da.current_epoch)
        train_writer.add_scalar("traincorrect_epo", train_correct.item(), da.current_epoch)

        val_loss = torch.mean(torch.stack(vlosses))
        train_loss = torch.mean(torch.stack(trainlosses))

        print(f"epoch: {da.current_epoch}\tval_loss: {val_loss:.6f}")  # \t\tval_sym: {val_sym:.3f}\tval_mag: {val_mag:.3f}\tval_abs: {int(val_abs)}\tval_sixnorm: {val_sixnorm:.3f}')
        print(f"epoch: {da.current_epoch}\ttrain_loss: {train_loss:.6f}")  # \t\tval_sym: {val_sym:.3f}\tval_mag: {val_mag:.3f}\tval_abs: {int(val_abs)}\tval_sixnorm: {val_sixnorm:.3f}')

        train_writer.add_scalar("vloss_epo", val_loss.item(), da.current_epoch)
        train_writer.add_scalar("train_loss_total_epo", train_losses_total, da.current_epoch)

        da.running_val_losses.append(val_loss)
        da.running_train_losses.append(train_loss)
        da.running_test_correct.append(test_correct)
        da.running_train_correct.append(train_correct)
        da.running_val_correct.append(val_correct)

        # ['running_train_losses','running_val_losses','test_correct','val_correct','train_correct']

        save_rwd_model_pkl(reward_model, da, condition="latest")

        if val_loss == min(da.running_val_losses):
            save_rwd_model_pkl(reward_model, da, condition="lowest_val_loss")
            # torch.save(reward_model.state_dict(), os.path.join(model_dir,f'model_state_dict_best.pth'))                               #save model

        if val_correct == max(da.running_val_correct):
            save_rwd_model_pkl(reward_model, da, condition="highest_val_acc")
            # torch.save(reward_model.state_dict(), os.path.join(model_dir,f'model_state_dict_best.pth'))                               #save model

        if da.current_epoch % 50 == 0:
            save_rwd_model_pkl(reward_model, da, condition=f"epo_{da.current_epoch}")

            # torch.save(reward_model.state_dict(), os.path.join(model_dir,f'model_state_epo_{da.current_epoch}.pth'))

        if plot_dists_as_train:
            sel_keys = ["test", "val"]
            type_of_comparison = "perm"

            # if epoch%50==0:
            #    sel_keys.append('train')
            from collections import OrderedDict

            # for k in dset_dict.keys():
            for k in sel_keys:
                import pandas as pd

                all_indiv_e = dset_dict[k].dsets[0].return_all_indiv_examples()
                all_unique_seeds = dset_dict[k].dsets[0].get_sorted_unique_seeds()

                dict_of_seeds = OrderedDict()
                dict_nogood = OrderedDict()

                for ttt, e in zip(all_unique_seeds, all_indiv_e):
                    dict_of_seeds[ttt] = e

                    if ttt < 100000:
                        dict_nogood[ttt] = e

                # all_indiv_e=[a.permute(1,0).unsqueeze(0) for a in all_indiv_e]
                # dict_of_seeds=ordered{k:e for k,e in zip(all_indiv_e,all_unique_seeds)}

                # dict_nogood={k:dict_of_seeds[k] for k in dict_of_seeds.keys() if k<100000}

                for condition in ["all", "nogood"]:
                    if condition == "nogood":
                        for ttt in all_unique_seeds:
                            if ttt > 100000:
                                del dict_of_seeds[ttt]

                    all_indiv_e = torch.cat([d for d in dict_of_seeds.values()], 0).cuda()
                    all_unique_seeds = [d for d in dict_of_seeds.keys()]

                    tload = torch.utils.data.DataLoader(all_indiv_e, batch_size=8, drop_last=False, shuffle=False)

                    gfv = []
                    for tbatch in tqdm(tload):
                        with torch.no_grad():
                            global_feature_vectors = reward_model.forward(tbatch)
                            gfv.append(global_feature_vectors)
                    global_feature_vectors = torch.cat(gfv, 0)
                    tep = global_feature_vectors  # torch.cat([torch.cat(tp['global_feature_preds'],0) for tp in test_pred])

                    idx = torch.arange(tep.shape[0])

                    wins = []
                    pwins = []

                    with torch.no_grad():
                        for i in idx:
                            idxe = [k for k in idx if k != i]
                            lll = reward_model.forward_from_cat_global_vectors(tep[i][None, ...].repeat(len(idxe), 1), tep[idxe])
                            lll_r = reward_model.forward_from_cat_global_vectors(tep[idxe], tep[i][None, ...].repeat(len(idxe), 1))
                            win = torch.logical_and(lll[:, 0] > 0.5, lll_r[:, 1] > 0.5)
                            pwin = lll[:, 0] * lll_r[:, 1]

                            wins.append(win)
                            pwins.append(pwin)

                    n_wins = [torch.sum(w) for w in wins]
                    p_wins = [torch.mean(p) for p in pwins]

                    n_wins = torch.hstack(n_wins).cpu().numpy()
                    p_wins = torch.hstack(p_wins).cpu().numpy()

                    pdr = pd.DataFrame([all_unique_seeds, n_wins, p_wins]).transpose()

                    pdr.columns = ["seed", "total_g_05", "mean"]
                    # mean_df.columns=['mean']
                    # tot_df.columns=['total_g_05']
                    # seed_df.columns=['seed']

                    # joined_all=mean_df.join((tot_df,seed_df))
                    norms = torch.norm(global_feature_vectors, dim=1).detach()
                    # seeds=dset_dict[k].dsets[0].get_sorted_unique_seeds()
                    dd = {s: n.item() for s, n in zip(all_unique_seeds, norms)}
                    pds = pd.DataFrame.from_dict(dd, orient="index").reset_index()
                    pds.columns = ["seed", "l2norm_global_embedding"]
                    joined_all = pdr.merge(pds, on="seed")

                    import os

                    import imageio.v3 as imageio  # # install imageio-ffmpeg

                    joined_all.to_csv(
                        os.path.join(
                            da.model_dir,
                            f"{k}_df_results_epoch_{da.current_epoch}_{condition}.csv",
                        )
                    )

                    # Create an empty list to hold the images

                    seeds_ascending = joined_all.sort_values(by="mean", ascending=True).seed.astype(np.int32).values
                    ordered_meshes_fn = [os.path.join(ddir_func(s), f"mesh_cat_s_{s}.jpg") for s in seeds_ascending]

                    vid_ims = []
                    # Load the images and add them to the list
                    for filename in ordered_meshes_fn:
                        vid_ims.append(imageio.imread(filename))

                    # Create the output video file
                    imageio.imwrite(
                        os.path.join(da.model_dir, f"{k}_meshvid_{condition}.mp4"),
                        vid_ims,
                        fps=2,
                        quality=5,
                    )

                    N_MESHES = 10
                    # read the joined_all dataframe from a csv file
                    joined_all = pd.read_csv(
                        os.path.join(
                            da.model_dir,
                            f"{k}_df_results_epoch_{da.current_epoch}_{condition}.csv",
                        )
                    ).astype({"seed": "int32"})

                    # sort the dataframe by the mean column in descending order
                    sorted_joined_all = joined_all.sort_values(by="mean", ascending=False)

                    # extract the top 5 seeds and loss values
                    top_5_seeds = sorted_joined_all["seed"].head(N_MESHES).tolist()
                    top_5_losses = sorted_joined_all["mean"].head(N_MESHES).tolist()

                    seedmeshes = [os.path.join(ddir_func(s), f"mesh_cat_s_{s}.jpg") for s in top_5_seeds]
                    pics = [os.path.join(ddir_func(s), f"triple_rgb_s_{s}_1.jpg") for s in top_5_seeds]
                    overall_images = plot_overall_images(seedmeshes, pics)
                    grid_img = torchvision.utils.make_grid(overall_images, nrow=N_MESHES)
                    out_fn_t = os.path.join(
                        da.model_dir,
                        f"{k}_type_{type_of_comparison}_ top_5_{da.current_epoch}_{condition}.jpg",
                    )
                    PIL.Image.fromarray(grid_img.numpy().transpose(1, 2, 0)).save(out_fn_t)

                    top_5_seeds = sorted_joined_all["seed"].tail(N_MESHES).tolist()
                    top_5_losses = sorted_joined_all["mean"].tail(N_MESHES).tolist()

                    # print("Top 5 seeds:", top_5_seeds)
                    # print("Top 5 losses:", top_5_losses)
                    seedmeshes = [os.path.join(ddir_func(s), f"mesh_cat_s_{s}.jpg") for s in top_5_seeds]
                    pics = [os.path.join(ddir_func(s), f"triple_rgb_s_{s}_1.jpg") for s in top_5_seeds]
                    overall_images = plot_overall_images(seedmeshes, pics)
                    grid_img = torchvision.utils.make_grid(overall_images, nrow=N_MESHES)
                    out_fn_b = os.path.join(
                        da.model_dir,
                        f"{k}_type_{type_of_comparison}_ bottom_5_{da.current_epoch}_{condition}.jpg",
                    )
                    PIL.Image.fromarray(grid_img.numpy().transpose(1, 2, 0)).save(out_fn_b)

                    # get top five which are not from the good mesh category...
                    # joined_all = pd.read_csv(os.path.join(da.model_dir,f'{dt}_type_df_results_epoch_{epoch}.csv'))

                    # sort the dataframe by the mean column in descending order
                    # sorted_joined_all = joined_all.sort_values(by='rwd', ascending=False)
                    sorted_joined_all = sorted_joined_all[sorted_joined_all.seed < 100000].sort_values(by="mean", ascending=False)  # 100000 is starting seed for the good meshes
                    # extract the top 5 seeds and loss values
                    top_5_seeds = sorted_joined_all["seed"].head(N_MESHES).tolist()
                    top_5_losses = sorted_joined_all["mean"].head(N_MESHES).tolist()

                    # print("Top 5 seeds:", top_5_seeds)
                    # print("Top 5 losses:", top_5_losses)
                    seedmeshes = [os.path.join(ddir_func(s), f"mesh_cat_s_{s}.jpg") for s in top_5_seeds]
                    pics = [os.path.join(ddir_func(s), f"triple_rgb_s_{s}_1.jpg") for s in top_5_seeds]
                    overall_images = plot_overall_images(seedmeshes, pics)
                    grid_img = torchvision.utils.make_grid(overall_images, nrow=N_MESHES)
                    out_fn_ng = os.path.join(
                        da.model_dir,
                        f"{k}_type_{type_of_comparison}_nogood_bottom_5_{da.current_epoch}_{condition}.jpg",
                    )
                    PIL.Image.fromarray(grid_img.numpy().transpose(1, 2, 0)).save(out_fn_ng)

                    tope = PIL.Image.open(out_fn_t)
                    bote = PIL.Image.open(out_fn_ng)

                    w1, h1 = tope.size
                    w2, h2 = bote.size

                    new_img = Image.new("RGB", (max(w1, w2), h1 + h2), (255, 255, 255))
                    new_img.paste(tope, (0, 0))
                    new_img.paste(bote, (0, h1))

                    # out_fn=os.path.join(da.model_dir,f'{dt}_top_bottom_5_{epoch}.jpg')

                    tope = new_img  # .save(out_fn)

                    bote = PIL.Image.open(out_fn_b)

                    w1, h1 = tope.size
                    w2, h2 = bote.size

                    new_img = Image.new("RGB", (max(w1, w2), h1 + h2), (255, 255, 255))
                    new_img.paste(tope, (0, 0))
                    new_img.paste(bote, (0, h1))

                    out_fn = os.path.join(
                        da.model_dir,
                        f"{k}_type_{type_of_comparison}_top_bottom_5_{da.current_epoch}_{condition}.jpg",
                    )

                    img = new_img
                    img.resize((int(img.size[0] * 0.4), int(img.size[1] * 0.4))).save(out_fn)

                    os.remove(out_fn_b)
                    os.remove(out_fn_t)
                    os.remove(out_fn_ng)

                    mean_score_vals = pdr["mean"].values
                    sum_g_50_pc_vals = pdr.total_g_05.values
                    plt.clf()
                    plt.hist(mean_score_vals, bins=100)
                    plt.title("hist: mean_p_win_over_test_set_all_comparisons")
                    plt.xlabel("mean probabiilltiy success")
                    plt.ylabel("n")
                    plt.savefig(
                        os.path.join(
                            da.model_dir,
                            f"{k}_type_{type_of_comparison}_mean_p_epoch{da.current_epoch}_{condition}.png",
                        )
                    )

                    plt.clf()
                    plt.hist(sum_g_50_pc_vals, bins=100)
                    plt.title("hist: sum_cases_p_greater_05_for_each_example")
                    plt.xlabel("sum of successful comparisons")
                    plt.ylabel("n")
                    plt.savefig(
                        os.path.join(
                            da.model_dir,
                            f"{k}_type_{type_of_comparison}_sum_wins_epoch{da.current_epoch}_{condition}.png",
                        )
                    )

        da.current_epoch += 1

    train_writer.close()


def load_rwd_mdl_from_pkl(pkl_dir, state_dict=True):
    with open(pkl_dir, "rb") as f:
        rwd_mdl = pickle.load(f)

    m_init_params = rwd_mdl["m_init_params"]
    classifier = eval(m_init_params["MODEL_CLASS"])(**m_init_params)  # this is reward model....

    if state_dict:
        state_dict = rwd_mdl["model_state_dict"]
        classifier.load_state_dict(state_dict)

    classifier.cuda()
    classifier.eval()
    print(f"Loaded best model from {pkl_dir}")

    da = dargs()

    if "da" in rwd_mdl.keys():
        da = rwd_mdl["da"]

    if classifier.reward_model_type == "rwd_model_3dmap_vgg_minimal":
        da.vgg19_to_4096_model = vgg19_to_4096()
        da.vgg19_to_4096_model = da.vgg19_to_4096_model.cuda()

    if classifier.reward_model_type == "rwd_model_3dmap_vggface2_minimal":
        da.vggface2_to_512_model = vggface2_to_512()
        da.vggface2_to_512_model = da.vggface2_to_512_model.cuda()

    if "da_interpolation_mode" not in m_init_params.keys():
        da.interpolation_mode = "bilinear"
    else:
        da.interpolation_mode = m_init_params["da_interpolation_mode"]

    return classifier, da


def save_rwd_model_pkl(optimal_rwd_model, da, condition=""):
    # optimal_rwd_models_dir=os.path.join(da.RLHF_DIR,'optimal_reward_models')
    savedir = da.model_dir
    os.makedirs(savedir, exist_ok=True)
    for k in ["vgg19_to_4096_model", "vggface2_to_512_model"]:
        if hasattr(da, k):
            delattr(da, k)
    bundle_pkl = {}
    bundle_pkl["m_init_params"] = da.m_init_params
    bundle_pkl["model_state_dict"] = optimal_rwd_model.state_dict()
    bundle_pkl["reward_model_type"] = optimal_rwd_model.reward_model_type
    bundle_pkl["da"] = da

    pkl_save_name = os.path.join(savedir, f"chkpt_{condition}.pkl")

    with open(pkl_save_name, "wb") as f:
        pickle.dump(bundle_pkl, f)

    print("saved optimal rwd_model")


def plot_overall_images(seedmeshes, pics):
    overall_images = []

    for s, p in zip(seedmeshes, pics):
        # open and resize the images
        img1 = Image.open(s)  # .resize((200, 200))
        img2 = Image.open(p)  # .resize((200, 300))

        # get the dimensions of the images
        w1, h1 = img1.size
        w2, h2 = img2.size

        # compute the difference in height
        dh = abs(h1 - h2)

        # create a new image with the maximum width and the sum of the heights
        new_img = Image.new("RGB", (max(w1, w2), h1 + h2), (255, 255, 255))

        # paste the first image at the top
        new_img.paste(img1, (0, 0))

        # paste the second image at the bottom, with padding if necessary
        if h1 > h2:
            new_img.paste(img2, (0, h1))
        else:
            new_img.paste(img2, (0, h1 - dh))

        overall_images.append(torch.from_numpy(np.array(new_img)).unsqueeze(0))

    overall_images = torch.cat(overall_images, 0).permute(0, 3, 1, 2)

    return overall_images


def train_rwd_model_from_scratch(
    da,
    optimizer,
    reward_model,
    train_loader,
    val_loader,
    test_loader,
    plot_dists_as_train=True,
    n_epochs=20,
):
    model_dir = da.model_dir
    # create a summary writer object for train_log
    train_writer = SummaryWriter(model_dir)

    running_val_losses = []
    estop_threshold = 20
    estop_counter = 0
    for epoch in range(n_epochs):
        # print(f'epoch: {epoch}')
        reward_model.train()
        total_loss = 0
        train_losses_total = 0
        for k, batch in enumerate(iter(train_loader)):
            optimizer.zero_grad()
            loss, vals = da.batch_format_func(reward_model, batch, da)
            loss.backward()
            optimizer.step()
            train_losses_total += loss

            if k % 10 == 0:
                print(k)

        reward_model.eval()
        vlosses = []
        with torch.no_grad():
            for k, batch in enumerate(iter(val_loader)):
                vl, vals = da.batch_format_func(reward_model, batch, da)
                vlosses.append(vl)
                if k % 10 == 0:
                    print(k)

        val_loss = torch.mean(torch.stack(vlosses))

        print(f"epoch: {epoch}\tval_loss: {val_loss:.6f}")  # \t\tval_sym: {val_sym:.3f}\tval_mag: {val_mag:.3f}\tval_abs: {int(val_abs)}\tval_sixnorm: {val_sixnorm:.3f}')

        train_writer.add_scalar("vloss_epo", val_loss.item(), epoch)
        train_writer.add_scalar("train_loss_total_epo", train_losses_total, epoch)

        running_val_losses.append(val_loss)

        if val_loss == min(running_val_losses):
            torch.save(reward_model.state_dict(), os.path.join(model_dir, "model_state_dict_best.pth"))  # save model

        if epoch > 10:
            if running_val_losses[-1] > running_val_losses[-2]:
                estop_counter += 1
            else:
                estop_counter = 0
            if estop_counter > estop_threshold:
                print("early stopping")
                break

        # fig,normalisation_terms,optimal_rwd_model=model_evaluate_with_plot(da.model_dir,reward_model,test_loader,train_loader,val_loader,da)

        if plot_dists_as_train:
            fig, _ = evaluate_model(reward_model, test_loader, train_loader, val_loader, da)
            fig.savefig(
                os.path.join(da.model_dir, f"reward_model_epoch_{epoch}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            fig.clf()
            fig.clear()
            plt.close(fig)
            plt.close("all")

            tmp_eval_file = os.path.join(da.model_dir, "tmp_eval_file.txt")

            with open(tmp_eval_file, "a") as f:
                print(f"epoch: {epoch}\tval_loss: {val_loss:.6f}", file=f)  # \t\tval_sym: {val_sym:.3f}\tval_mag: {val_mag:.3f}\tval_abs: {int(val_abs)}\tval_sixnorm: {val_sixnorm:.3f}')

            os.rename(
                dst=os.path.join(da.model_dir, f"reward_model_epoch_{epoch}.txt"),
                src=tmp_eval_file,
            )

        # current_rwd_ims=glob.glob(os.path.join(da.model_dir,'reward_model_evaluation*.png'))
        # next_idx=len(current_rwd_ims)
        # save ut

    train_writer.close()


def get_optimiser(reward_model, LR=1e-4, WEIGHT_DECAY=1e-5):
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # , weight_decay=0.0001)
    return optimizer


def form_pcd_path(seed, condition):
    sd = f"seed{seed:04d}"
    return f"/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/{condition}/{sd}.pcd"


def form_json_path_three_dmaps(seed, condition):
    sd = f"seed{seed:04d}"
    return os.path.join(RLHF_DIR, f"rlhf_meshes/{condition}/{sd}_three_dmaps.json")


def form_json_path(seed, condition):
    sd = f"seed{seed:04d}"
    return os.path.join(RLHF_DIR, f"rlhf_meshes/{condition}/{sd}.json")


def return_z_from_json(json_fn):
    with open(json_fn) as f:
        z = json.load(f)["z"]
    return z


def return_dmap_from_json(json_fn):
    with open(json_fn) as f:
        z = json.load(f)["image_depth"]
    return z


def pull_out_depth_pair_from_rankings_df(rankings_df, input_pair):
    idx1 = input_pair[0]
    idx2 = input_pair[1]
    json_path1 = rankings_df.json_path[idx1]
    json_path2 = rankings_df.json_path[idx2]
    z1 = rankings_df.z[idx1]
    z2 = rankings_df.z[idx2]
    dmap1 = rankings_df.dmap[idx1]
    dmap2 = rankings_df.dmap[idx2]
    total1 = rankings_df.total[idx1]
    total2 = rankings_df.total[idx2]
    unique_id_1 = rankings_df.unique_id[idx1]
    unique_id_2 = rankings_df.unique_id[idx2]
    first_combination = (json_path1, z1, dmap1, total1, unique_id_1)
    second_combination = (json_path2, z2, dmap2, total2, unique_id_2)
    combo_list = [first_combination, second_combination]
    assert total1 != total2
    high_score_idx = np.argmax([total1, total2])
    low_score_idx = np.argmin([total1, total2])
    high_comb = combo_list[high_score_idx]
    low_comb = combo_list[low_score_idx]
    return (high_comb, low_comb)


def pull_out_pcd_pair_from_rankings_df(rankings_df, input_pair):
    idx1 = input_pair[0]
    idx2 = input_pair[1]
    json_path1 = rankings_df.json_path[idx1]
    json_path2 = rankings_df.json_path[idx2]
    z1 = rankings_df.z[idx1]
    z2 = rankings_df.z[idx2]
    pcd_path1 = rankings_df.pcd_path[idx1]
    pcd_path2 = rankings_df.pcd_path[idx2]
    total1 = rankings_df.total[idx1]
    total2 = rankings_df.total[idx2]
    first_combination = (json_path1, z1, pcd_path1, total1)
    second_combination = (json_path2, z2, pcd_path2, total2)
    combo_list = [first_combination, second_combination]
    assert total1 != total2
    high_score_idx = np.argmax([total1, total2])
    low_score_idx = np.argmin([total1, total2])
    high_comb = combo_list[high_score_idx]
    low_comb = combo_list[low_score_idx]
    return (high_comb, low_comb)


def pull_depth_maps_hilo_three(current_pair, nrs=128):
    high_dmap = current_pair[0][2]
    low_dmap = current_pair[1][2]
    hdmp = high_dmap.replace("[", "").replace("]", "").split(",")
    hdmp = np.array([float(h) for h in hdmp])
    hdmp_split = np.array_split(hdmp, 3)
    hdmp_list = [h.reshape(nrs, nrs) for h in hdmp_split]
    hdmp = np.array(hdmp_list)
    ldmp = low_dmap.replace("[", "").replace("]", "").split(",")
    ldmp = np.array([float(h) for h in ldmp])
    ldmp_split = np.array_split(ldmp, 3)
    ldmp_list = [h.reshape(nrs, nrs) for h in ldmp_split]
    ldmp = np.array(ldmp_list)

    return dict(high=hdmp, low=ldmp)


def pull_z_hilo_three(current_pair):
    high_Z = current_pair[0][1]
    low_Z = current_pair[1][1]
    z_dim = 512
    hdmp = high_Z.replace("[", "").replace("]", "").split(",")
    hdmp = np.array([float(h) for h in hdmp])
    hdmp = hdmp.reshape(1, 3, z_dim)
    ldmp = low_Z.replace("[", "").replace("]", "").split(",")
    ldmp = np.array([float(h) for h in ldmp])
    ldmp = ldmp.reshape(1, 3, z_dim)
    return (hdmp, ldmp)


def pull_depth_maps_hilo(current_pair, nrs=128):
    high_dmap = current_pair[0][2]
    low_dmap = current_pair[1][2]
    hdmp = high_dmap.replace("[", "").replace("]", "").split(",")
    hdmp = np.array([float(h) for h in hdmp])
    hdmp = hdmp.reshape(nrs, nrs)
    ldmp = low_dmap.replace("[", "").replace("]", "").split(",")
    ldmp = np.array([float(h) for h in ldmp])
    ldmp = ldmp.reshape(nrs, nrs)
    return dict(high=hdmp, low=ldmp)


def pull_z_hilo(current_pair, z_dim=512):
    high_Z = current_pair[0][1]
    low_Z = current_pair[1][1]
    hdmp = high_Z.replace("[", "").replace("]", "").split(",")
    hdmp = np.array([float(h) for h in hdmp])
    hdmp = hdmp.reshape(1, z_dim)
    ldmp = low_Z.replace("[", "").replace("]", "").split(",")
    ldmp = np.array([float(h) for h in ldmp])
    ldmp = ldmp.reshape(1, z_dim)
    return (hdmp, ldmp)


def pull_score_hilo(current_pair, z_dim=512):
    high_Z = current_pair[0][3]
    low_Z = current_pair[1][3]
    hdmp = high_Z.replace("[", "").replace("]", "").split(",")
    hdmp = np.array([float(h) for h in hdmp])
    hdmp = hdmp.reshape(1, z_dim)
    ldmp = low_Z.replace("[", "").replace("]", "").split(",")
    ldmp = np.array([float(h) for h in ldmp])
    ldmp = ldmp.reshape(1, z_dim)
    return (hdmp, ldmp)


def reshape_hilo_maps(pulled_depths, dmap_res=128):
    hi = pulled_depths["high"].reshape(-1, dmap_res, dmap_res)
    lo = pulled_depths["low"].reshape(-1, dmap_res, dmap_res)
    return (hi, lo)


def pull_scores(input_pair):
    hiscore = input_pair[0][3]
    loscore = input_pair[1][3]
    return (hiscore, loscore)


def compare_scores(rankings_df, pair_tuple):
    first_score = rankings_df.total[pair_tuple[0]]
    second_score = rankings_df.total[pair_tuple[1]]
    if first_score == second_score:
        return "tie"
    else:
        return "not tie"


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def get_n_params_w_grad(model):
    ll = list(model.parameters())
    tot = 0
    for l in ll:
        if l.requires_grad:
            n = l.numel()
            tot += n
    return tot


def collate_predict_single_pnet(classifier, dl):
    single_preds = [predict_single_pnet(classifier, dl, idx=i) for i in range(dl[0].shape[0])]
    wins = [x[0].flatten().detach().cpu()[0].item() for x in single_preds]
    losses = [x[1].flatten().detach().cpu()[0].item() for x in single_preds]
    scores = [x[2].flatten().detach().cpu() for x in single_preds]
    score_win = [x[0].item() for x in scores]
    score_lose = [x[1].item() for x in scores]
    df = pd.DataFrame(dict(win_pred=wins, lose_pred=losses, win_score=score_win, lose_score=score_lose))
    return df


def pull_ids(input_pair):
    hiscore = input_pair[0][4]
    loscore = input_pair[1][4]
    return (hiscore, loscore)


def convert_df_to_dataset_dmap_3(rankings_df):
    # print([c for c in rankings_df.columns])
    test_list = [i for i in rankings_df.index]
    res = []
    n = len(test_list)
    for i in range(n):
        for j in range(i + 1, n):
            res.append((test_list[i], test_list[j]))
    print("n pairs")
    print(len(res))
    print("\n\n")

    compared_scores = [compare_scores(rankings_df, pair_tuple) for pair_tuple in res]
    compared_scores_counter = Counter(compared_scores)
    print("here are the counts of tie and not tie")
    print(compared_scores_counter)
    not_tied_pairs = np.array(res)[[c != "tie" for c in compared_scores]]
    tied_pairs = np.array(res)[[c == "tie" for c in compared_scores]]

    print("not tied pairs shape")
    print(not_tied_pairs.shape)
    print("tied pairs shape")
    print(tied_pairs.shape)
    pairs_of_combos_depth = [pull_out_depth_pair_from_rankings_df(rankings_df, input_pair) for input_pair in not_tied_pairs]
    pairs_of_combos_depth[0][0]
    pairs_of_combos_depth = [pull_out_depth_pair_from_rankings_df(rankings_df, input_pair) for input_pair in not_tied_pairs]
    high_low_zs = [(high_comb[1], low_comb[1]) for high_comb, low_comb in pairs_of_combos_depth]

    # Z SEED (REMOVE)

    print("pairs of combos depth")
    print(pairs_of_combos_depth[0][1][-1])
    high_low_zs = [(high_comb[1], low_comb[1]) for high_comb, low_comb in pairs_of_combos_depth]
    print("zs")
    pairs_of_combos_depth = [pull_out_depth_pair_from_rankings_df(rankings_df, input_pair) for input_pair in not_tied_pairs]
    z_pairs = [(high_comb[1], low_comb[1]) for high_comb, low_comb in pairs_of_combos_depth]

    # --------------------------------------------------------------------------------------------
    # Z SEED (REMOVE)
    print("pairs of combos depth")
    print("zseeds")
    high_low_zs = [pull_z_hilo_three(input_pair) for input_pair in pairs_of_combos_depth]

    print("hi lo z shape")
    print(torch.tensor(high_low_zs).shape)

    # --------------------------------------------------------------------------------------------

    # DEPTH MAP
    print("dmaps")
    pairs_of_depth_pulled = [pull_depth_maps_hilo_three(pair) for pair in pairs_of_combos_depth]
    high_low_dmaps = [reshape_hilo_maps(pulled_depths) for pulled_depths in pairs_of_depth_pulled]
    print("hi lo dmap shape")
    print(torch.tensor(high_low_dmaps).shape)

    # --------------------------------------------------------------------------------------------

    # HIGH LOW SCORE

    high_low_totals = [pull_scores(pair) for pair in pairs_of_combos_depth]
    print("scores")
    print(torch.tensor(high_low_totals).shape)

    X_dmaps = torch.tensor(high_low_dmaps)
    X_zall = torch.tensor(high_low_zs)
    yall = torch.tensor(high_low_totals)

    return dict(X_dmaps=X_dmaps, X_zall=X_zall, yall=yall)


def convert_df_to_dataset_dmap_single(rankings_df):
    print([c for c in rankings_df.columns])

    # initializing list
    test_list = [i for i in rankings_df.index]

    res = []
    n = len(test_list)
    for i in range(n):
        for j in range(i + 1, n):
            res.append((test_list[i], test_list[j]))
    print("n pairs")
    print(len(res))
    print("\n\n")
    compared_scores = [compare_scores(rankings_df, pair_tuple) for pair_tuple in res]
    compared_scores_counter = Counter(compared_scores)
    print("here are the counts of tie and not tie")
    print(compared_scores_counter)
    not_tied_pairs = np.array(res)[[c != "tie" for c in compared_scores]]
    tied_pairs = np.array(res)[[c == "tie" for c in compared_scores]]
    print("not tied pairs shape")
    print(not_tied_pairs.shape)
    print("tied pairs shape")
    print(tied_pairs.shape)
    pairs_of_combos_depth = [pull_out_depth_pair_from_rankings_df(rankings_df, input_pair) for input_pair in not_tied_pairs]
    # Z SEED (REMOVE)
    print("pairs of combos depth")
    print(pairs_of_combos_depth[0][1][-1])
    high_low_zs = [(high_comb[1], low_comb[1]) for high_comb, low_comb in pairs_of_combos_depth]
    print("zs")
    # --------------------------------------------------------------------------------------------
    # Z SEED (REMOVE)
    print("pairs of combos depth")
    print("zseeds")
    high_low_zs = [pull_z_hilo(input_pair) for input_pair in pairs_of_combos_depth]
    print("hi lo z shape")
    print(torch.tensor(high_low_zs).shape)
    # --------------------------------------------------------------------------------------------
    # DEPTH MAP
    print("dmaps")
    pairs_of_depth_pulled = [pull_depth_maps_hilo(pair) for pair in pairs_of_combos_depth]
    high_low_dmaps = [reshape_hilo_maps(pulled_depths) for pulled_depths in pairs_of_depth_pulled]
    print("hi lo dmap shape")
    print(torch.tensor(high_low_dmaps).shape)
    # --------------------------------------------------------------------------------------------
    # HIGH LOW SCORE
    high_low_totals = [pull_scores(pair) for pair in pairs_of_combos_depth]
    print("scores")
    print(torch.tensor(high_low_totals).shape)
    high_low_ids = [pull_ids(pair) for pair in pairs_of_combos_depth]
    print("hi lo ids")
    print(torch.tensor(high_low_ids).shape)
    X_dmaps = torch.tensor(high_low_dmaps)
    X_zall = torch.tensor(high_low_zs)
    yall = torch.tensor(high_low_totals)

    return dict(X_dmaps=X_dmaps, X_zall=X_zall, yall=yall, ids=high_low_ids)


def make_dataset_train(X_dmaps, X_zall, yall, ids, nrs, batch_size=32):
    # Create dataset from several tensors with matching first dimension
    # Samples will be drawn from the first dimension (rows)

    if nrs == 64:
        X_dmaps = torch.nn.functional.interpolate(X_dmaps, size=[3, 64, 64], mode="nearest")

    dataset = TensorDataset(
        Tensor(X_dmaps).to(device).float(),
        Tensor(X_zall).to(device).float(),
        Tensor(yall.float()).to(device).float(),
        Tensor(ids).to(device).int(),
    )

    print("len dataset")
    print(len(dataset))
    print("----------")
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def make_dataset_vt(X_dmaps, X_zall, yall, ids, nrs, batch_size=None):
    if nrs == 64:
        X_dmaps = torch.nn.functional.interpolate(X_dmaps, size=[3, 64, 64], mode="nearest")

    dataset = TensorDataset(
        Tensor(X_dmaps).to(device).float(),
        Tensor(X_zall).to(device).float(),
        Tensor(yall.float()).to(device).float(),
        Tensor(ids).to(device).int(),
    )

    if batch_size is None:
        batch_size = len(dataset)
    print("len dataset")
    print(len(dataset))
    print("----------")
    dloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dloader


def make_dataset(X_dmaps, X_zall, yall, nrs):
    # Create dataset from several tensors with matching first dimension
    # Samples will be drawn from the first dimension (rows)

    if nrs == 64:
        X_dmaps = torch.nn.functional.interpolate(X_dmaps, size=[3, 64, 64], mode="nearest")

    dataset = TensorDataset(
        Tensor(X_dmaps).to(device).float(),
        Tensor(X_zall).to(device).float(),
        Tensor(yall.float()).to(device).float(),
        Tensor(ids).to(device).int(),
    )

    print("len dataset")
    print(len(dataset))
    print("----------")
    splits = [0.8, 0.1, 0.1]
    sint = [
        int(len(dataset) * splits[0]),
        int(len(dataset) * splits[1]),
        int(len(dataset) * splits[2]),
    ]
    lengths = [sint[0], sint[1]]

    lengths.append(len(dataset) - sum(lengths))

    generator1 = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, lengths, generator=generator1)

    print("\nnum cases in each partition\ntrain, val, test")
    print(len(train_set), len(val_set), len(test_set))
    print("--------")
    print("")

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set))
    test_loader = DataLoader(test_set, batch_size=len(test_set))

    tt = next(iter(train_loader))

    test_example = next(iter(test_loader))
    train_example = next(iter(train_loader))

    print("train example shape")
    print(train_example[0].shape)
    print("test example shape")
    print(test_example[0].shape)
    # val_all=next(iter(val_loader))

    return dict(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_example=train_example,
        test_example=test_example,
    )


def compare_partitions_for_identical(dset_dict):
    # now do same for val set

    test_loader = dset_dict["test_loader"]

    test_example = next(iter(test_loader))

    flattened_maps_all_test = [t[:].flatten().detach().cpu().numpy() for t in test_example[0]]

    print("number of pairs in test set:")
    print(len(flattened_maps_all_test))

    print("shape of flattened maps:")
    print(flattened_maps_all_test[0].shape)

    # check all are unique in flattened_maps_all

    unique_combos = np.unique(np.array(flattened_maps_all_test), axis=0)

    print("shape of unique combos, ie should match above or be less than, if greater than something is wrong")
    print(unique_combos.shape)

    n_unique_combos = unique_combos.shape[0]
    print("-----------")
    print("does n_unique_combos match number of pairs in test set?")

    print(n_unique_combos == len(flattened_maps_all_test))

    val_loader = dset_dict["val_loader"]
    val_example = next(iter(val_loader))

    flattened_maps_all_val = [t[:].flatten().detach().cpu().numpy() for t in val_example[0]]

    print("number of pairs in val set:")
    print(len(flattened_maps_all_val))

    print("shape of flattened maps:")
    print(flattened_maps_all_val[0].shape)

    # check all are unique in flattened_maps_all

    unique_combos = np.unique(np.array(flattened_maps_all_val), axis=0)

    print("shape of unique combos, ie should match above or be less than, if greater than something is wrong")
    print(unique_combos.shape)

    n_unique_combos = unique_combos.shape[0]
    print("-----------")
    print("does n_unique_combos match number of pairs in val set?")

    print(n_unique_combos == len(flattened_maps_all_val))

    flattened_maps_all_train = []

    for k, batch in enumerate(iter(train_loader)):
        flattened_train_maps = [t[:].flatten().detach().cpu().numpy() for t in batch[0]]

        flattened_maps_all_train.extend(flattened_train_maps)

    print("number of pairs in train set:")
    print(len(flattened_maps_all_train))

    print("shape of flattened maps:")
    print(flattened_maps_all_train[0].shape)

    # check all are unique in flattened_maps_all

    unique_combos = np.unique(np.array(flattened_maps_all_train), axis=0)

    print("shape of unique combos, ie should match above or be less than, if greater than something is wrong")
    print(unique_combos.shape)

    n_unique_combos = unique_combos.shape[0]
    print("-----------")
    print("does n_unique_combos match number of pairs in train set?")

    print(n_unique_combos == len(flattened_maps_all_train))

    combined_flattened_maps = flattened_maps_all_test + flattened_maps_all_val + flattened_maps_all_train

    unique_combos = np.unique(np.array(combined_flattened_maps), axis=0)

    print("number of pairs in val+test+train set:")

    print(len(combined_flattened_maps))

    print("shape of unique combos, ie should match above or be less than, if greater than something is wrong")
    print(unique_combos.shape)

    n_unique_combos = unique_combos.shape[0]
    print("-----------")
    print("does n_unique_combos match number of pairs in val+test+train set?")

    print(n_unique_combos == len(combined_flattened_maps))


def get_resid_vec(vvw):
    vvw[torch.logical_and(0 <= vvw, vvw < 6)] = vvw[torch.logical_and(0 <= vvw, vvw < 6)] - vvw[torch.logical_and(0 <= vvw, vvw < 6)]
    vvw[vvw > 6] -= 6

    vvw[vvw < 0] = torch.abs(vvw[vvw < 0])

    vvw += 1e-6

    return vvw


def read_rankings(ratings_fn):
    ods_pth = os.path.join(RLHF_DIR, f"{ratings_fn}")
    sheet = get_book(file_name=ods_pth)
    sheet_names = sheet.sheet_names()
    # kept_names=[s for s in sheet_names if 'rebal' not in s]
    # sheet_names=kept_names
    df_list = []

    for s in sheet_names:
        df = read_ods(ods_pth, s, headers=True)
        df = df.astype("int32")
        df = df[["seed", "total"]]
        df["condition"] = s
        df["pcd_path"] = [form_pcd_path(df_seed, s) for df_seed in df.seed]
        df["json_path"] = [form_json_path(df_seed, s) for df_seed in df.seed]
        df["json_path_dmaps_three"] = [form_json_path_three_dmaps(df_seed, s) for df_seed in df.seed]
        df_list.append(df)

    df_list[0].head()
    df_all = pd.concat(df_list, axis=0, ignore_index=True)

    return df_all


def read_rankings_norebal(ratings_fn):
    ods_pth = os.path.join(RLHF_DIR, f"{ratings_fn}")
    sheet = get_book(file_name=ods_pth)
    sheet_names = sheet.sheet_names()
    kept_names = [s for s in sheet_names if "rebal" not in s]
    sheet_names = kept_names
    df_list = []

    for s in sheet_names:
        df = read_ods(ods_pth, s, headers=True)
        df = df.astype("int32")
        df = df[["seed", "total"]]
        df["condition"] = s
        df["pcd_path"] = [form_pcd_path(df_seed, s) for df_seed in df.seed]
        df["json_path"] = [form_json_path(df_seed, s) for df_seed in df.seed]
        df["json_path_dmaps_three"] = [form_json_path_three_dmaps(df_seed, s) for df_seed in df.seed]
        df_list.append(df)

    df_list[0].head()
    df_all = pd.concat(df_list, axis=0, ignore_index=True)

    return df_all


def convert_df_to_data_dmap_single(df_all, rwd_data_formatted_fn):
    df_all["z"] = [return_z_from_json(json_fn) for json_fn in df_all.json_path]
    df_all["dmap"] = [return_dmap_from_json(json_fn) for json_fn in df_all.json_path]

    df_all["unique_id"] = df_all.index

    df_all.to_csv(os.path.join(RLHF_DIR, rwd_data_formatted_fn))


def create_mlp(mlp_layers):
    layers_list = []

    for z, (l0, l1) in enumerate(zip(mlp_layers[:-1], mlp_layers[1:])):
        layer = nn.Linear(l0, l1)
        layers_list.append(layer)
        if z < len(mlp_layers) - 2:
            layers_list.append(nn.ReLU())

    mlp = nn.Sequential(*layers_list)
    return mlp


# ----------
# CONTRASTIVE LOSS
# taken from https://theaisummer.com/simclr/


def device_as(t1, t2):
    """Moves t1 to the device of t2."""
    return t1.to(t2.device)


class ContrastiveLoss(nn.Module):
    """Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper."""

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        """proj_1 and proj_2 are batched embeddings [batch, embedding_dim] where corresponding indices are pairs z_i, z_j in the SimCLR paper."""
        batch_size = proj_1.shape[0]

        self.batch_size = batch_size

        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)
        return loss


def get_ranking_folder(r):
    return f"/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/000_RLHF_AM/composed_for_binary_ranking/rlhf_meshes_ffhq512-128_const_noise_t1/binary_ranking_{r}/"


def get_rankings_csv(r):
    folder = get_ranking_folder(r)
    rfn = "results_of_ranking.csv"
    rfn = os.path.join(folder, rfn)
    if not os.path.exists(rfn):
        pass
    else:
        return pd.read_csv(rfn)


def rwd_to_df(rwds, cond):
    rwds = pd.DataFrame(rwds)
    rwds["condition"] = cond
    rwds.columns = ["rwd", "c"]

    return rwds


def create_new_classifier(m_init_params, lr=5e-4, weight_decay=1e-5):
    classifier = eval(m_init_params["MODEL_CLASS"])(**m_init_params)  # this is reward model....
    classifier.cuda()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)  # , weight_decay=0.0001)

    print("total params for reward model")

    nparams = get_n_params(classifier)
    print("nparams: ", f"{nparams:,}")

    # print('total params for dmap net (x3 in reward model)')
    # print(get_n_params(classifier.dmap_net_first))

    return (classifier, optimizer)


def make_dset_naked(X_dmaps, yall, ids, nrs, batch_size=None):
    if nrs == 64:
        X_dmaps = torch.nn.functional.interpolate(X_dmaps, size=[3, 64, 64], mode="nearest")

    dataset = TensorDataset(
        Tensor(X_dmaps).to(device).float(),
        Tensor(yall.float()).to(device).float(),
        Tensor(ids).to(device).int(),
    )

    if batch_size is None:
        batch_size = len(dataset)

    if batch_size > len(dataset):
        batch_size = len(dataset)
    print("len dataset")
    print(len(dataset))
    print("----------")
    dloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dloader


def set_winning_idx(rankings_df):
    rankings_df["winning_idx"] = 0
    rankings_df.loc[rankings_df.rankings == "A", "winning_idx"] = rankings_df.loc[rankings_df.rankings == "A", "first_pair_idx"]
    rankings_df.loc[rankings_df.rankings == "B", "winning_idx"] = rankings_df.loc[rankings_df.rankings == "B", "second_pair_idx"]
    return rankings_df


def get_pairs_and_win_idx_binary(rankings_df):
    first_idx_list = [l for l in rankings_df["first_pair_idx"]]
    second_idx_list = [l for l in rankings_df["second_pair_idx"]]
    winning_idx_list = [l for l in rankings_df["winning_idx"]]
    pairs = [((i, j), w) for i, j, w in zip(first_idx_list, second_idx_list, winning_idx_list)]
    return pairs


def create_instance_pair_binary(
    rankings_df,
    input_pair_win,
    list_of_depths,
    rendering_options=None,
    normalise=False,
    norm_min=None,
    style_codes_dict=None,
):
    input_pair = input_pair_win[0]
    idx1 = input_pair[0]
    idx2 = input_pair[1]
    winning_idx = input_pair_win[1]

    dmap1 = list_of_depths[idx1]
    dmap2 = list_of_depths[idx2]
    unique_id_1 = torch.tensor(idx1).unsqueeze(0)
    unique_id_2 = torch.tensor(idx2).unsqueeze(0)
    stylecode1 = style_codes_dict[idx1]
    stylecode2 = style_codes_dict[idx2]

    assert unique_id_1 != unique_id_2
    assert unique_id_1 == idx1
    assert unique_id_2 == idx2

    first_combination = (dmap1, unique_id_1, stylecode1)
    second_combination = (dmap2, unique_id_2, stylecode2)
    combo_list = [first_combination, second_combination]

    high_score_idx = np.where(np.array(input_pair) == winning_idx)[0][0]
    low_score_idx = np.where(np.array(input_pair) != winning_idx)[0][0]
    high_comb = combo_list[high_score_idx]
    low_comb = combo_list[low_score_idx]
    combo = (high_comb, low_comb)
    high_dmap = combo[0][0].squeeze().unsqueeze(0)
    low_dmap = combo[1][0].squeeze().unsqueeze(0)

    if normalise:
        low_dmap = normalise_dmap_vals(low_dmap, rendering_options, min=norm_min)
        high_dmap = normalise_dmap_vals(high_dmap, rendering_options, min=norm_min)

    depths = [high_dmap, low_dmap]
    scores = [1000, -1000]
    scores = [torch.as_tensor(s).unsqueeze(0) for s in scores]
    unique_ids = [combo[0][1], combo[1][1]]
    stylecodes = [combo[0][2], combo[1][2]]
    return dict(depths=depths, scores=scores, unique_ids=unique_ids, stylecodes=stylecodes)


def create_dset_dict_binary(rankings_df, list_of_depths, style_codes_dict, normalise=False, norm_min=None):
    pairs_win_binary = get_pairs_and_win_idx_binary(rankings_df)

    combos = [
        create_instance_pair_binary(
            rankings_df=rankings_df,
            input_pair_win=p,
            list_of_depths=list_of_depths,
            rendering_options=ffhq_rendering_options,
            normalise=normalise,
            norm_min=norm_min,
            style_codes_dict=style_codes_dict,
        )
        for p in pairs_win_binary
    ]

    depth_maps = [torch.cat(dd, dim=0).unsqueeze(0) for dd in [c["depths"] for c in list(combos)]]
    X_dmaps = torch.cat(depth_maps, dim=0)

    scores = [torch.cat(dd, dim=0).unsqueeze(0) for dd in [c["scores"] for c in list(combos)]]
    yall = torch.cat(scores, dim=0)

    ids = [torch.cat(dd, dim=0).unsqueeze(0) for dd in [c["unique_ids"] for c in list(combos)]]
    ids = torch.cat(ids, dim=0)
    return dict(X_dmaps=X_dmaps, yall=yall, ids=ids)


def create_new_classifier(m_init_params, lr=5e-4, weight_decay=1e-5):
    classifier = eval(m_init_params["MODEL_CLASS"])(**m_init_params)  # this is reward model....
    classifier.cuda()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)  # , weight_decay=0.0001)
    print("total params for reward model")
    nparams = get_n_params(classifier)
    print("nparams: ", f"{nparams:,}")
    return (classifier, optimizer)


def remove_old_classifier_results(da):
    existing_state_dict = glob.glob(os.path.join(da.model_dir, "model_state_dict_*.pth"))
    for i in existing_state_dict:
        os.remove(i)
    existing_losses = glob.glob(os.path.join(da.model_dir, "model_losses*.pth"))
    for i in existing_losses:
        os.remove(i)
    print("purged model dir")


def load_last_model(da):
    model_dir = da.model_dir
    all_models = glob.glob(os.path.join(model_dir, "model_state_dict_*.pth"))
    all_models.sort(key=os.path.getmtime)

    epoch_max = 0
    best_model_path = None
    best_val_loss = float("inf")
    for filename in os.listdir(model_dir):
        if filename.startswith("model_losses_") and filename.endswith(".pth"):
            epoch = int(filename[len("model_losses_") : -len(".pth")])
            checkpoint = torch.load(os.path.join(model_dir, filename))

            if epoch > epoch_max:
                epoch_max = epoch
                best_model_path = os.path.join(model_dir, f"model_state_dict_{epoch_max}.pth")

    # get json of model init params
    json_path = os.path.join(model_dir, "model_init_params.json")
    m_init_params = json.load(open(json_path))

    # Load the best model
    if best_model_path is not None:
        optimal_classifier = eval(m_init_params["MODEL_CLASS"])(**m_init_params)  # this is reward model....
        optimal_classifier.load_state_dict(torch.load(best_model_path))
        optimal_classifier.eval()
        optimal_classifier.cuda()
        print(f"Loaded last model from {best_model_path}, epoch: {epoch_max}")
    else:
        print("No model found in model_dir")

    return optimal_classifier


def concat_predictions_model(optimal_classifier, dataloader, dargs):
    list_of_test_results = []
    for test_example in iter(dataloader):
        zall = -1

        if dargs.model_class == "rwd_model_stylecode":
            win_batch = test_example[3][:, 0, 0, :].unsqueeze(0)
            lose_batch = test_example[3][:, 1, 0, :].unsqueeze(0)

        elif dargs.model_class == "rwd_model_2d_landmarks_98":
            win_batch = test_example[4][:, 0, :, :].reshape(-1, 2 * 98)
            lose_batch = test_example[4][:, 1, :, :].reshape(-1, 2 * 98)

        elif dargs.model_class == "rwd_model_2d_landmarks_98_triple":
            win_batch = test_example[5][:, 0, :, :]  # .reshape(-1,3,:)
            lose_batch = test_example[5][:, 1, :, :]  # .reshape(-1,3,2*98)

        elif dargs.model_class == "rl_decoder_three_inet":
            win_batch = test_example[0][:, 0, :, :]
            lose_batch = test_example[0][:, 1, :, :]

        elif dargs.model_class == "rwd_model_3dmap_vgg_minimal":
            win_batch = test_example[0][:, 0, :, :]
            lose_batch = test_example[0][:, 1, :, :]

            win_batch = dargs.vgg19_to_4096_model(win_batch)
            lose_batch = dargs.vgg19_to_4096_model(lose_batch)

        elif dargs.model_class == "rwd_model_3dmap_vggface2_minimal":
            if dargs.model_kwargs["using_precomputed_dataset"] == True:
                ########
                X_dmap = test_example[0].squeeze(3).to(torch.float32)
                Lengths = test_example[1].to(torch.uint8)

                dmaps_computed = [dargs.vggface2_to_512_model.forward(rcrop_160(xd[:L])) for xd, L in zip(X_dmap, Lengths)]

            else:
                X_dmap = test_example.file_batch.squeeze(3).cuda()
                Lengths = test_example.lens_batch.cuda()
                dmaps_computed = [dargs.vggface2_to_512_model.forward(xd[:L]) for xd, L in zip(X_dmap, Lengths)]
            cpreds = [optimal_classifier.forward(dmc) for dmc in dmaps_computed]
            ordered_combos = [list(itertools.combinations(range(L), 2)) for L in Lengths]

            intermediate_losses = [[torch.cat([cp[o[0]].unsqueeze(0), cp[o[1]].unsqueeze(0)]).flatten() for o in oc] for oc, cp in zip(ordered_combos, cpreds)]

            iml_stacked = torch.vstack([torch.vstack(il) for il in intermediate_losses])
            iml_stacked = iml_stacked.detach().cpu().numpy()
            df_preds = pd.DataFrame(iml_stacked, columns=["R_score_win", "R_score_lose"])
            list_of_test_results.append(df_preds)
            continue

        else:
            assert 1 == 0, "not implemented for concat pred"

        win_pred = optimal_classifier(win_batch).detach().cpu().numpy().flatten()
        lose_pred = optimal_classifier(lose_batch).detach().cpu().numpy().flatten()

        df_preds = pd.DataFrame(np.vstack([win_pred, lose_pred]).T, columns=["R_score_win", "R_score_lose"])

        list_of_test_results.append(df_preds)

    df_preds = pd.concat(list_of_test_results, axis=0, ignore_index=True)

    return df_preds


def print_pred_percentages(df_preds, condition="all"):
    win_pc = df_preds[df_preds["R_score_win"] > df_preds["R_score_lose"]].shape[0] / df_preds.shape[0]
    lose_pc = df_preds[df_preds["R_score_win"] < df_preds["R_score_lose"]].shape[0] / df_preds.shape[0]
    tie_pc = df_preds[df_preds["R_score_win"] == df_preds["R_score_lose"]].shape[0] / df_preds.shape[0]
    print_str = f"{condition}\t win: {win_pc:.3f}, lose: {lose_pc:.3f}, tie: {tie_pc:.3f},\tn={df_preds.shape[0]}"
    return print_str


def plot_rwd_histograms_from_flattened(train_preds_flat, val_preds_flat, test_preds_flat):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    plt.title("Histogram of Reward Scores")

    axs[0].hist(np.hstack([train_preds_flat, val_preds_flat, test_preds_flat]), bins=100)
    axs[0].set_title("All Predictions")
    axs[0].set_xlabel("Reward score")
    axs[0].set_ylabel("Count")

    axs[1].hist(train_preds_flat, bins=100)
    axs[1].set_title("Train Predictions")
    axs[1].set_xlabel("Reward score")
    axs[1].set_ylabel("Count")

    axs[2].hist(val_preds_flat, bins=100)
    axs[2].set_title("Validation Predictions")
    axs[2].set_xlabel("Reward score")
    axs[2].set_ylabel("Count")

    axs[3].hist(test_preds_flat, bins=100)
    axs[3].set_title("Test Predictions")
    axs[3].set_xlabel("Reward score")
    axs[3].set_ylabel("Count")

    plt.tight_layout()

    return fig


def evaluate_model(optimal_classifier, test_loader, train_loader, val_loader, dargs):
    optimal_classifier.eval()
    with torch.no_grad():
        test_preds = concat_predictions_model(optimal_classifier, test_loader, dargs)
        train_preds = concat_predictions_model(optimal_classifier, train_loader, dargs)
        val_preds = concat_predictions_model(optimal_classifier, val_loader, dargs)
        all_preds = pd.concat([train_preds, val_preds, test_preds], axis=0, ignore_index=True)

    tmp_eval_file = os.path.join(dargs.model_dir, "tmp_eval_file.txt")
    with open(tmp_eval_file, "w") as f:
        print(print_pred_percentages(all_preds, condition="all"), file=f)
        print("--------", file=f)
        print(print_pred_percentages(train_preds, condition="train"), file=f)
        print(print_pred_percentages(val_preds, condition="val"), file=f)
        print(print_pred_percentages(test_preds, condition="test"), file=f)
    fig = plot_rwd_histograms(all_preds, train_preds, val_preds, test_preds)

    normalisation_terms = get_nterms(all_preds)
    return (fig, normalisation_terms)


def evaluate_model_return_precomputed_pc_correct(pred_pairs, indiv_scores, dargs):
    train_preds = pred_pairs["train"]
    val_preds = pred_pairs["val"]
    test_preds = pred_pairs["test"]

    train_preds_flat = indiv_scores["train"]
    val_preds_flat = indiv_scores["val"]
    test_preds_flat = indiv_scores["test"]
    # train_preds_flat, val_preds_flat, test_preds_flat
    preds_pc_correct = []
    for df_preds in [train_preds, val_preds, test_preds]:
        pc_correct = df_preds[df_preds["R_score_win"] > df_preds["R_score_lose"]].shape[0] / df_preds.shape[0]
        preds_pc_correct.append(pc_correct)

    keys = ["train_pc_correct", "val_pc_correct", "test_pc_correct"]
    preds_pc_correct = {keys[i]: preds_pc_correct[i] for i in range(len(keys))}

    tmp_eval_file = os.path.join(dargs.model_dir, "tmp_eval_file.txt")

    all_preds = pd.concat([train_preds, val_preds, test_preds])

    with open(tmp_eval_file, "w") as f:
        print(print_pred_percentages(all_preds, condition="all"), file=f)
        print("--------", file=f)
        print(print_pred_percentages(train_preds, condition="train"), file=f)
        print(print_pred_percentages(val_preds, condition="val"), file=f)
        print(print_pred_percentages(test_preds, condition="test"), file=f)

    fig = plot_rwd_histograms_from_flattened(train_preds_flat, val_preds_flat, test_preds_flat)

    # fig=plot_rwd_histograms(all_preds, train_preds, val_preds, test_preds)

    # normalisation_terms=get_nterms(all_preds)
    return (fig, preds_pc_correct)


def convert_scores_to_df(pred):
    tpp = [tp["pairwise_comp"] for tp in pred]
    tppp = torch.cat(tpp, 0).detach().cpu().numpy()

    tppp = pd.DataFrame(tppp)

    tppp.columns = ["R_score_win", "R_score_lose"]

    return tppp


def retrieve_flat_rwd_scores(pred):
    tpp = [torch.cat(tp["rwd_vals"], 0) for tp in pred]
    flat_scores = torch.cat(tpp, 0).flatten().detach().cpu().numpy()

    # tppp=pd.DataFrame(tppp)

    # tppp.columns=['R_score_win','R_score_lose']

    return flat_scores


def evaluate_model_return_pc_correct(optimal_classifier, test_loader, train_loader, val_loader, dargs):
    optimal_classifier.eval()
    with torch.no_grad():
        test_preds = concat_predictions_model(optimal_classifier, test_loader, dargs)
        train_preds = concat_predictions_model(optimal_classifier, train_loader, dargs)
        val_preds = concat_predictions_model(optimal_classifier, val_loader, dargs)
        all_preds = pd.concat([train_preds, val_preds, test_preds], axis=0, ignore_index=True)

    preds_pc_correct = []
    for df_preds in [train_preds, val_preds, test_preds]:
        pc_correct = df_preds[df_preds["R_score_win"] > df_preds["R_score_lose"]].shape[0] / df_preds.shape[0]
        preds_pc_correct.append(pc_correct)

    keys = ["train_pc_correct", "val_pc_correct", "test_pc_correct"]
    preds_pc_correct = {keys[i]: preds_pc_correct[i] for i in range(len(keys))}

    tmp_eval_file = os.path.join(dargs.model_dir, "tmp_eval_file.txt")
    with open(tmp_eval_file, "w") as f:
        print(print_pred_percentages(all_preds, condition="all"), file=f)
        print("--------", file=f)
        print(print_pred_percentages(train_preds, condition="train"), file=f)
        print(print_pred_percentages(val_preds, condition="val"), file=f)
        print(print_pred_percentages(test_preds, condition="test"), file=f)
    fig = plot_rwd_histograms(all_preds, train_preds, val_preds, test_preds)

    normalisation_terms = get_nterms(all_preds)
    return (fig, normalisation_terms, preds_pc_correct)


def get_nterms(all_preds):
    mean = all_preds[["R_score_win", "R_score_lose"]].values.flatten().mean()
    std = all_preds[["R_score_win", "R_score_lose"]].values.flatten().std()
    return dict(mean=mean, std=std)


def rescale_dmap_single(dmap, rescale_size, mode="nearest"):
    dxm = dmap.reshape(-1, 1, 128, 128)
    # dxm=F.interpolate(dxm.type(torch.FloatTensor),size=(rescale_size,rescale_size),mode=mode)
    dxm = F.interpolate(dxm, size=(rescale_size, rescale_size), mode=mode)
    dxm = dxm.reshape(-1, 3, rescale_size, rescale_size)
    return dxm


def load_rwd_mdl(pkl_dir, state_dict=True):
    with open(pkl_dir, "rb") as f:
        rwd_mdl = pickle.load(f)

    m_init_params = rwd_mdl["m_init_params"]
    classifier = eval(m_init_params["MODEL_CLASS"])(**m_init_params)  # this is reward model....

    if state_dict:
        state_dict = rwd_mdl["model_state_dict"]
        classifier.load_state_dict(state_dict)

    classifier.cuda()
    classifier.eval()
    print(f"Loaded best model from {pkl_dir}")

    da = dargs()
    if classifier.reward_model_type == "rwd_model_3dmap_vgg_minimal":
        da.vgg19_to_4096_model = vgg19_to_4096()
        da.vgg19_to_4096_model = da.vgg19_to_4096_model.cuda()

    if classifier.reward_model_type == "rwd_model_3dmap_vggface2_minimal":
        da.vggface2_to_512_model = vggface2_to_512()
        da.vggface2_to_512_model = da.vggface2_to_512_model.cuda()

    if "da_interpolation_mode" not in m_init_params.keys():
        da.interpolation_mode = "bilinear"
    else:
        da.interpolation_mode = m_init_params["da_interpolation_mode"]

    return classifier, da


def get_gaussian_mean_rwds(rwds_unseen):
    mean = np.mean(rwds_unseen)
    std_dev = 1
    num_samples = 1000
    gaussian_samples = np.random.normal(mean, std_dev, num_samples)
    return gaussian_samples


def get_rwds_vgg(reward_model, seeds_dict_unseen, da, check_lims=True):
    rwds_unseen = []

    for s in seeds_dict_unseen.keys():
        dmap = seeds_dict_unseen[s]["dmap"]
        dmap = torch.tensor(dmap).to(device)
        norm_min = -1.0
        dmap = normalise_dmap_vals(dmap, rendering_options=ffhq_rendering_options, min=norm_min, check_lims=check_lims)
        dmap = rescale_dmap_single(dmap, 224)
        dmap = da.vgg19_to_4096_model(dmap)
        rwd = reward_model(
            dmap,
        ).item()

        rwds_unseen.append(rwd)

        if len(rwds_unseen) % 100 == 0:
            print(len(rwds_unseen))

    return rwds_unseen


def get_rwds_vggface2(reward_model, seeds_dict_unseen, da, check_lims=True):
    rwds_unseen = []

    for s in seeds_dict_unseen.keys():
        dmap = seeds_dict_unseen[s]["dmap"]
        dmap = torch.tensor(dmap).to(device)
        norm_min = -1.0
        dmap = normalise_dmap_vals(dmap, rendering_options=ffhq_rendering_options, min=norm_min, check_lims=check_lims)
        dmap = rescale_dmap_single(dmap, 160, mode=da.interpolation_mode)
        dmap = da.vggface2_to_512_model(dmap)
        rwd = reward_model(
            dmap,
        ).item()

        rwds_unseen.append(rwd)

        if len(rwds_unseen) % 100 == 0:
            print(len(rwds_unseen))

    return rwds_unseen


def get_embeddings_vggface2(reward_model, seeds_dict_unseen, da, check_lims=True):
    rwds_unseen = []

    for s in seeds_dict_unseen.keys():
        dmap = seeds_dict_unseen[s]["dmap"]
        dmap = torch.tensor(dmap).to(device)
        norm_min = -1.0
        dmap = normalise_dmap_vals(dmap, rendering_options=ffhq_rendering_options, min=norm_min, check_lims=check_lims)
        dmap = rescale_dmap_single(dmap, 160)
        dmap = da.vggface2_to_512_model(dmap)
        rwd = reward_model.feature_embedding(
            dmap,
        )

        rwds_unseen.append(rwd)

        if len(rwds_unseen) % 100 == 0:
            print(len(rwds_unseen))

    return rwds_unseen


def get_raw_embedding_vggface2(reward_model, dmap, da, check_lims=True):
    # rwds_unseen=[]

    # for s in seeds_dict_unseen.keys():#
    # dmap=seeds_dict_unseen[s]['dmap']
    dmap = torch.tensor(dmap).to(device)
    norm_min = -1.0
    dmap = normalise_dmap_vals(dmap, rendering_options=ffhq_rendering_options, min=norm_min, check_lims=check_lims)
    dmap = rescale_dmap_single(dmap, 160)
    dmap = da.vggface2_to_512_model(dmap)
    embedding = reward_model.feature_embedding(
        dmap,
    )
    return embedding


def get_rwd_vggface2_nodict(reward_model, dmap, da, check_lims=True):
    # rwds_unseen=[]

    # for s in seeds_dict_unseen.keys():
    # dmap=seeds_dict_unseen[s]['dmap']
    # dmap=torch.tensor(dmap).to(device)
    norm_min = -1.0
    dmap = normalise_dmap_vals(dmap, rendering_options=ffhq_rendering_options, min=norm_min, check_lims=check_lims)
    dmap = rescale_dmap_single(dmap, 160)
    dmap = da.vggface2_to_512_model(dmap)
    rwd = reward_model(
        dmap,
    )
    return rwd


# --------------------------------------------------------
# New dataloader stuff

import imageio.v3 as iio
import torch


def create_pt_fn(ddir, ot, seed):
    retval = os.path.join(ddir, f"{ot}_s_{seed}.pt")
    return retval


def assemble_triple_lmks(seed, ddir):
    fns = [os.path.join(ddir, f"triple_rgb_lmks_98_s_{seed}_{k}.pt") for k in range(3)]
    lmks = [torch.load(fn).unsqueeze(0) for fn in fns]
    lmks_trip = torch.cat(lmks, 0)
    return lmks_trip


def assemble_single_lmks(seed, ddir):
    fn = os.path.join(ddir, f"triple_rgb_lmks_98_s_{seed}_1.pt")
    lmks = torch.load(fn)
    return lmks


def assemble_triple_rgb(seed, ddir):
    fns = [os.path.join(ddir, f"triple_rgb_s_{seed}_{k}.jpg") for k in range(3)]
    rgbs = [torch.tensor(iio.imread(fn), device=torch.device("cuda")).permute(2, 0, 1).unsqueeze(0) for fn in fns]
    rgbs_trip = torch.cat(rgbs, 0)
    return rgbs_trip


def assemble_single_rgb(seed, ddir):
    fn = os.path.join(ddir, f"triple_rgb_s_{seed}_{1}.jpg")
    rgb = torch.tensor(iio.imread(fn), device=torch.device("cuda")).permute(2, 0, 1).unsqueeze(0)
    return rgb


def normalise_dmap_vals(dmap, rmin=2.25, rmax=3.3, min=-1.0, check_lims=True):
    # rmin=rendering_options['ray_start']
    # rmax=rendering_options['ray_end']
    if min == 0.0:
        dmap = (dmap - rmin) / (rmax - rmin)
        dm_min = 0.0
    elif min == -1.0:
        dmap = (((dmap - rmin) / (rmax - rmin)) * 2) - 1
        dm_min = -1.0

    dmap[dmap < dm_min] = dm_min
    dmap[dmap > 1.0] = 1.0
    # dmap.clamp_(min=dm_min,max=1.0) #clamp between vals

    if check_lims:
        assert dmap.max() <= 1.0
        assert dmap.min() >= dm_min
    return dmap


def normfunc(dmap):
    retval = ((dmap - 2.25) / (3.3 - 2.25)) * 2 - 1
    retval[retval < -1.0] = -1.0
    retval[retval > 1.0] = 1.0

    return retval


upsample_rnd_crop_160_normalise = v2.Compose(
    [
        v2.Resize(
            size=(256, 256),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),  # Or Resize(antialias=True)
        v2.RandomCrop(size=160),
        v2.Lambda(normfunc),
    ]
)


class dset_smulti_stream(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, dsets):
        "Initialization"
        dset_lens = [len(d) for d in dsets]  # get len of dsets....

        assert len(set(dset_lens)) == 1, "error more than 1 value for sizes of each dset..."
        self.dsets = dsets
        self.length = dset_lens[0]
        self.n_dsets = len(dsets)

    def __len__(self):
        "Denotes the total number of samples"
        return self.length

    def __getitem__(self, index):
        "Generates one sample of data"
        dset_return = [self.dsets[k][index] for k in range(self.n_dsets)]

        return dset_return



# entire_spec_fn = "/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/000_RLHF_AM/mesh_outputs_spec_20_09_2023.ods"
# expname = "rlhf_meshes_ffhq512-128_const_noise_t1_augment"
# expnames_spec = read_ods(entire_spec_fn, sheet="eg3d_model_experiment_settings", headers=True).set_index("condition")


# def ddir_func(query_val):
#     e_conds = expnames_spec[(~(expnames_spec.seed_start).isna()) & (~(expnames_spec.seed_end).isna())]  # .set_index('condition')
#     query_val = int(query_val)
#     starts = e_conds.seed_start
#     ends = e_conds.seed_end
#     for e in e_conds.index:
#         if starts.loc[e] <= query_val <= ends.loc[e]:
#             relevant_condition = e
#             expname = str(relevant_condition)
#             ddir = f"/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/000_RLHF_AM/rlhf_meshes/{expname}"
#             return ddir
#     assert False, f"error an appropriate condition not found for seed query val {query_val}"


# ---------------------------------------------------
