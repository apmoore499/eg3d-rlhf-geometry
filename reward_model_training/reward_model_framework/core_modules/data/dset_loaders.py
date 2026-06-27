# #get the permutation
# # Open an Image
# import copy
# import glob
# import itertools
# import os
# import sys

# import matplotlib as mpl
# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import torch
# import torch_geometric
# import torch_geometric.transforms as T
# import torchvision.transforms.v2 as v2
# import tqdm
# import trimesh
# import torch.multiprocessing as mp
# mp.set_start_method('spawn')
# Open an Image
# get the permutation
# Open an Image
# import rootutils

# this will automatically find the .project-root file upwards and add the root to sys.path
# root = rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)
# print("Repo root set to:", root)


import math
import os
import random
import sys
from pathlib import Path

# import core_modules.data.misc_small_utils.create_pt_fn as create_pt_fn
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import sys
import autoroot  # noqa: F401

# sys.path.append("/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/eg3d")
# breakpoint()
from eg3d.training.volumetric_rendering.ray_sampler import RaySampler

from .misc_small_utils import (
    assemble_triple_lmks,
    assemble_single_lmks,
    assemble_single_rgb,
    assemble_triple_rgb,
    create_pt_fn,
    get_lmks_mask_aw98_no_edit,
    ddir_func,
)

from .aug_and_tforms.augtforms_pointcloud import (
    ensemble_pointcloud_transforms,
)
from . import centroid_patches as cp


from .io_geometry_utils import imd_to_xyz_with_radius_cutoff
import core_modules
from core_modules.utils import depth_to_pcd as dpcd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
REWARD_MODEL_TRAINING_DIR = PROJECT_ROOT / "reward_model_training"
DATASET_CACHE_DIR = REWARD_MODEL_TRAINING_DIR / "datasets"

# AW98 facial-landmark indices for the keypoint regions used by the patch dtypes
# (aw98_patch_*). Indexed into the 98-point AW98 scheme. Kept here (not inlined as
# bare `np.array([[54],[90],...])`) so the regions are self-documenting.
AW98_REGION_LANDMARKS = {
    "nose": 54,
    "mouth": 90,
    "left_eye": 96,
    "right_eye": 97,
}


# main helper claass to load a specific data type eg depth map, point cloud, sigma field
class dset_single_stream_ordered_minimal(torch.utils.data.Dataset):
    """One ranked batch of a single representation (dtype), for the reward model.

    This is the core data-parsing class. Given a ranking row (an ordered list of
    seeds, best->worst), it loads each seed's sample for `self.dtype`, applies the
    configured preprocessing/augmentation, and pads the batch to a fixed width so
    rows of different lengths collate. The per-seed load + preprocessing for every
    representation lives in `return_single_data` (a long per-dtype dispatch).

    `self.dtype` is one of the keys in configs/data/data_defaults.yaml `dset_dict`
    (sigma_field_256/128/64/512, point_cloud_entire and pcd_* variants,
    single_dmap/triple_dmap, the landmark/centroid types, ...); that file documents
    what each one is.

    dset_version: "one" = legacy binary-feature loader; "three" = current
    (top+bottom pairs, i.e. datamodule_third); "inverted" = reversed-rank
    anti-preference experiment.

    Design note: `return_single_data` deliberately fuses three stages per dtype --
    raw load from disk, format conversion (e.g. depth map -> point cloud), and
    downstream processing (subsampling / augmentation). The underlying pieces are
    already modular -- conversion delegates to `utils/depth_to_pcd.py`, and
    augmentation is a Hydra-configured callable injected at construction -- so what
    is fused here is mainly the per-dtype dispatch, not the logic itself. A cleaner
    design would split these into separate Hydra-configured stages
    (load -> convert -> transform); they are left fused as the verified, working
    path, reflecting how the project accreted representations over time rather than
    a target architecture.

    Augmentation contract (two distinct knobs):
      * `self.augmentations` -- the always-on transform applied to EVERY batch,
        right at the end just before returning (e.g. a normalising transform for
        sigma fields). Set once at construction.
      * `pre_augmentation` (on `return_single_data`) / `pre_augmentations` (on the
        sigma loaders) -- OPTIONAL extra augs applied BEFORE `self.augmentations`.
        Default None (no extra). Order when present: `pre_augmentations` first,
        then `self.augmentations`.
    (Renamed from the colliding `augmentation`/`augmentations` and fixed a latent
    double-application: `pre_augmentation` previously defaulted to
    `self.augmentations`, so the always-on transform ran twice.)
    """

    def __init__(self, all_combined_rankings, dtype, ddir_func, seed_func, augmentations=None, goodmesh_augment=None, dset_partition="", batch_augmentations=None, map_on=None, **kwargs):
        super().__init__()
        "Initialization"
        #!!!! MUST BE ORDERED OR METHOD WILL FAIL!!!!#
        self.all_combined_rankings_ordered = all_combined_rankings  # this is the list of ordered combo for the current batch. precomputed.
        self.dtype = dtype
        self.ddir_func = ddir_func
        self.seed_func = seed_func
        # self.device=torch.device('cuda')
        self.transforms = augmentations if augmentations is not None else nn.Identity()
        # always-on augs, applied LAST to every batch (see class docstring contract)
        self.augmentations = augmentations if augmentations is not None else nn.Identity()

        # self.set_sorted_unique_seeds()
        self.n_point_samples_per_pcd_batch = 16384

        canon_cam = core_modules.data.misc_small_utils.get_canonical_dmap_cams()
        self.cam2world_matrix = canon_cam["cam2world_matrix"]
        self.intrinsics = canon_cam["intrinsics"]
        self.ray_sampler_static = RaySampler()
        self.lim_pts = dpcd.build_lim_pts()

        # self.pcd_transform=get_pcd_transform_plain(n_points=self.n_point_samples_per_pcd_batch)
        self.radius_for_pcd_lmks = 9

        def _aug_applicable():
            if isinstance(self.augmentations, nn.Identity):
                return False
            return self.dtype in ("single_dmap", "triple_dmap")

        self.using_transform = kwargs.get("using_transforms", False) or _aug_applicable()
        self.use_transforms = False

        self.using_wandb = False

        self.list_of_good_seeds = []  # if we are using goodseeds #self.goodmesh_train

        self.list_of_bad_seeds = []  # if we are using some bad seeds

        self.goodseed_idx = 0  # init seed for rndm, detrminisc
        self.badseed_idx = 0  # init seed for rndm,d eterministc

        self.current_good_seeds_perm = []
        self.current_bad_seeds_perm = []

        self.ept = ensemble_pointcloud_transforms(1024)
        self.ept.reset_random_domains_for_train()

        # self.set_lim_pts()

        self.map_loc = "cpu"

        if goodmesh_augment is not None:
            self.good_transform = goodmesh_augment

        self.dset_partition = dset_partition

        # self.rng = np.random.default_rng(seed=10000) #nb this will make the mminibatch random good and bad seeds fixed for each worker in batch.

        # overwrite augmented with a very bad mesh....
        aug_bad = core_modules.data.custom_transforms.smooth_front_of_face_transform()

        smooth_face = core_modules.data.custom_transforms.smooth_front_of_face_transform()  # k=-2,quantile=60,ps=0.1)
        smooth_mouth_corner = core_modules.data.custom_transforms.mouth_corner_smoothe()
        aug = core_modules.data.custom_transforms.smooth_front_of_face_transform_bad()
        aug_good = core_modules.data.custom_transforms.transforms_composition_helper(dict(smooth_front_of_face_transform=smooth_face, mouth_corner_smoothe=smooth_mouth_corner), dtype="sigma_field_256")

        self.smooth_mouth_corner = smooth_mouth_corner
        self.smooth_face = smooth_face

        self.aug_good_function = self.smooth_face.forward
        self.aug_bad_function = core_modules.data.custom_transforms.smooth_front_of_face_transform_bad()

        self.using_badseeds = True

        self.include_goodseed = kwargs["include_goodseed"]

        self.goodseed_pred_prob = 0.0
        self.badseed_pred_prob = 0.0
        self.augment_orig_prob = 0.0
        self.augment_good_prob = 0.0
        self.keep_all_orig_prob = 0.0
        self.keep_all_good_prob = 0.0
        self.sample_seed_pair_from_rankings = False

        if batch_augmentations is not None:
            # breakpoint()
            self.goodseed_pred_prob = getattr(batch_augmentations, "goodseed_pred_prob", 0.0)
            self.badseed_pred_prob = getattr(batch_augmentations, "badseed_pred_prob", 0.0)
            self.augment_orig_prob = getattr(batch_augmentations, "augment_orig_prob", 0.0)
            self.augment_good_prob = getattr(batch_augmentations, "augment_good_prob", 0.0)
            self.keep_all_orig_prob = getattr(batch_augmentations, "keep_all_orig_prob", 0.0)
            self.keep_all_good_prob = getattr(batch_augmentations, "keep_all_good_prob", 0.0)
            self.sample_seed_pair_from_rankings = getattr(batch_augmentations, "sample_seed_pair_from_rankings", False)

        self.map_on = map_on

        # self.quantile=quantile
        # self.k=k
        # self.ps=ps#0.1

        self.worker_seeds = {}

        if "dset_version" in kwargs.keys():
            self.dset_version = kwargs["dset_version"]

    def reset_goodseeds_perm(self):
        # ensure determinsitce
        torch.manual_seed(self.goodseed_idx)

        # increment seed

        self.goodseed_idx = self.goodseed_idx + 1

        # makt eh perm
        self.current_good_seeds_perm = torch.randperm(len(self.list_of_good_seeds))

    def get_new_goodseed(self):
        if len(self.current_good_seeds_perm) == 0:
            self.reset_goodseeds_perm()

        self.current_good_seeds_perm, sel_seed_idx = self.current_good_seeds_perm[:-1], self.current_good_seeds_perm[-1]
        sel_seed = self.list_of_good_seeds[sel_seed_idx]

        worker_info = torch.utils.data.get_worker_info()

        return sel_seed

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.all_combined_rankings_ordered)

    def update_attrs_from_run_dict(self, run_dict):
        for k in run_dict.keys():
            setattr(self, k, run_dict[k])

        # self.pcd_transform=get_pcd_transform_plain(n_points=self.n_point_samples_per_pcd_batch)

        return self

    def return_single_data(self, seed, pre_augmentation=None):
        """Load + preprocess one seed's sample for `self.dtype`.

        Long per-dtype dispatch: each representation (sigma_field_*, point_cloud_*/
        pcd_*, *_dmap, landmark/centroid types, ...) has its own branch that reads
        the sample from disk via `self.ddir_func` and applies the right transforms.
        See the dtype catalog in configs/data/data_defaults.yaml.
        """
        o = self.dtype
        # `pre_augmentation`: optional per-call EXTRA augmentation, applied (inside the
        # sigma loaders) BEFORE the always-on `self.augmentations`. Defaults to None
        # (no extra); `self.augmentations` is applied regardless. (Previously this
        # defaulted to `self.augmentations`, which double-applied it -- removed.)

        # check if seed in goodmeshesfn, use the hack to randomly resample the seed.

        seeds_in_batch = [seed]  # to make compatible with earlier data...
        if o == "triple_rgb_lmks_98":
            files = [assemble_triple_lmks(ddir=self.ddir_func(s), seed=self.seed_func(s)) for s in seeds_in_batch]  # (3, 98, 2)

        elif o == "canonical_rgb_lmks_98":
            files = [assemble_single_lmks(ddir=self.ddir_func(s), seed=self.seed_func(s)) for s in seeds_in_batch]  # (1, 98, 2) canonical view only

        elif o == "triple_rgb":
            files = [assemble_triple_rgb(ddir=self.ddir_func(s), seed=self.seed_func(s)) for s in seeds_in_batch]

        elif o == "single_rgb":
            files = [assemble_single_rgb(ddir=self.ddir_func(s), seed=self.seed_func(s)) for s in seeds_in_batch]

        elif o == "single_rgb_and_dmap":
            rgb_files = [assemble_single_rgb(ddir=self.ddir_func(s), seed=self.seed_func(s)) for s in seeds_in_batch]

            # do a transform to it.....
            # -----

            # -----
            rgb_files = [((r + 1 / 2) * 0.95 + 2.35)[None, None, ...].view(1, 3, 512, 512) for r in rgb_files]
            rgb_files = [torch.nn.functional.interpolate(r, size=(128, 128), mode="bilinear", align_corners=False) for r in rgb_files]
            # resized_image = F.interpolate(normalized_image, size=(128, 128), mode='bilinear', align_corners=False)

            fns = [create_pt_fn(ddir=self.ddir_func(s), ot="triple_dmap", seed=self.seed_func(s)) for s in seeds_in_batch]
            dmap_files = [torch.load(f, map_location=torch.device("cpu"))[1][None, None, ...] for f in fns]  # take idx=1 for the middle/canonical depth map, unsqueeze it or else!!

            # cat it

            files = [torch.cat((rgb.view(1, 1, 3, 128, 128), dmap), 2) for rgb, dmap in zip(rgb_files, dmap_files)]

        elif o == "single_dmap":  # reduce dependency on extra functions ie load_canonical_dmap or load_single_dmap
            map_on = self.map_on if self.map_on is not None else "cpu"
            fns = [create_pt_fn(ddir=self.ddir_func(s), ot="triple_dmap", seed=self.seed_func(s)) for s in seeds_in_batch]
            files = [torch.load(f, map_location=torch.device(map_on))[1][None, None, ...] for f in fns]  # take idx=1 for the middle/canonical depth map, unsqueeze it or else!!
            # from torchvision.transforms import v2

            # offset=torch.ones_like(files[0])
            # import random
            # rand_offset=random.random()*0.10-0.05

            # files=[f+rand_offset for f in files]
            # cc=v2.CenterCrop(size=40)
            # pp=v2.Pad(44)
            # #cc=v2.CenterCrop(size=4)
            # #pp=v2.Pad(62)

            # #pp(cc(tti))
            # files = [pp(cc(f)) for f in files]

            if self.using_transform:
                files = [self.transforms(f) for f in files]
            else:
                files = files

        elif o == "triple_dmap":
            map_on = self.map_on if self.map_on is not None else "cpu"

            fns = [create_pt_fn(ddir=self.ddir_func(s), ot=o, seed=self.seed_func(s)) for s in seeds_in_batch]
            files = [torch.load(f, map_location=torch.device(map_on)) for f in fns]
            if self.using_transform:

                def apply_tf(d):
                    if isinstance(d, (list, tuple)):
                        views = []
                        for v in d:
                            if isinstance(v, torch.Tensor):
                                v_in = v.unsqueeze(0).unsqueeze(0) if v.ndim == 2 else v
                                views.append(self.transforms(v_in))
                        if len(views) == 0:
                            return d
                        # views are shaped (1,1,H,W); concatenate along channel to get (1, N, H, W)
                        return torch.cat(views, dim=1)
                    return self.transforms(d)

                files = [apply_tf(f) for f in files]
            else:
                files = files

        elif o == "nose_512":
            files = [self.sigma_nose_512(s, map_on="cpu", transforms=self.use_transforms, pre_augmentations=pre_augmentation) for s in seeds_in_batch]

            files = [f.unsqueeze(0) for f in files]

        elif o == "sigma_field_256":
            map_on = self.map_on if self.map_on is not None else "cpu"

            files = [self.sigma_field_256_entire(s, map_on=map_on, transforms=self.use_transforms, pre_augmentations=pre_augmentation) for s in seeds_in_batch]
            files = [f.unsqueeze(0) for f in files]

        elif o == "sigma_field_128":
            map_on = self.map_on if self.map_on is not None else "cpu"
            files = [self.sigma_field_128_entire(s, map_on=map_on, transforms=self.use_transforms, pre_augmentations=pre_augmentation) for s in seeds_in_batch]
            files = [f.unsqueeze(0) for f in files]

        elif o == "sigma_field_64":
            files = [self.sigma_field_64_entire(s, map_on="cpu", transforms=self.use_transforms, pre_augmentations=pre_augmentation) for s in seeds_in_batch]  # sigma fields stored raw as float16 (values are well below the ~65504 ceiling; no scale-by-1000)
            files = [f.unsqueeze(0) for f in files]

        elif o == "pcd_nose_combined":
            # One PCD that oversamples the nose: centre + mean-scale the full depth-map
            # PCD, then concatenate a downsampled GENERAL sample with a downsampled
            # NOSE-RADIUS sample. Result is a single cloud (general ++ nose), permuted
            # to (1, 3, N) with N = n_points + min(nose_count, n_points). (The
            # centre/mean-scale-before-nose-subset is by design but never validated.)
            files = [self.modules_depthmap_to_pcd(s) for s in seeds_in_batch]
            files = [self.ept.center_points(f) for f in files]
            files = [self.ept.mean_scale_pts(f) for f in files]
            nose_points = [self.subset_from_nose_radius(f) for f in files]  # subset from the nose radius
            nose_points = [self.ept.downsample_pcd_points(f) for f in nose_points]

            general_points = [self.ept.downsample_pcd_points(f) for f in files]

            files = [torch.cat([g, n], dim=0) for g, n in zip(general_points, nose_points)]

            files = [f.permute(1, 0).unsqueeze(0) for f in files]

        elif o == "ws_code_view_conditioned":
            fns = [create_pt_fn(ddir=self.ddir_func(s), ot="ws_code_view_conditioned", seed=self.seed_func(s)) for s in seeds_in_batch]

            files = [torch.load(f, map_location=torch.device("cpu"))[:, 0, :] for f in fns]

        # ---- AW98 keypoint / "centroids" family ----
        # AW98 = the 98-point facial-keypoint scheme. aw98_landmark_to_pcd_index
        # maps each AW98 2D landmark to the nearest point in the depth-map-derived
        # PCD; a "centroid" is that PCD point. The dtypes below build on that:
        #   * aw98_3d_lmks       : the 98 keypoints as 3D coords sampled FROM the PCD,
        #                          shape (1, 98, 3). Depends only on AW98 2D landmarks
        #                          + the depth map. (The old file-backed 3D-landmark
        #                          loader was removed — it needed un-exported data.)
        #   * aw98_patch_*       : small image-shaped PATCHES of the PCD cropped
        #                          around keypoint centroids, built for a CNN/CoAtNet:
        #       - aw98_patch_rgb_4region_32  : 4 regions (nose/mouth/eyes), 32x32,
        #                                      9ch (xyz+rgb+freq). (1, 4, 9, 32, 32)
        #       - aw98_patch_geom_nose_8     : nose only, 8x8, 6ch (xyz+freq).
        #                                      (1, 1, 6, 8, 8)
        #       - aw98_patch_normals_nose_8  : nose only, 8x8, 9ch (xyz+normals+freq).
        #                                      (1, 1, 9, 8, 8)
        # Region landmark indices come from AW98_REGION_LANDMARKS (module scope).
        # Consumers: aw98_3d_lmks backs the archived aw98 MLP/transformer configs;
        # the patch dtypes were built for CoAtNet_centroids
        # (models/modules_coatnet.py), which has no Hydra config. All remain covered
        # by tests/test_data_types_loadable.py.
        elif o == "aw98_3d_lmks":
            pcds = [self.modules_depthmap_to_pcd(s) for s in seeds_in_batch]  # B,N,3

            seed_centroids = [
                self.aw98_landmark_to_pcd_index(
                    seed=s,
                    nrs=128,
                )
                for s in seeds_in_batch
            ]

            # pcd[dict_of_centroids[g]]

            sc_keys = seed_centroids[0].keys()

            coords_3d = [[pcd[sd_dict[k]] for k in sc_keys] for pcd, sd_dict in zip(pcds, seed_centroids)]

            coords_3d = [torch.vstack(c3d) for c3d in coords_3d]

            files = [f.unsqueeze(0) for f in coords_3d]  # should be shape B,98,3

        elif o == "aw98_patch_rgb_4region_32":
            # 4 keypoint regions (nose/mouth/eyes), each a 32x32 PCD patch with 9
            # channels (xyz + rgb + freq-magnitude). Return shape (1, 4, 9, 32, 32).
            pcds = [self.modules_depthmap_to_pcd(s) for s in seeds_in_batch]  # B,N,3

            seed_centroids = [
                self.aw98_landmark_to_pcd_index(
                    seed=s,
                    nrs=128,
                )
                for s in seeds_in_batch
            ]

            rgb_files = [assemble_single_rgb(ddir=self.ddir_func(s), seed=self.seed_func(s)) for s in seeds_in_batch]
            rgb_files = [torch.nn.functional.interpolate(r[None, None, ...].view(1, 3, 512, 512), size=(128, 128), mode="bilinear", align_corners=False).squeeze(0).reshape(3, -1).permute(1, 0) for r in rgb_files]

            all_patches = []

            region_lmks = np.array(list(AW98_REGION_LANDMARKS.values()))  # nose, mouth, left eye, right eye

            for pcd, dict_of_centroids, rgbs, s in zip(pcds, seed_centroids, rgb_files, seeds_in_batch):
                centroids_list = []

                for g in region_lmks:
                    centroids_list.append(pcd[dict_of_centroids[g]])

                centroids = torch.vstack(centroids_list)

                patches = cp.get_processed_patches_rgb(pcd, rgbs, centroids=centroids, patch_size=len(centroids), point_size=1024, input_channel=9, input_size=32)

                all_patches.append(patches.unsqueeze(0))

            del rgb_files
            del seed_centroids
            del pcds
            del seeds_in_batch

            files = all_patches

        elif o == "aw98_patch_normals_nose_8":
            # nose keypoint only, 8x8 PCD patch with 9 channels (xyz + surface
            # normals + freq-magnitude). Return shape (1, 1, 9, 8, 8).
            pcds = [self.modules_depthmap_to_pcd(s) for s in seeds_in_batch]  # B,N,3

            pcds = [cp.normalize_point_cloud(p) for p in pcds]

            lmks_aw98 = [get_lmks_mask_aw98_no_edit(s) for s in seeds_in_batch]

            lmks_aw98 = [self.jitter_aw98_lmks(lmk) for lmk in lmks_aw98]

            seed_centroids = [self.aw98_landmark_to_pcd_index(seed=s, nrs=128, lmks_aw98=lmk_batch) for s, lmk_batch in zip(seeds_in_batch, lmks_aw98)]

            all_patches = []

            region_lmks = np.array([AW98_REGION_LANDMARKS["nose"]])

            for pcd, dict_of_centroids, s in zip(pcds, seed_centroids, seeds_in_batch):
                centroids_list = []

                for g in region_lmks:
                    centroids_list.append(pcd[dict_of_centroids[g]])

                centroids = torch.vstack(centroids_list)

                patches = cp.get_processed_patches_normals(pcd, centroids=centroids, patch_size=len(centroids), point_size=64, input_channel=9, input_size=8, add_freq=True, center_at_centroid=False)

                all_patches.append(patches.unsqueeze(0))

            del seed_centroids
            del pcds
            del seeds_in_batch

            files = all_patches

        elif o == "aw98_patch_geom_nose_8":
            # nose keypoint only, 8x8 PCD patch with 6 channels (xyz + freq-magnitude),
            # geometry only (no colour). Return shape (1, 1, 6, 8, 8).
            pcds = [self.modules_depthmap_to_pcd(s) for s in seeds_in_batch]  # B,N,3

            lmks_aw98 = [get_lmks_mask_aw98_no_edit(s) for s in seeds_in_batch]

            lmks_aw98 = [self.jitter_aw98_lmks(lmk) for lmk in lmks_aw98]

            seed_centroids = [self.aw98_landmark_to_pcd_index(seed=s, nrs=128, lmks_aw98=lmk_batch) for s, lmk_batch in zip(seeds_in_batch, lmks_aw98)]

            region_lmks = np.array([AW98_REGION_LANDMARKS["nose"]])
            files = cp.normalize_pcd_and_get_processed_patches_no_colour(pcds, seed_centroids, region_lmks)

        elif o == "aw98_patch_geom_all98_8":
            # ALL 98 AW98 keypoints as centroids, each an 8x8 PCD patch with 6
            # channels (xyz + freq-magnitude), geometry only. Return shape
            # (1, 98, 6, 8, 8).
            pcds = [self.modules_depthmap_to_pcd(s) for s in seeds_in_batch]  # B,N,3

            lmks_aw98 = [get_lmks_mask_aw98_no_edit(s) for s in seeds_in_batch]

            lmks_aw98 = [self.jitter_aw98_lmks(lmk) for lmk in lmks_aw98]

            seed_centroids = [self.aw98_landmark_to_pcd_index(seed=s, nrs=128, lmks_aw98=lmk_batch) for s, lmk_batch in zip(seeds_in_batch, lmks_aw98)]

            region_lmks = np.arange(98)  # all AW98 keypoints
            files = cp.normalize_pcd_and_get_processed_patches_no_colour(pcds, seed_centroids, region_lmks)

        elif o == "point_cloud_entire":
            # Full depth-map PCD (N = 128*128 = 16384 pts). The per-split transform
            # (`self.augmentations`, from the data/augmentations config) does the
            # subsample + normalise + train-only augment on the (N,3) cloud; with
            # the no-aug default it's Identity, so the full cloud passes through.
            map_on = self.map_on if self.map_on is not None else "cpu"

            files = [self.modules_depthmap_to_pcd(s, map_on=map_on) for s in seeds_in_batch]  # B,N,3

            files = [self.augmentations(f) for f in files]  # (N,3) -> (M,3)
            files = [f.permute(1, 0).unsqueeze(0) for f in files]  # B,3,N

        return files[0]

    def sample_one_item(self, n):
        # Your sampling logic here
        return random.randint(1, n)  # Example sampling logic

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Single-process data loading
            worker_id = 0
        else:
            worker_id = worker_info.id

        if worker_id not in self.worker_seeds:
            # Initialize a new random number generator for this worker
            self.worker_seeds[worker_id] = np.random.RandomState(seed=worker_id)

        # Use the worker-specific random number generator
        rng = self.worker_seeds[worker_id]
        # rng=self.rng #replicating cosine etc.

        ordered_batch = self.all_combined_rankings_ordered[index]
        batch_len = len(ordered_batch[ordered_batch != -1])
        padded_vals_len = len(ordered_batch) - batch_len
        seeds_in_batch = [int(o) for o in ordered_batch[:batch_len]]  # .astype(int)  # .unique()

        # self.single_augment_prob = 0.1  # flag for only if we have good/bad transform

        # self.goodseed_pred_prob = 1.0
        # self.badseed_pred_prob = 1.0
        # self.augment_orig_prob = 0.5
        # self.augment_good_prob = 0.5
        # self.keep_all_orig_prob = 1.0
        # self.keep_all_good_prob = 1.0

        # self.goodseed_pred_prob = 1.0
        # self.badseed_pred_prob = 1.0
        # self.augment_orig_prob = 0.5
        # self.augment_good_prob = 0.5
        # self.keep_all_orig_prob = 1.0
        # self.keep_all_good_prob = 1.0

        with_good = hasattr(self, "good_transform") and self.good_transform is not None and self.dset_version != "first"
        original_seeds_in_batch = seeds_in_batch
        if with_good:
            # self.single_augment_prob = 0.1  # flag for only if we have good/bad transform

            # self.goodseed_pred_prob = 1.0
            # self.badseed_pred_prob = 1.0
            # self.augment_orig_prob = 0.5
            # self.augment_good_prob = 0.5
            # self.keep_all_orig_prob = 1.0 #0.5
            # self.keep_all_good_prob = 0.5

            # sample the seeds_in_batch depending on how mnay...

            if self.sample_seed_pair_from_rankings:
                if len(seeds_in_batch) == 2:
                    original_seeds_in_batch = seeds_in_batch

                elif len(seeds_in_batch) > 2:
                    # sib = [self.rng.choice(seeds_in_batch, size=1).item()]
                    # original_seeds_in_batch = sib  # max len 2
                    sel_seed_idx = rng.choice(list(range(len(seeds_in_batch))), size=2)  # .item()
                    sel_seed_idx.sort()
                    sib = [seeds_in_batch[i] for i in sel_seed_idx]
                    original_seeds_in_batch = sib  # max len 2

                # elif len(seeds_in_batch) == 3:
                #     sib = [self.rng.choice(seeds_in_batch, size=1).item()]
                #     original_seeds_in_batch = sib  # max len 2
                #     # sel_seed_idx = self.rng.choice(list(range(len(seeds_in_batch))), size=2)  # .item()
                #     # sel_seed_idx.sort()
                #     # sib = [seeds_in_batch[i] for i in sel_seed_idx]
                #     # original_seeds_in_batch = sib  # max len 2

                #     #original_seeds_in_batch = sib  # max len 2

                # elif 3 < len(seeds_in_batch) <= 6:
                #     sel_seed_idx = self.rng.choice(list(range(len(seeds_in_batch))), size=2)  # .item()
                #     sel_seed_idx.sort()
                #     sdiff = sel_seed_idx[1] - sel_seed_idx[0]

                #     while sdiff <= 1:
                #         sel_seed_idx = self.rng.choice(list(range(len(seeds_in_batch))), size=2)  # .item()
                #         sel_seed_idx.sort()
                #         sdiff = sel_seed_idx[1] - sel_seed_idx[0]

                #     sib = [seeds_in_batch[i] for i in sel_seed_idx]

                #     original_seeds_in_batch = sib  # max len 2

            else:
                original_seeds_in_batch = seeds_in_batch

        # sib=rankings manually ranked, nothing fancy

        # import random
        sel_goodseed = random.random() <= self.goodseed_pred_prob

        # if self.dset_partition=='test':
        #     print('pausingher')
        good_seeds_in_batch = []
        # sel_single=random.random()<=self.single_augment_prob

        # include goodseed here (ie goodseed is high quality training example)
        if sel_goodseed and len(self.list_of_good_seeds) > 0 and self.dset_version != "first" and self.include_goodseed:
            goodseed = rng.choice(self.list_of_good_seeds)
            good_seeds_in_batch = [int(goodseed)]

        sel_seeds = []

        original_data_list = []

        # if keep_all_orig:
        concat_data_list = []
        for o in good_seeds_in_batch + original_seeds_in_batch:
            concat_data_list.append(self.return_single_data(o))

            sel_seeds.append(o)

        entire_new_batch = concat_data_list

        batch_len = len(entire_new_batch)

        assert batch_len <= 5, f"error batch len > 5, batch len: {batch_len}"

        padded_vals_len = 5 - batch_len  # set 7 batch len manually

        files = entire_new_batch  # first_in_batch+rest_in_batch

        extra_pad = torch.zeros_like(files[0])
        padlist = [extra_pad for k in range(padded_vals_len)]
        files += padlist
        o = self.dtype

        seeds_in_batch = torch.Tensor(sel_seeds + [-1] * padded_vals_len).to(int)

        return o, dict(files=files, batch_len=batch_len, ordered_seeds=seeds_in_batch)

    def jitter_aw98_lmks(self, lmks_aw98):
        # jitter them a bit
        a = lmks_aw98.shape[0]
        b = lmks_aw98.shape[1]

        if len(lmks_aw98.shape) == 3:
            a = lmks_aw98.shape[1]
            b = lmks_aw98.shape[2]

        r1 = 10

        offset = torch.FloatTensor(a, b).uniform_(-r1, r1).to(int).to(lmks_aw98.device)

        offset = offset.view_as(lmks_aw98)
        lmks_aw98 = (lmks_aw98 + offset).to(int)

        return lmks_aw98

    def sigma_field_256_entire(self, seed, map_on="cpu", transforms=False, pre_augmentations=None):
        fn_sigma_field = create_pt_fn(ddir=self.ddir_func(seed), ot="entire_sigma_field_256", seed=self.seed_func(seed))

        if os.path.exists(fn_sigma_field):
            # NB: the on-disk volume is NOT a clean 256^3 -- it's already cropped to
            # the slab (pads_vals), so its raw dims are e.g. ~257x282x256, not 256^3.
            sigma_field = torch.load(fn_sigma_field, map_location=torch.device(map_on)).to(torch.float32)

        else:
            assert False, "error fn sigma field file name not exist!"

        # `pre_augmentations` (per-call extra) is applied first; `self.augmentations`
        # (always-on, e.g. a normalising transform) is applied last. See the
        # class docstring for the contract.
        if pre_augmentations is not None:
            sigma_field = pre_augmentations(sigma_field)

        if self.augmentations is not None:
            sigma_field = self.augmentations(sigma_field)

        # X,Y,Z -> Z,X,Y. This ordering is the verified one: it's what the live
        # sigma_field_256 final (sfield_256) trained on. (The cropped-slab axes
        # are not a symmetric cube, so the permute is load-bearing -- do not "fix".)
        sigma_field = sigma_field.permute(2, 1, 0).contiguous()

        return sigma_field

    def sigma_field_128_entire(self, seed, map_on="cpu", transforms=False, pre_augmentations=None):
        fn_sigma_field = create_pt_fn(ddir=self.ddir_func(seed), ot="entire_sigma_field_128", seed=self.seed_func(seed))
        if os.path.exists(fn_sigma_field):
            sigma_field = torch.load(fn_sigma_field, map_location=torch.device(map_on)).to(torch.float32)
        else:
            assert False, f"sigma_field_128: file does not exist: {fn_sigma_field}"
        if pre_augmentations is not None:
            sigma_field = pre_augmentations(sigma_field)
        if self.augmentations is not None:
            sigma_field = self.augmentations(sigma_field)
        sigma_field = sigma_field.permute(2, 1, 0).contiguous()  # X,Y,Z -> Z,X,Y
        return sigma_field

    def sigma_field_64_entire(self, seed, map_on="cpu", transforms=False, pre_augmentations=None):
        fn_depth = create_pt_fn(ddir=self.ddir_func(seed), ot="entire_sigma_field_64", seed=self.seed_func(seed))
        if os.path.exists(fn_depth):
            sigma_field = torch.load(fn_depth, map_location=torch.device(map_on)).to(torch.float32)
        else:
            # 64-res volume not synthesised on disk; derive it (approximately) by
            # trilinear-downsampling the 128-res volume (scale 0.5). Derive from
            # 128, NOT 256: the 256 volume is already cropped to the slab
            # (pads_vals) so it isn't the full cube; the 128 volume is the entire
            # 128^3 cube. NB: still not identical to extracting at 64 directly
            # from EG3D.
            fn128 = create_pt_fn(ddir=self.ddir_func(seed), ot="entire_sigma_field_128", seed=self.seed_func(seed))
            vol = torch.load(fn128, map_location=torch.device(map_on)).to(torch.float32)
            sigma_field = torch.nn.functional.interpolate(vol[None, None, ...], scale_factor=0.5, mode="trilinear", align_corners=False)[0, 0]

        if pre_augmentations is not None:
            sigma_field = pre_augmentations(sigma_field)

        if self.augmentations is not None:
            sigma_field = self.augmentations(sigma_field)

        # sigma_field=sigma_field.permute(2,1,0).unsqueeze(0) #X,Y,Z -> B,Z,X,Y, B=1
        sigma_field = sigma_field.permute(2, 1, 0)  # X,Y,Z -> Z,X,Y

        return sigma_field

    def sigma_nose_512(self, seed, map_on="cpu", transforms=True, pre_augmentations=None):
        fn_depth = create_pt_fn(ddir=self.ddir_func(seed), ot="nose_512", seed=self.seed_func(seed))
        sigma_field = torch.load(fn_depth, map_location=torch.device(map_on))  # .view(52, 78, 251).permute(2,1,0)# so that we have Z,X,Y.

        if pre_augmentations is not None:
            sigma_field = pre_augmentations(sigma_field)
        if self.augmentations is not None:
            sigma_field = self.augmentations(sigma_field)

        sigma_field = sigma_field.permute(2, 1, 0)

        return sigma_field

    def modules_depthmap_to_pcd(self, seed, return_im=False, with_lim_pts=False, map_on="cpu"):
        # Returns the full canonical depth-map PCD: nrs*nrs = 128*128 = 16384 points,
        # shape (16384, 3). (Removed the never-set `rescale_in_z_dir_ratio` arg and
        # the arbitrary `cutoff=4.0` depth filter -- no caller set either; dropping
        # the cutoff also makes the full-grid `randperm(128**2)` indexing in the
        # pcd_as_pt path unconditionally valid.)
        nrs = 128

        fn_depth = create_pt_fn(ddir=ddir_func(seed), ot="triple_dmap", seed=self.seed_func(seed))

        modules_depthmap_image = torch.load(fn_depth, map_location=torch.device(map_on))[1].unsqueeze(0).squeeze(0, 1)

        ptc = dpcd.modules_depthmap_to_pcd_from_image(
            modules_depthmap_image=modules_depthmap_image,
            ray_sampler=self.ray_sampler_static,
            gen_c=None,
            nrs=nrs,
            radius_cutoff=None,
            lim_pts=self.lim_pts if with_lim_pts else None,
        )

        if return_im:
            return (ptc, modules_depthmap_image)

        return ptc

    def subset_from_nose_radius(self, pcd, radius_cutoff=1.1):
        nose_idx = torch.tensor([8127, 8128, 8255, 8256])
        points = pcd[:, [0, 1, 2]]

        nose_mean_point = points[nose_idx].mean(0)
        nose_mask = torch.norm(points - nose_mean_point, dim=1, p=2) < radius_cutoff
        return pcd[nose_mask]

    def get_ws_code_view_conditioned_first(self, seed, return_im=False):
        # Loads the saved EG3D W+ "style code" (mapping-network output for z,
        # conditioned on the canonical view direction) and takes the
        # first/canonical row. NB: this data is NOT currently synthesised into
        # eg3dredo_data -- the generator-side synthesise-and-save step still needs
        # wiring up before this (and the ws_code_view_conditioned dtype) can load.
        # (Previously took unused ddir_value/seed_value args and referenced an
        # undefined `seed`; fixed to the single-seed convention used elsewhere.)
        fn_ws = create_pt_fn(ddir=self.ddir_func(seed), ot="ws_code_view_conditioned", seed=self.seed_func(seed))
        ws = torch.load(fn_ws, map_location=torch.device("cpu"))[:, 0, :]  # row 1, size 512
        return ws

    # for a list, to return patches
    def aw98_landmark_to_pcd_index(self, seed, nrs=128, lmks_aw98=None):
        """Map each AW98 2D landmark to a FLAT index into the depth-map PCD.

        NOT a centroid or a 3D point: the returned values are flat indices into the
        `nrs*nrs` grid of `modules_depthmap_to_pcd` (the PCD is the grid flattened
        row-major). Callers index the PCD with these to fetch the 3D coord at each
        landmark, e.g. `pcd[idx_dict[i]]`. (Renamed from the misleading
        `return_centroid_points_aw98_idx`.)

        `nrs` must match the PCD grid res (128 = 128*128 points); all callers pass
        128, and the default is set to 128 so the bare call can't silently build a
        256-grid index that doesn't fit the 128-grid PCD.

        Returns: {landmark_i: flat_index_tensor}.
        """
        if lmks_aw98 is None:
            lmks_aw98 = get_lmks_mask_aw98_no_edit(seed)

        # grid of flat indices: ddo[r, c] = r*nrs + c
        ddo = torch.arange(nrs * nrs).reshape(nrs, nrs)

        # AW98 landmarks are stored at 256-res (1-indexed); -> 0-indexed 128-res
        lmks = ((lmks_aw98 - 1) / 2).int().squeeze(0)

        idx_dict = {}

        for sel_lmk in range(len(lmks)):
            # NB: unpack (h, w) but index ddo[w, h]. This swap is load-bearing for
            # the stored landmark / PCD-flattening convention -- verified against the
            # working aw98_3d_lmks path; do NOT "fix" without re-checking that path.
            lmk_h, lmk_w = lmks[sel_lmk]
            lmk_w = torch.clamp(lmk_w, 0, 127)
            lmk_h = torch.clamp(lmk_h, 0, 127)
            idx_dict[sel_lmk] = ddo[lmk_w, lmk_h]

        return idx_dict

    def get_sorted_unique_seeds(self):
        all_batch_seeds = [self.all_combined_rankings_ordered[i] for i in range(len(self))]
        all_batch_seeds = np.concatenate(all_batch_seeds)
        all_unique_seeds = np.unique(all_batch_seeds).astype(np.int32)
        all_unique_seeds = all_unique_seeds[all_unique_seeds != -1]
        all_unique_seeds = np.sort(all_unique_seeds, axis=None)
        return all_unique_seeds

    def return_all_indiv_examples(self):
        files = [self.return_single_data(s) for s in self.get_sorted_unique_seeds()]

        return files

    def return_single_example_by_seed(self, seed):
        # assume seed is a singl enumber

        data = self.return_single_data(seed)

        return data


# can be employed to chain multiple instances of single stream ordered minimal (multimodal) and will return ranked batches of each data type in the training data loader. Not really used very much but works.
class dset_smulti_stream(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, dsets):
        super().__init__()
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


class dset_contrastive_second_stage(dset_single_stream_ordered_minimal):
    "# for fine tuning on good,bad pairs"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.set_all_combined_rankings()

        # also have to make sure self.augmentations is None

        self.augmentations_good = self.augmentations.good_example
        self.augmentations_bad = self.augmentations.bad_example

        self.augmentations = None

    def set_all_combined_rankings(self):
        aco = self.all_combined_rankings_ordered  # .unique().reshape(-1,1) #should just give all seeds here...
        aco = np.unique(aco).reshape(-1, 1)
        self.all_unique_rankings = aco

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.all_unique_rankings)

    def __getitem__(self, index):
        "Generates one sample of data ordered so that we have [good_sample,bad_sample] conforming to expected order for samples [s1,s2] where s1 is preferred to s2"
        # Select sample
        entire_new_batch = []
        self.augmentations = None

        ordered_batch = self.all_unique_rankings[index]  # single seed at this point
        batch_len = 2
        seed_in_batch = [int(o) for o in ordered_batch][0]  # .astype(int)  # .unique()

        entire_new_batch.append(self.return_single_data(seed_in_batch, augmentation=self.augmentations_good))
        entire_new_batch.append(self.return_single_data(seed_in_batch, augmentation=self.augmentations_bad))
        files = entire_new_batch  # first_in_batch+rest_in_batch

        o = self.dtype

        return o, dict(files=files, batch_len=batch_len, ordered_seeds=torch.Tensor([seed_in_batch, seed_in_batch]))
