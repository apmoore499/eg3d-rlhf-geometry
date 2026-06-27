import autoroot  # noqa: F401

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
import os
import sys
# Open an Image
# get the permutation
# Open an Image
import copy
import glob
import itertools
import math
import numbers
import os
import random
import sys
from pathlib import Path

import cv2
import imageio.v3 as iio
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import PIL
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.transforms as T
import torch_geometric.transforms as geom_T
import torchvision
import torchvision.transforms.v2 as v2
import tqdm
import trimesh

from torch import nn
from core_modules.data.aug_and_tforms.augtforms_pointcloud import ensemble_pointcloud_transforms, random_pointcloud_transforms


# Point-cloud transforms now imported from core_modules.data.aug_and_tforms.augtforms_pointcloud

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
REWARD_MODEL_TRAINING_DIR = PROJECT_ROOT / "reward_model_training"
DATASET_CACHE_DIR = REWARD_MODEL_TRAINING_DIR / "datasets"

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


# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/random_jitter.html


# --------------------------------------------

# Contrastive Loss, Point Cloud

# _-------------------------------------------


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

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.all_seeds_in_batch)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        seed = self.all_seeds_in_batch[index]
        fn = create_pt_fn(ddir=self.ddir_func(seed), ot="pcd_as_pt", seed=seed)  # for s in seeds_in_batch]
        pcd = torch.load(fn, map_location=torch.device("cpu"))
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


def create_pcd_dsets_for_contrastive():
    selected_dtypes = ["pcd_as_pt"][0]

    e_conds = expnames_spec[(~(expnames_spec.seed_start).isna()) & (~(expnames_spec.seed_end).isna())]

    ecc = e_conds[(e_conds.seed_start > 19999) & (e_conds.seed_end < 101000)]
    starts = ecc.seed_start.values
    ends = ecc.seed_end.values

    los = [[r for r in range(int(s), int(e))] for s, e in zip(starts, ends)]
    los = los[0] + los[1]
    los = np.array(los).flatten()
    los = np.unique(los)
    los_train, los_valt = train_test_split(los, test_size=0.3)
    train_dataset = dset_pcd_for_closs(los_train, dtype=selected_dtypes, ddir_func=ddir_func)
    val_dataset = dset_pcd_for_closs(los_valt, dtype=selected_dtypes, ddir_func=ddir_func)

    names_of_pt = dict(
        train=str(DATASET_CACHE_DIR / "train_dset_closs_03.pt"),
        val=str(DATASET_CACHE_DIR / "valtest_dset_closs_03.pt"),
    )
    # test='/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/notebooks/legacy/test_dset_03_10_2023_10112_pcd_g_p1.pt')

    torch.save(obj=train_dataset, f=names_of_pt["train"])
    torch.save(obj=val_dataset, f=names_of_pt["val"])
