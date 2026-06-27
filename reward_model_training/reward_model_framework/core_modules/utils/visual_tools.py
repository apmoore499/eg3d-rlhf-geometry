# get the permutation
# Open an Image
import autoroot  # noqa: F401

import copy
import glob
import itertools
import os
import sys

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch_geometric
import torch_geometric.transforms as T
import torchvision.transforms.v2 as v2
import tqdm
import trimesh

import cv2
from camera_utils import FOV_to_intrinsics, LookAtPoseSampler
from pandas_ods_reader import read_ods
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pyexcel import get_book
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from training.triplane import TriPlaneGenerator
from training.volumetric_rendering.ray_sampler import RaySampler


# for loaded depth map, returns images with landmarks and the number of vertices we can keep if we increase radius of landmarks
# for deciding how we could prune point cloud
def visualise_modules_depthmap_with_lmks(dmp):
    # nimg = np.array(pim)
    ocvim = cv2.cvtColor(dmp, cv2.COLOR_RGB2BGR)

    # import cv2
    pil_images = []
    resulting = []
    for radius in range(30):
        ocvim = cv2.cvtColor(dmp, cv2.COLOR_RGB2BGR)

        for l in lmks:
            x, y = l
            ocvim = cv2.circle(ocvim, (x, y), radius=radius, color=(0, 0, 255), thickness=-1)

        color_converted = cv2.cvtColor(ocvim, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_converted)

        npa = np.array(pil_image)

        npoints = npa[npa == [0, 0, 255]].shape[0]

        resulting.append([radius, npoints])

        pil_images.append(npa)

    df = pd.DataFrame(resulting)
    df.columns = ["radius", "npoints"]

    return dict(results=df, images=pil_images)
