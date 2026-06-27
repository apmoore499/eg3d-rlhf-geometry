import os

import cv2
import imageio.v3 as iio
import numpy as np
import torch

from . import io_geometry_utils as io_utils


def create_pt_fn(ddir, ot, seed):
    return io_utils.create_pt_fn(ddir, ot, seed)


def assemble_triple_lmks(seed, ddir):
    return io_utils.assemble_triple_lmks(seed, ddir)


def assemble_single_lmks(seed, ddir):
    return io_utils.assemble_single_lmks(seed, ddir)


def assemble_triple_dmap(seed, ddir):
    return io_utils.assemble_triple_dmap(seed, ddir)


def assemble_single_dmap(seed, ddir):
    return io_utils.assemble_single_dmap(seed, ddir)


def assemble_mediapipe_468_lmks(seed, ddir, dmap_res=256):
    return io_utils.assemble_mediapipe_468_lmks(seed, ddir, dmap_res)


def assemble_single_rgb(seed, ddir):
    return io_utils.assemble_single_rgb(seed, ddir)


def assemble_triple_rgb(seed, ddir):
    return io_utils.assemble_triple_rgb(seed, ddir)


def get_canonical_dmap_cams():
    return io_utils.get_canonical_dmap_cams()


def load_vertex_sampling_weights_dmap_128():
    return io_utils.load_vertex_sampling_weights_dmap_128()


def imd_to_xyz_with_radius_cutoff(image_depth, ray_origins, ray_directions, neural_rendering_resolution, radius_cutoff=None):
    return io_utils.imd_to_xyz_with_radius_cutoff(image_depth, ray_origins, ray_directions, neural_rendering_resolution, radius_cutoff)


def rescale_im_dmp_for_lmk(dmap):
    return io_utils.rescale_im_dmp_for_lmk(dmap)


def seed_func_default(s):
    return io_utils.seed_func_default(s)


def ddir_func(query_val):
    return io_utils.ddir_func(query_val)


def return_lmks_mask_mediapipe_468(s, radius=9, return_im=False):
    return io_utils.return_lmks_mask_mediapipe_468(s, radius=radius, return_im=return_im)


def get_lmks_mask_aw98_no_edit(s):  #:L, radius=9, return_im=False,randomize_sel=False):
    return io_utils.return_lmks_mask_aw98_no_edit(s)


def return_lmks_mask_aw98(s, radius=9, return_im=False, randomize_sel=False):
    return io_utils.return_lmks_mask_aw98(s, radius=radius, return_im=return_im, randomize_sel=randomize_sel)


# just does eyes, nose, mouth found from 98 lmks here:
# https://camo.githubusercontent.com/6e2f31c0c5d81660e7249dc31a2f570bb41685c21bea8ba4e1f1572ea692ed18/68747470733a2f2f777977752e6769746875622e696f2f70726f6a656374732f4c41422f737570706f72742f57464c575f616e6e6f746174696f6e2e706e67
def return_only_five_lmks_mask(s, radius=9, return_im=False):
    return io_utils.return_only_five_lmks_mask(s, radius=radius, return_im=return_im)
