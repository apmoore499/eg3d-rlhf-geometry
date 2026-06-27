"""Small live helpers for EG3D mesh preview and snapshot image stacking."""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import PIL
import torch


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
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
    return samples.unsqueeze(0), voxel_origin, voxel_size


def imd_to_xyz(image_depth, ray_origins, ray_directions, neural_rendering_resolution):
    final_dim = neural_rendering_resolution * neural_rendering_resolution
    imd = image_depth.unsqueeze(2).expand(1, final_dim, 3)
    return ray_origins + imd * ray_directions


def stack_snapshot_images_fn(image_files):
    all_images = [PIL.Image.open(f) for f in image_files]

    max_width = max(img.width for img in all_images)
    total_height = sum(img.height for img in all_images)

    stacked_image = PIL.Image.new("RGB", (max_width, total_height))

    current_y = 0
    for img in all_images:
        stacked_image.paste(img, (0, current_y))
        current_y += img.height

    return stacked_image
