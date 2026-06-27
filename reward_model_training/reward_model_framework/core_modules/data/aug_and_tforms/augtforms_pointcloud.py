"""Point-cloud transforms used by the data loaders and the contrastive path.

`ensemble_pointcloud_transforms` is the shared transform pipeline the dataset
loaders call directly (via `self.ept`); `random_pointcloud_transforms` adds a
re-randomising wrapper used by the contrastive RLHF data path.
"""

import math
import random

import torch
import torch.nn as nn

# ---------------------- Point-cloud utilities ---------------------- #


class ensemble_pointcloud_transforms(nn.Module):
    """Shared point-cloud transform pipeline used across loaders and tooling."""

    def __init__(
        self,
        n_points: int = 4096,
        translation_dist: float = 0.2,
        random_scale_margins: float = 0.05,
        degrees_pitch_range: float = 15,
        degrees_yaw_range: float = 60,
        jitter_range: float = 0.001,
    ):
        super().__init__()

        self.translation_dist = translation_dist
        self.random_scale_margins = random_scale_margins
        self.degrees_pitch_range = degrees_pitch_range
        self.degrees_yaw_range = degrees_yaw_range
        self.jitter_range = jitter_range
        self.n_points = n_points

        self.resample_transform_parameters()

    def reset_random_domains_for_train(self):
        self.translation_dist = 0.2
        self.random_scale_margins = 0.05
        self.degrees_pitch_range = 15
        self.degrees_yaw_range = 60
        self.jitter_range = 0.001

        self.resample_transform_parameters()
        return self

    def set_no_random_for_validation(self):
        self.translation_dist = 0.0
        self.random_scale_margins = 0.00
        self.degrees_pitch_range = 0
        self.degrees_yaw_range = 0
        self.jitter_range = 0.0

        self.resample_transform_parameters()
        return self

    def apply_transforms_no_downsample_no_rescale_no_rotation(self, in_pcd: torch.Tensor) -> torch.Tensor:
        # Preserve incoming points and apply only translation jitter.
        return self.random_translate_points(in_pcd)

    def apply_downsample_translate_only(self, in_pcd: torch.Tensor) -> torch.Tensor:
        # Downsample then random-translate only -- no mean-scale (despite the old
        # name `apply_transforms_downsample_mean_only`, which was a misnomer).
        out_pcd = self.downsample_pcd_points(in_pcd)
        return self.random_translate_points(out_pcd)

    def apply_all_transforms_fixed(self, in_pcd: torch.Tensor) -> torch.Tensor:
        out_pcd = self.downsample_pcd_points(in_pcd)
        out_pcd = self.center_points(out_pcd)
        out_pcd = self.rotate_points_3d_random(out_pcd)
        out_pcd = self.jitter_points_uniform(out_pcd)
        out_pcd = self.center_points(out_pcd)
        out_pcd = self.random_scale_points_along_axes(out_pcd)
        out_pcd = self.mean_scale_pts(out_pcd)
        return self.random_translate_points(out_pcd)

    def apply_downsample_center_mean_only(self, in_pcd: torch.Tensor) -> torch.Tensor:
        out_pcd = self.downsample_pcd_points(in_pcd)
        out_pcd = self.center_points(out_pcd)
        return self.mean_scale_pts(out_pcd)

    def resample_transform_parameters(self):
        self.translation_offset = torch.empty(1, 3).uniform_(-self.translation_dist, self.translation_dist)
        self.scale_rndm = torch.empty(1, 3).uniform_(1 - self.random_scale_margins, 1 + self.random_scale_margins)

        jitter_components = [torch.empty((self.n_points,)).uniform_(-abs(self.jitter_range), abs(self.jitter_range)) for _ in range(3)]
        self.jitter_offset = torch.stack(jitter_components, dim=-1)

        degrees_pitch = (-abs(self.degrees_pitch_range), abs(self.degrees_pitch_range))
        self.degrees_pitch = math.pi * random.uniform(*degrees_pitch) / 180.0

        degrees_yaw = (-abs(self.degrees_yaw_range), abs(self.degrees_yaw_range))
        self.degrees_yaw = math.pi * random.uniform(*degrees_yaw) / 180.0
        return self

    def mean_scale_pts(self, ttl: torch.Tensor) -> torch.Tensor:
        ttl_c = ttl - ttl.mean(dim=0, keepdim=True)
        scale = (1 / ttl_c.abs().max()) * 0.999999
        return ttl_c * scale

    def center_points(self, ttl: torch.Tensor) -> torch.Tensor:
        return ttl - ttl.mean(dim=0, keepdim=True)

    def downsample_pcd_points(self, ttl: torch.Tensor) -> torch.Tensor:
        perm = torch.randperm(ttl.size(0))
        idx = perm[: min(ttl.size(0), self.n_points)]
        return ttl[idx]

    def random_translate_points(self, ttl: torch.Tensor) -> torch.Tensor:
        translation = self.translation_offset.expand(ttl.shape)
        return ttl + translation

    def random_scale_points_along_axes(self, ttl: torch.Tensor) -> torch.Tensor:
        translation = self.scale_rndm.expand(ttl.shape)
        return ttl * translation

    def jitter_points_uniform(self, ttl: torch.Tensor) -> torch.Tensor:
        return ttl + self.jitter_offset.view_as(ttl)

    def rotate_points_3d_random(self, points: torch.Tensor) -> torch.Tensor:
        pos = points
        device = torch.device("cpu") if pos.get_device() == -1 else torch.device(f"cuda:{pos.get_device()}")
        orig_shape = pos.shape
        assert len(pos.shape) == 2, "error u need the 2 dim positions for rotation thing"

        if pos.shape[-1] != 3:
            pos = pos.permute(1, 0)

        sin, cos = math.sin(self.degrees_pitch), math.cos(self.degrees_pitch)
        matrix = torch.tensor([[1, 0, 0], [0, cos, sin], [0, -sin, cos]], device=device)
        pos = pos @ matrix

        sin, cos = math.sin(self.degrees_yaw), math.cos(self.degrees_yaw)
        matrix = torch.tensor([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]], device=device)
        pos = pos @ matrix

        return pos.view(orig_shape)


class random_pointcloud_transforms(ensemble_pointcloud_transforms):
    def apply_all_transforms_randomly(self, in_pcd: torch.Tensor) -> torch.Tensor:
        self.resample_transform_parameters()
        return self.apply_all_transforms_fixed(in_pcd)


# ---------------- Composable per-call transforms (config slots) ---------------- #
# Small nn.Module transforms composed via `transforms_composition_helper` and
# selected per split (train/val/test/tune) from a `data/augmentations` config.
# Unlike `ensemble_pointcloud_transforms` (which pre-samples its random params),
# each RE-SAMPLES its randomness on every forward -> a fresh augmentation per
# batch. All operate on a single cloud shaped (N, 3).


class pcd_subsample(nn.Module):
    """Subsample a cloud to `n_points`. `random=True` (train) draws a random
    subset; `random=False` (val/test/tune) takes a deterministic even-stride
    subset so eval is reproducible. No-op when N <= n_points."""

    def __init__(self, n_points: int = 2048, random: bool = True):
        super().__init__()
        self.n_points = n_points
        self.random = random

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_in = x.size(0)
        n = min(n_in, self.n_points)
        if self.random:
            idx = torch.randperm(n_in, device=x.device)[:n]
        else:
            idx = torch.linspace(0, n_in - 1, n, device=x.device).round().long()
        return x[idx]


class pcd_center(nn.Module):
    """Translate the cloud to zero centroid."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=0, keepdim=True)


class pcd_mean_scale(nn.Module):
    """Center then scale into the unit cube (max |coord| -> ~1). Matches the
    legacy `ensemble_pointcloud_transforms.mean_scale_pts`, so composing
    center -> mean_scale reproduces the old `point_cloud_entire_center_mean`
    normalisation exactly."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x - x.mean(dim=0, keepdim=True)
        scale = (1.0 / xc.abs().max()) * 0.999999
        return xc * scale


class pcd_random_scale(nn.Module):
    """Train-time augmentation: multiply by a per-axis random scale drawn fresh
    each call from U(1 - margin, 1 + margin). Applied AFTER normalisation so the
    scale jitter is not cancelled by a later unit-scale."""

    def __init__(self, margin: float = 0.05):
        super().__init__()
        self.margin = margin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.empty(1, 3, device=x.device).uniform_(1 - self.margin, 1 + self.margin)
        return x * scale


class pcd_jitter(nn.Module):
    """Train-time augmentation: add per-point uniform noise in [-r, r], fresh
    each call."""

    def __init__(self, jitter_range: float = 0.001):
        super().__init__()
        self.jitter_range = jitter_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = abs(self.jitter_range)
        return x + torch.empty_like(x).uniform_(-r, r)


__all__ = [
    "ensemble_pointcloud_transforms",
    "random_pointcloud_transforms",
    "pcd_subsample",
    "pcd_center",
    "pcd_mean_scale",
    "pcd_random_scale",
    "pcd_jitter",
]
