"""Depth-map augmentations and transforms (standalone definitions).

This module mirrors the point-cloud aug file: keep depth-map transforms
small, composable, and importable directly from configs/models without
pulling in heavy data-loader logic.
"""

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import omegaconf


def _ensure_4d(dmap: torch.Tensor) -> torch.Tensor:
    """Coerce a depth map to [B,1,H,W].

    Accepts [B,H,W], [B,1,H,W], or the canonical reward-model single-dmap format
    [B,1,1,H,W] (the loader emits (1,1,1,128,128)); the singleton view dim is
    squeezed away.
    """
    if dmap.ndim == 5 and dmap.shape[1] == 1:
        dmap = dmap.squeeze(1)  # canonical [B,1,1,H,W] -> [B,1,H,W]
    if dmap.ndim == 3:
        return dmap.unsqueeze(1)
    if dmap.ndim != 4:
        raise ValueError(f"Expected depth map of shape [B,1,H,W] or [B,H,W]; got {tuple(dmap.shape)}")
    return dmap


def _normalize_unit(dmap: torch.Tensor) -> torch.Tensor:
    dmin = dmap.amin(dim=(-2, -1), keepdim=True)
    dmax = dmap.amax(dim=(-2, -1), keepdim=True)
    denom = (dmax - dmin).clamp(min=1e-6)
    return (dmap - dmin) / denom


def _gaussian_mask(h: int, w: int, sigma: float, device, dtype) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    cy, cx = h // 2, w // 2
    mask = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    return mask / mask.max()


def high_pass_image_filter(im_tensor: torch.Tensor, low_freq_size: int = 50) -> torch.Tensor:
    """High-pass filter via FFT with a center block removed."""
    fft_image = torch.fft.fft2(im_tensor)
    fft_image_shifted = torch.fft.fftshift(fft_image)

    rows, cols = im_tensor.shape[-2:]
    crow, ccol = rows // 2, cols // 2
    mask = torch.ones((rows, cols), dtype=torch.bool, device=im_tensor.device)
    mask[crow - low_freq_size : crow + low_freq_size, ccol - low_freq_size : ccol + low_freq_size] = False

    fft_image_shifted_high_pass = fft_image_shifted * mask
    fft_image_high_pass = torch.fft.ifftshift(fft_image_shifted_high_pass)
    image_high_pass = torch.fft.ifft2(fft_image_high_pass).real
    return image_high_pass


def low_pass_image_filter(im_tensor: torch.Tensor, low_freq_size: int = 50) -> torch.Tensor:
    """Low-pass filter via FFT keeping only a center block."""
    fft_image = torch.fft.fft2(im_tensor)
    fft_image_shifted = torch.fft.fftshift(fft_image)

    rows, cols = im_tensor.shape[-2:]
    crow, ccol = rows // 2, cols // 2
    mask = torch.zeros((rows, cols), dtype=torch.bool, device=im_tensor.device)
    mask[crow - low_freq_size : crow + low_freq_size, ccol - low_freq_size : ccol + low_freq_size] = True

    fft_image_shifted_low_pass = fft_image_shifted * mask
    fft_image_low_pass = torch.fft.ifftshift(fft_image_shifted_low_pass)
    image_low_pass = torch.fft.ifft2(fft_image_low_pass).real
    return image_low_pass


class DepthMapPreprocessor(nn.Module):
    """Resize, normalise, optional hi/lo-pass and cropping for depth maps."""

    def __init__(
        self,
        out_size: int = 224,
        normalize_range: bool = True,
        depth_min: float = 2.25,
        depth_max: float = 3.3,
        invert: bool = True,
        hipass: bool = False,
        laplace: bool = False,
        normalise_sides_crop: bool = False,
        run_lowpass_sides: bool = False,
        hp_scale: float = 1.0,
        lp_scale: float = 1.0,
        highpass_low_freq: int = 50,
        lowpass_low_freq: int = 100,
        mask_sigma_range: Tuple[float, float] = (1.0, 5.0),
        sides_lower_cutoff: float = 0.65,
    ):
        super().__init__()
        self.out_size = out_size
        self.normalize_range = normalize_range
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.invert = invert
        self.hipass = hipass
        self.laplace = laplace
        self.normalise_sides_crop = normalise_sides_crop
        self.run_lowpass_sides = run_lowpass_sides
        self.hp_scale = hp_scale
        self.lp_scale = lp_scale
        self.highpass_low_freq = highpass_low_freq
        self.lowpass_low_freq = lowpass_low_freq
        self.mask_sigma_range = mask_sigma_range
        self.sides_lower_cutoff = sides_lower_cutoff

        kernel = torch.tensor([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
        self.register_buffer("laplace_kernel", kernel.view(1, 1, 3, 3), persistent=False)

    def forward(
        self,
        dmap: torch.Tensor,
        hipass: bool = None,
        laplace: bool = None,
        normalise_sides_crop: bool = None,
        run_lowpass_sides: bool = None,
        Llp: float = None,
        Lhp: float = None,
    ) -> torch.Tensor:
        """Apply pre-processing; bools/weights can override constructor defaults."""
        dmap = _ensure_4d(dmap)
        dmap = F.interpolate(dmap, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)

        if self.normalize_range:
            dmap = (dmap - self.depth_min) / (self.depth_max - self.depth_min)
        if self.invert:
            dmap = -1 * (dmap - 0.5) + 0.5

        use_hipass = self.hipass if hipass is None else hipass
        use_laplace = self.laplace if laplace is None else laplace
        use_sides_crop = self.normalise_sides_crop if normalise_sides_crop is None else normalise_sides_crop
        use_lowpass_sides = self.run_lowpass_sides if run_lowpass_sides is None else run_lowpass_sides
        lp_scale = self.lp_scale if Llp is None else Llp
        hp_scale = self.hp_scale if Lhp is None else Lhp

        if use_hipass:
            mask = self._gaussian_mask_like(dmap)
            hi = high_pass_image_filter(dmap, low_freq_size=self.highpass_low_freq)
            dmap = _normalize_unit(dmap + hp_scale * mask * hi)

        if use_laplace:
            mask = self._gaussian_mask_like(dmap)
            lap = F.conv2d(dmap, self.laplace_kernel.to(dmap.dtype), padding=1)
            dmap = _normalize_unit(dmap + lp_scale * mask * lap)

        if use_sides_crop:
            dmap = _normalize_unit(self._sides_crop_filter(dmap))

        if use_lowpass_sides:
            lp = low_pass_image_filter(dmap, low_freq_size=self.lowpass_low_freq)
            dmap = _normalize_unit(lp * dmap.pow(5))

        return dmap

    def _gaussian_mask_like(self, dmap: torch.Tensor) -> torch.Tensor:
        h, w = dmap.shape[-2:]
        sigma = torch.empty(1).uniform_(self.mask_sigma_range[0], self.mask_sigma_range[1]).item()
        mask = _gaussian_mask(h, w, sigma=sigma, device=dmap.device, dtype=dmap.dtype)
        return mask.expand_as(dmap)

    def _create_inward_fade_mask(self, img: torch.Tensor, fade_width: int = None) -> torch.Tensor:
        _, _, height, width = img.shape
        if fade_width is None:
            fade_width = max(1, int(height / 20))
        y = torch.linspace(0.3, 1.0, steps=fade_width, device=img.device, dtype=img.dtype)
        fade_y = torch.cat([y, torch.ones(height - 2 * fade_width, device=img.device, dtype=img.dtype), y.flip(0)])
        fade_x = torch.cat([y, torch.ones(width - 2 * fade_width, device=img.device, dtype=img.dtype), y.flip(0)])
        fade_mask = fade_y[:, None] * fade_x[None, :]
        batch = img.shape[0]
        c = img.shape[1]
        return fade_mask.expand(batch, c, height, width)

    def _sides_crop_filter(self, dmap: torch.Tensor) -> torch.Tensor:
        filtered = dmap.clone()
        filtered[filtered <= self.sides_lower_cutoff] = 0.0
        filtered = filtered.clamp(min=0.0)
        filtered = _normalize_unit(filtered)
        fade_mask = self._create_inward_fade_mask(filtered)
        return filtered * fade_mask


class DepthMapTo224Transform(DepthMapPreprocessor):
    """Backward-compatible alias for 224x224 depth-map preprocessing."""

    def __init__(self, **kwargs):
        super().__init__(out_size=224, **kwargs)


class DepthMapTo160Transform(DepthMapPreprocessor):
    """Backward-compatible alias for 160x160 depth-map preprocessing."""

    def __init__(self, **kwargs):
        super().__init__(out_size=160, **kwargs)


def build_depthmap_preprocessors(
    maps_transforms: omegaconf.DictConfig,
    out_size: int = 224,
    **kwargs,
) -> Iterable[DepthMapPreprocessor]:
    """Utility to instantiate per-view preprocessors from a Hydra maps_transforms config."""
    hipass_flags = maps_transforms.get("hipass", [])
    laplace_flags = maps_transforms.get("laplace", [])
    normalise_sides_crop_flags = maps_transforms.get("normalise_sides_crop", [])
    run_lowpass_sides_flags = maps_transforms.get("run_lowpass_sides", [])

    num = max(len(hipass_flags), len(laplace_flags), len(normalise_sides_crop_flags), len(run_lowpass_sides_flags))
    preprocessors = []
    for idx in range(num):
        hip = hipass_flags[idx if idx < len(hipass_flags) else -1]
        lap = laplace_flags[idx if idx < len(laplace_flags) else -1]
        crop = normalise_sides_crop_flags[idx if idx < len(normalise_sides_crop_flags) else -1]
        lowpass = run_lowpass_sides_flags[idx if idx < len(run_lowpass_sides_flags) else -1]
        preprocessors.append(
            DepthMapPreprocessor(
                out_size=out_size,
                hipass=hip,
                laplace=lap,
                normalise_sides_crop=crop,
                run_lowpass_sides=lowpass,
                **kwargs,
            )
        )
    return preprocessors


__all__ = [
    "DepthMapPreprocessor",
    "DepthMapTo160Transform",
    "DepthMapTo224Transform",
    "build_depthmap_preprocessors",
    "high_pass_image_filter",
    "low_pass_image_filter",
]
