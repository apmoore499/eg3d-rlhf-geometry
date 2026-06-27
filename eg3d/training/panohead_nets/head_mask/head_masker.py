#!/usr/bin/python
# -*- encoding: utf-8 -*-
#
# HeadMasker: on-the-fly head-silhouette masks for the PanoHead
# discriminator's segmentation channel.
#
# Uses the vendored BiSeNet (zllrunning/face-parsing.PyTorch, MIT) trained
# on CelebAMask-HQ. Given a real-image batch (EG3D reals are float in
# [-1, 1]), it produces a binary HEAD silhouette (skin + hair + neck +
# accessories) as a [B, 1, H, W] float tensor in [0, 1] on the same
# device/dtype as the input image. Inference only (no grad).
#
# CelebAMask-HQ / BiSeNet 19-class index map:
#   0 background  1 skin     2 l_brow   3 r_brow  4 l_eye   5 r_eye
#   6 eye_g       7 l_ear    8 r_ear    9 ear_r  10 nose   11 mouth
#  12 u_lip      13 l_lip   14 neck    15 neck_l 16 cloth  17 hair
#  18 hat
#
# Head silhouette = whole head incl. hair, EXCLUDING background(0),
# necklace(15) and clothing(16).

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import BiSeNet

# Whole-head foreground classes (face skin + features + ears + neck +
# glasses + hair + hat). Excludes 0 (bg), 15 (necklace), 16 (cloth).
HEAD_CLASSES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18)

# Default vendored checkpoint path (next to this file).
_DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "79999_iter.pth"
)

# ImageNet normalization expected by the BiSeNet face-parsing model.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# BiSeNet was trained at 512x512 input.
_BISENET_RES = 512


class HeadMasker:
    """Callable that maps a real RGB batch -> binary head silhouette mask.

    Args:
        device: torch device to run BiSeNet on.
        weights_path: path to the 79999_iter.pth checkpoint.
        n_classes: number of segmentation classes (19 for CelebAMask-HQ).
        head_classes: iterable of class indices treated as foreground.
        out_resolution: spatial resolution of the returned mask
            (default 512; the D resizes image_mask internally so 512 is
            fine regardless of image resolution).
    """

    def __init__(
        self,
        device,
        weights_path=_DEFAULT_WEIGHTS,
        n_classes=19,
        head_classes=HEAD_CLASSES,
        out_resolution=512,
    ):
        self.device = torch.device(device)
        self.out_resolution = int(out_resolution)
        self.head_classes = torch.tensor(
            list(head_classes), device=self.device, dtype=torch.long
        )

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"BiSeNet weights not found at {weights_path}"
            )

        net = BiSeNet(n_classes=n_classes)
        sd = torch.load(weights_path, map_location="cpu")
        net.load_state_dict(sd, strict=True)
        net = net.to(self.device).eval().requires_grad_(False)
        self.net = net

        mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self.register_mean = mean
        self.register_std = std

    @torch.no_grad()
    def __call__(self, img_bchw):
        """Compute head-silhouette masks for a real image batch.

        Args:
            img_bchw: [B, 3, H, W] float tensor of real images in [-1, 1]
                (the exact tensor fed to the discriminator).

        Returns:
            [B, 1, out_res, out_res] float tensor in {0., 1.} on the same
            device/dtype as img_bchw. Detached (constant D input).
        """
        in_dtype = img_bchw.dtype
        x = img_bchw.detach().to(self.device, dtype=torch.float32)

        # [-1, 1] -> [0, 1]
        x = (x + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)

        # Resize to BiSeNet's expected input resolution.
        if x.shape[-1] != _BISENET_RES or x.shape[-2] != _BISENET_RES:
            x = F.interpolate(
                x,
                size=(_BISENET_RES, _BISENET_RES),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        # ImageNet normalization.
        x = (x - self.register_mean) / self.register_std

        logits = self.net(x)[0]  # [B, 19, 512, 512]
        parsing = logits.argmax(dim=1)  # [B, 512, 512]

        # Binary foreground mask: pixel class in head_classes.
        mask = torch.isin(parsing, self.head_classes)  # bool [B, 512, 512]
        mask = mask.unsqueeze(1).to(torch.float32)  # [B, 1, 512, 512]

        if mask.shape[-1] != self.out_resolution:
            mask = F.interpolate(
                mask,
                size=(self.out_resolution, self.out_resolution),
                mode="nearest",
            )

        return mask.to(device=img_bchw.device, dtype=in_dtype)


if __name__ == "__main__":
    # Tiny self-test on a random tensor.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[HeadMasker self-test] device={dev}")
    masker = HeadMasker(device=dev)
    b = torch.rand(2, 3, 512, 512, device=dev) * 2 - 1  # [-1, 1]
    out = masker(b)
    print("input :", tuple(b.shape), b.dtype, "range", float(b.min()),
          float(b.max()))
    print("output:", tuple(out.shape), out.dtype, "device", out.device)
    print("output range:", float(out.min()), "-", float(out.max()))
    uniq = torch.unique(out)
    print("unique values:", uniq.detach().cpu().tolist())
    # Also test a non-512 input to exercise resize paths.
    b2 = torch.rand(1, 3, 256, 256, device=dev) * 2 - 1
    out2 = masker(b2)
    print("non-512 input -> output:", tuple(out2.shape))
    print("[HeadMasker self-test] OK")
