# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import dnnlib
import torch
from torch_utils import persistence

from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.volumetric_rendering.ray_sampler import RaySampler
from training.volumetric_rendering.renderer import ImportanceRenderer


@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality.
        c_dim,  # Conditioning label (C) dimensionality.
        w_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # Output resolution.
        img_channels,  # Number of output color channels.
        sr_num_fp16_res=0,
        mapping_kwargs={},  # Arguments for MappingNetwork.
        rendering_kwargs={},
        sr_kwargs={},
        **synthesis_kwargs,  # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()
        self.backbone = StyleGAN2Backbone(
            z_dim,
            c_dim,
            w_dim,
            img_resolution=256,
            img_channels=32 * 3,
            mapping_kwargs=mapping_kwargs,
            **synthesis_kwargs,
        )
        self.superresolution = dnnlib.util.construct_class_by_name(
            class_name=rendering_kwargs["superresolution_module"],
            channels=32,
            img_resolution=img_resolution,
            sr_num_fp16_res=sr_num_fp16_res,
            sr_antialias=rendering_kwargs["sr_antialias"],
            **sr_kwargs,
        )
        self.decoder = OSGDecoder(
            32,
            {
                "decoder_lr_mul": rendering_kwargs.get("decoder_lr_mul", 1),
                "decoder_output_dim": 32,
            },
        )
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs

        self._last_planes = None

    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs["c_gen_conditioning_zero"]:
            c = torch.zeros_like(c)
        return self.backbone.mapping(
            z,
            c * self.rendering_kwargs.get("c_scale", 0),
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            update_emas=update_emas,
        )

    def synthesis(
        self,
        ws,
        c,
        neural_rendering_resolution=None,
        update_emas=False,
        cache_backbone=False,
        use_cached_backbone=False,
        drop_super_res=False,  # AM 01_03_2024
        depth_only=False,  # for just depth maps...
        rgbs_thru_sigmoid=False,
        **synthesis_kwargs,
    ):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs)  # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution

        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        if depth_only:
            return {"image": -1, "image_raw": -1, "image_depth": depth_image}

        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = torch.ones((rgb_image.shape[0], 1)).float().to(rgb_image.device)
        if not drop_super_res:  # ie by default
            sr_image = self.superresolution(
                rgb_image,
                feature_image,
                ws,
                noise_mode=self.rendering_kwargs["superresolution_noise_mode"],
                **{k: synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != "noise_mode"},
            )

        return {"image": sr_image, "image_raw": rgb_image, "image_depth": depth_image}

    def synthesis_with_custom_rays(
        self,
        ws,
        c,
        ray_origins,
        ray_directions,
        neural_rendering_resolution=None,
        update_emas=False,
        cache_backbone=False,
        use_cached_backbone=False,
        drop_super_res=False,  # AM 01_03_2024
        depth_only=False,  # for just depth maps...
        rgbs_thru_sigmoid=False,
        **synthesis_kwargs,
    ):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            old_nrs=self.neural_rendering_resolution
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        # ray_origins, ray_directions = self.ray_sampler(
        #    cam2world_matrix, intrinsics, neural_rendering_resolution
        # )

        # ray_directions=torch.zeros_like(ray_directions)

        # ray_directions[...,-1]=-1

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs)  # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution

        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        if depth_only:
            self.neural_rendering_resolution=old_nrs
            return {"image": -1, "image_raw": -1, "image_depth": depth_image}

        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = torch.ones((rgb_image.shape[0], 1)).float().to(rgb_image.device)
        if not drop_super_res:  # ie by default
            sr_image = self.superresolution(
                rgb_image,
                feature_image,
                ws,
                noise_mode=self.rendering_kwargs["superresolution_noise_mode"],
                **{k: synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != "noise_mode"},
            )
        self.neural_rendering_resolution=old_nrs
        return {"image": sr_image, "image_raw": rgb_image, "image_depth": depth_image}

    def synthesis_with_perpendicular_rays(
        self,
        ws,
        c,
        neural_rendering_resolution=None,
        update_emas=False,
        cache_backbone=False,
        use_cached_backbone=False,
        drop_super_res=False,  # AM 01_03_2024
        depth_only=False,  # for just depth maps...
        rgbs_thru_sigmoid=False,
        **synthesis_kwargs,
    ):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            old_nrs=self.neural_rendering_resolution
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        ray_directions = torch.zeros_like(ray_directions)

        ray_directions[..., -1] = 1

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs)  # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution

        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        if depth_only:
            self.neural_rendering_resolution=old_nrs
            
            return {"image": -1, "image_raw": -1, "image_depth": depth_image}

        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = torch.ones((rgb_image.shape[0], 1)).float().to(rgb_image.device)
        if not drop_super_res:  # ie by default
            sr_image = self.superresolution(
                rgb_image,
                feature_image,
                ws,
                noise_mode=self.rendering_kwargs["superresolution_noise_mode"],
                **{k: synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != "noise_mode"},
            )

            
        self.neural_rendering_resolution=old_nrs

        return {"image": sr_image, "image_raw": rgb_image, "image_depth": depth_image}

    def sample(
        self,
        coordinates,
        directions,
        z,
        c,
        truncation_psi=1,
        truncation_cutoff=None,
        update_emas=False,
        rgbs_thru_sigmoid=True,
        **synthesis_kwargs,
    ):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes.
        ws = self.mapping(
            z,
            c,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            update_emas=update_emas,
        )
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs, rgbs_thru_sigmoid=rgbs_thru_sigmoid)

    def sample_mixed(
        self,
        coordinates,
        directions,
        ws,
        truncation_psi=1,
        truncation_cutoff=None,
        update_emas=False,
        rgbs_thru_sigmoid=True,
        **synthesis_kwargs,
    ):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs, rgbs_thru_sigmoid=rgbs_thru_sigmoid)

    def forward(
        self,
        z,
        c,
        truncation_psi=1,
        truncation_cutoff=None,
        neural_rendering_resolution=None,
        update_emas=False,
        cache_backbone=False,
        use_cached_backbone=False,
        **synthesis_kwargs,
    ):
        # Render a batch of generated images.
        ws = self.mapping(
            z,
            c,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            update_emas=update_emas,
        )
        return self.synthesis(
            ws,
            c,
            update_emas=update_emas,
            neural_rendering_resolution=neural_rendering_resolution,
            cache_backbone=cache_backbone,
            use_cached_backbone=use_cached_backbone,
            **synthesis_kwargs,
        )


from training.networks_stylegan2 import FullyConnectedLayer


class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options["decoder_lr_mul"]),
            torch.nn.Softplus(),
            FullyConnectedLayer(
                self.hidden_dim,
                1 + options["decoder_output_dim"],
                lr_multiplier=options["decoder_lr_mul"],
            ),
        )

    def forward(self, sampled_features, ray_directions=None, sigmoid_final=True):  # nb sigmoid_final added cos want to inspect raw outputs before put into the sigmoid function. could b very high / low.
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N * M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        if not sigmoid_final:
            rgb = x[..., 1:]

        else:
            rgb = torch.sigmoid(x[..., 1:]) * (1 + 2 * 0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {"rgb": rgb, "sigma": sigma}
