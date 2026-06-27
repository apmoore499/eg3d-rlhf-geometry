"""
Conv3D reward model family.

Implementations were extracted from `all_models.py` to keep configs pointed at a
focused module while reducing the size of the monolith.
"""

import os
from typing import Optional, Sequence

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F

from core_modules.models.base import UniversalRWDModel


class Conv3DNetworkEnsemble(UniversalRWDModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.save_hyperparameters(logger=False)

        self.return_global_embedding = kwargs["return_global_embedding"]
        self.external = kwargs["external"]

        self.Conv3DModule = hydra.utils.instantiate(kwargs["Conv3DModule"])

        self.run_embeddings_individually = True  # save memory
        self.return_global_only = True

    def forward_to_global_feature_vec(self, x):
        if len(os.listdir(self.model_example_dir)) == 0:
            self.save_model_example_input(x)

        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        elif len(x.shape) == 5 and x.shape[1] == 1:
            next

        feature_vec = self.Conv3DModule.forward_to_global_vec(x, return_global_only=self.return_global_only)

        if self.return_global_only:
            feature_vec = self.MLP(feature_vec)
            return feature_vec

        recon_x = None
        orig_x = None
        if isinstance(feature_vec, dict):
            recon_x = feature_vec["recon_x"]
            orig_x = feature_vec["orig_x"]
            feature_vec = feature_vec["global_vector"]

        feature_vec = self.MLP(feature_vec)
        return dict(feature_vec=feature_vec, recon_x=recon_x, orig_x=orig_x)


class StdPool3d(nn.Module):
    def __init__(self, kernel_size: int, stride: Optional[int] = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = F.avg_pool3d(x, self.kernel_size, self.stride, self.padding)
        squared_diff = (x - mean) ** 2
        variance = F.avg_pool3d(squared_diff, self.kernel_size, self.stride, self.padding)
        std = torch.sqrt(variance + 1e-8)
        return std


class Conv3DNetworkSTD(nn.Module):
    def __init__(self, final_size_out=1024, final_size_in=131072, kernel_size=3, pool_strides=[2, 2, 2]):
        super().__init__()

        self.conv1 = nn.Conv3d(1, 8, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm3d(8)
        self.pool1 = StdPool3d(2, stride=pool_strides[0])

        self.conv2 = nn.Conv3d(8, 16, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm3d(16)
        self.pool2 = StdPool3d(2, stride=pool_strides[1])

        self.conv3 = nn.Conv3d(16, 32, kernel_size=kernel_size, stride=1, padding=1)
        self.norm3 = nn.BatchNorm3d(32)
        self.pool3 = StdPool3d(2, stride=pool_strides[2])

        self.fc1 = nn.Linear(final_size_in, final_size_out)
        self.grid_size = [128, 128, 128]

    def voxelize_point_cloud(self, points: torch.Tensor) -> torch.Tensor:
        min_bound = -1.0
        max_bound = 1.0
        batch_size = points.shape[0]

        points_normalized = (points.transpose(-1, -2) - min_bound) / (max_bound - min_bound)
        points_scaled = points_normalized * torch.tensor(self.grid_size, dtype=torch.float32, device=points.device)

        voxel_indices = torch.floor(points_scaled).to(torch.int32)
        voxel_indices = torch.clamp(voxel_indices, 0, self.grid_size[0] - 1)

        gradient_volume = torch.zeros((batch_size, *self.grid_size), device=torch.device("cuda"))

        for i, voxel_index in enumerate(zip(voxel_indices)):
            z, y, x = voxel_index[0].t()
            i = (torch.ones_like(z) * i).to(torch.int32)
            gradient_volume[i, z, y, x] = gradient_volume[i, z, y, x] + torch.sign(torch.mean(points_normalized[i][0].reshape(-1, 3) + 100, 1))

        return gradient_volume.unsqueeze(1)

    def check_is_vox(self, x: torch.Tensor) -> bool:
        xs1 = x.shape[-3]
        xs2 = x.shape[-2]
        xs3 = x.shape[-1]
        return xs1 == 128 and xs2 == 128 and xs3 == 128

    def forward_to_global_vec(self, x: torch.Tensor) -> torch.Tensor:
        is_vox = self.check_is_vox(x)
        if not is_vox:
            x = self.voxelize_point_cloud(x)

        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = self.pool3(F.relu(self.norm3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        return x


class Conv3DNetwork(nn.Module):  # 139264  #131072
    def __init__(self, final_size_out=1024, final_size_in=139264, kernel_size=3, pool_strides=[2, 2, 2], check_vox=False):
        super().__init__()

        self.conv1 = nn.Conv3d(1, 8, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm3d(8)
        self.pool1 = nn.MaxPool3d(2, stride=pool_strides[0])

        self.conv2 = nn.Conv3d(8, 16, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(2, stride=pool_strides[1])

        self.conv3 = nn.Conv3d(16, 32, kernel_size=kernel_size, stride=1, padding=1)
        self.norm3 = nn.BatchNorm3d(32)
        self.pool3 = nn.MaxPool3d(2, stride=pool_strides[2])

        self.fc1 = nn.Linear(final_size_in, final_size_out)
        self.grid_size = [128, 128, 128]
        self.check_vox = check_vox
        self.return_global_only = True

    def forward_to_global_vec(self, x: torch.Tensor, return_global_only: bool = True) -> torch.Tensor:
        if self.check_vox:
            is_vox = self.check_is_vox(x)
            if not is_vox:
                x = self.voxelize_point_cloud(x)

        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = self.pool3(F.relu(self.norm3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

    def voxelize_point_cloud(self, points: torch.Tensor) -> torch.Tensor:
        min_bound = -1.0
        max_bound = 1.0
        batch_size = points.shape[0]

        points_normalized = (points.transpose(-1, -2) - min_bound) / (max_bound - min_bound)
        points_scaled = points_normalized * torch.tensor(self.grid_size, dtype=torch.float32, device=points.device)

        voxel_indices = torch.floor(points_scaled).to(torch.int32)
        voxel_indices = torch.clamp(voxel_indices, 0, self.grid_size[0] - 1)

        gradient_volume = torch.zeros((batch_size, *self.grid_size), device=torch.device("cuda"))

        for i, voxel_index in enumerate(zip(voxel_indices)):
            z, y, x = voxel_index[0].t()
            i = (torch.ones_like(z) * i).to(torch.int32)
            gradient_volume[i, z, y, x] = gradient_volume[i, z, y, x] + torch.sign(torch.mean(points_normalized[i][0].reshape(-1, 3) + 100, 1))

        return gradient_volume.unsqueeze(1)

    def check_is_vox(self, x: torch.Tensor) -> bool:
        xs1 = x.shape[-3]
        xs2 = x.shape[-2]
        xs3 = x.shape[-1]
        return xs1 == 128 and xs2 == 128 and xs3 == 128


class Conv3DNetworkAdaptMax(nn.Module):
    def __init__(self, final_size_out=512, adapt_channels_final=16, kernel_size=3, pool_strides=[2, 2, 2], check_vox=False):
        super().__init__()

        self.conv1 = nn.Conv3d(1, 8, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm3d(8)
        self.pool1 = nn.MaxPool3d(2, stride=pool_strides[0])

        self.conv2 = nn.Conv3d(8, 16, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(2, stride=pool_strides[1])

        self.conv3 = nn.Conv3d(16, 32, kernel_size=kernel_size, stride=1, padding=1)
        self.norm3 = nn.BatchNorm3d(32)
        self.pool3 = nn.MaxPool3d(2, stride=pool_strides[2])

        self.final = nn.AdaptiveMaxPool3d(output_size=[adapt_channels_final, 1, 1])
        self.grid_size = [128, 128, 128]
        self.check_vox = check_vox
        self.final_size_out = final_size_out

    def forward_to_global_vec(self, x: torch.Tensor) -> torch.Tensor:
        if self.check_vox:
            is_vox = self.check_is_vox(x)
            if not is_vox:
                x = self.voxelize_point_cloud(x)

        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = self.pool3(F.relu(self.norm3(self.conv3(x))))

        x = self.final(x)
        x = x.view(x.size(0), 512)
        x = F.relu(x)
        return x

    def voxelize_point_cloud(self, points: torch.Tensor) -> torch.Tensor:
        min_bound = -1.0
        max_bound = 1.0
        batch_size = points.shape[0]

        points_normalized = (points.transpose(-1, -2) - min_bound) / (max_bound - min_bound)
        points_scaled = points_normalized * torch.tensor(self.grid_size, dtype=torch.float32, device=points.device)

        voxel_indices = torch.floor(points_scaled).to(torch.int32)
        voxel_indices = torch.clamp(voxel_indices, 0, self.grid_size[0] - 1)

        gradient_volume = torch.zeros((batch_size, *self.grid_size), device=torch.device("cuda"))

        for i, voxel_index in enumerate(zip(voxel_indices)):
            z, y, x = voxel_index[0].t()
            i = (torch.ones_like(z) * i).to(torch.int32)
            gradient_volume[i, z, y, x] = gradient_volume[i, z, y, x] + torch.sign(torch.mean(points_normalized[i][0].reshape(-1, 3) + 100, 1))

        return gradient_volume.unsqueeze(1)

    def check_is_vox(self, x: torch.Tensor) -> bool:
        xs1 = x.shape[-3]
        xs2 = x.shape[-2]
        xs3 = x.shape[-1]
        return xs1 == 128 and xs2 == 128 and xs3 == 128


class Conv3DNetworkLeftToRight(Conv3DNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_to_global_vec(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 1, 4, 2, 3)

        is_vox = self.check_is_vox(x)
        if not is_vox:
            x = self.voxelize_point_cloud(x)

        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = self.pool3(F.relu(self.norm3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


class Conv3DNetworKNose(Conv3DNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_to_global_vec(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :, -64:, 63 - 32 : 63 + 32, 63 - 32 : 63 + 32]

        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = self.pool3(F.relu(self.norm3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


class Conv3DNetworkSeven(Conv3DNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        pool_strides = kwargs[pool_strides]
        kernel_size = kwargs[kernel_size]
        self.conv1 = nn.Conv3d(1, 8, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm3d(8)
        self.pool1 = nn.MaxPool3d(2, stride=pool_strides[0])

        self.conv2 = nn.Conv3d(8, 16, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(2, stride=pool_strides[1])

        self.conv3 = nn.Conv3d(16, 32, kernel_size=kernel_size, stride=1, padding=1)
        self.norm3 = nn.BatchNorm3d(32)
        self.pool3 = nn.MaxPool3d(2, stride=pool_strides[2])

        self.fc1 = nn.Linear(final_size_in, final_size_out)
        self.grid_size = [128, 128, 128]


class Conv3DNetworkSevenFiveSeven(Conv3DNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        pool_strides = [2, 2, 2]
        ks = [3, 3, 3]

        self.conv1 = nn.Conv3d(1, 8, kernel_size=ks[0], stride=1, padding=1)
        self.norm1 = nn.BatchNorm3d(8)
        self.pool1 = nn.MaxPool3d(2, stride=pool_strides[0])

        self.conv2 = nn.Conv3d(8, 16, kernel_size=ks[1], stride=1, padding=1)
        self.norm2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(2, stride=pool_strides[1])

        self.conv3 = nn.Conv3d(16, 32, kernel_size=ks[2], stride=1, padding=1)
        self.norm3 = nn.BatchNorm3d(32)
        self.pool3 = nn.MaxPool3d(2, stride=pool_strides[2])

        self.fc1 = nn.Linear(final_size_in, final_size_out)
        self.grid_size = [128, 128, 128]


def get_block(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: Optional[int] = None, activation: bool = True):
    if padding is None:
        if kernel_size == 1:
            padding = 0
        elif kernel_size == 3:
            padding = 1
        elif kernel_size == 7:
            padding = 3
    conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    if activation:
        return nn.Sequential(conv, nn.BatchNorm3d(out_channels), nn.LeakyReLU(0.2))
    else:
        return nn.Sequential(conv, nn.BatchNorm3d(out_channels))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.b1 = get_block(in_channels, out_channels, 3, stride)
        self.b2 = get_block(out_channels, out_channels, 3, 1, activation=False)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.b1(x)
        out = self.b2(out)
        out += residual
        out = F.leaky_relu(out, 0.2)
        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.b1 = get_block(in_channels, out_channels, 1, 1)
        self.b2 = get_block(out_channels, out_channels, 3, stride)
        self.b3 = get_block(out_channels, out_channels * self.expansion, 1, 1, activation=False)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.b1(x)
        out = self.b2(out)
        out = self.b3(out)
        out += residual
        out = F.leaky_relu(out, 0.2)
        return out


class ResNet3D(nn.Module):
    def __init__(self, block: type[nn.Module], num_blocks: Sequence[int], num_channels: Sequence[int], num_classes: int, pooltype: str = "max"):
        super().__init__()
        self.init1 = get_block(1, num_channels[0], 7, 1)

        if pooltype == "max":
            self.init2 = nn.MaxPool3d(3, 2, 1)
        elif pooltype == "mean":
            self.init2 = nn.AvgPool3d(3, 2, 1)

        in_channels = num_channels[0]
        module_list = []
        for i, (t_num_blocks, out_channels) in enumerate(zip(num_blocks, num_channels)):
            this_layer = self._make_layer(block, t_num_blocks, in_channels, out_channels, 1 if i == 0 else 2)
            in_channels = block.expansion * out_channels
            module_list.append(this_layer)
        self.resmodules = nn.Sequential(*module_list)
        self.avgpool = nn.AdaptiveAvgPool3d((512, 1, 1))
        self.fc = nn.Linear(32768, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.init1(x)
        out = self.init2(out)
        out = self.resmodules(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def _make_layer(self, block: type[nn.Module], num_blocks: int, in_channels: int, out_channels: int, stride: int):
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = get_block(in_channels, out_channels * block.expansion, 1, stride, activation=False)
        else:
            downsample = None
        layers = [block(in_channels, out_channels, stride, downsample)]
        for _ in range(num_blocks - 1):
            layers.append(block(out_channels * block.expansion, out_channels))
        return nn.Sequential(*layers)


def init_weights_resnet3d(m: nn.Module):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Conv3DNetworkResnet3D(nn.Module):
    def __init__(self, final_size_out=512, num_blocks=[1, 2, 4, 8], pooltype="mean", **kwargs):
        super().__init__()

        model = ResNet3D(block=BasicBlock, num_blocks=num_blocks, num_channels=[res_d, res_d * 2, res_d * 4, res_d * 8], num_classes=1, pooltype=pooltype)
        model.apply(init_weights_resnet3d)

        self.resnet3d = model
        self.grid_size = [128, 128, 128]
        self.check_vox = False
        self.final_size_out = final_size_out

    def forward_to_global_vec(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet3d(x)
        return x


__all__ = [
    "Conv3DNetwork",
    "Conv3DNetworkEnsemble",
    "Conv3DNetworkLeftToRight",
    "Conv3DNetworkAdaptMax",
    "Conv3DNetworkSTD",
    "Conv3DNetworkSeven",
    "Conv3DNetworkSevenFiveSeven",
    "Conv3DNetworkResnet3D",
    "Conv3DNetworKNose",
    "StdPool3d",
    "ResNet3D",
    "BasicBlock",
    "BottleneckBlock",
    "init_weights_resnet3d",
    "get_block",
]
