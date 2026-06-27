"""
CurveNet reward and segmentation models extracted from the all_models monolith.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core_modules.models.base import UniversalRWDModel
from core_modules.models.utils_base import MLP
from core_modules.models.utils_curvenet import LPFA, CIC, PointNetFeaturePropagation

# Curve configuration used for both classification and segmentation variants.
curve_config = {"default": [[100, 5], [100, 5], None, None, None], "long": [[10, 30], None, None, None, None]}


class CurveNetRWD(nn.Module):
    def __init__(self, num_classes=1, k=20, setting="default", **kwargs):
        super().__init__()

        assert setting in curve_config

        additional_channel = 32
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        # encoder
        self.cic11 = CIC(
            npoint=1024,
            radius=0.05,
            k=k,
            in_channels=additional_channel,
            output_channels=64,
            bottleneck_ratio=2,
            mlp_num=1,
            curve_config=curve_config[setting][0],
        )
        self.cic12 = CIC(
            npoint=1024,
            radius=0.05,
            k=k,
            in_channels=64,
            output_channels=64,
            bottleneck_ratio=4,
            mlp_num=1,
            curve_config=curve_config[setting][0],
        )

        self.cic21 = CIC(
            npoint=1024,
            radius=0.05,
            k=k,
            in_channels=64,
            output_channels=128,
            bottleneck_ratio=2,
            mlp_num=1,
            curve_config=curve_config[setting][1],
        )
        self.cic22 = CIC(
            npoint=1024,
            radius=0.1,
            k=k,
            in_channels=128,
            output_channels=128,
            bottleneck_ratio=4,
            mlp_num=1,
            curve_config=curve_config[setting][1],
        )

        self.cic31 = CIC(
            npoint=256,
            radius=0.1,
            k=k,
            in_channels=128,
            output_channels=256,
            bottleneck_ratio=2,
            mlp_num=1,
            curve_config=curve_config[setting][2],
        )
        self.cic32 = CIC(
            npoint=256,
            radius=0.2,
            k=k,
            in_channels=256,
            output_channels=256,
            bottleneck_ratio=4,
            mlp_num=1,
            curve_config=curve_config[setting][2],
        )

        self.cic41 = CIC(
            npoint=64,
            radius=0.2,
            k=k,
            in_channels=256,
            output_channels=512,
            bottleneck_ratio=2,
            mlp_num=1,
            curve_config=curve_config[setting][3],
        )
        self.cic42 = CIC(
            npoint=64,
            radius=0.4,
            k=k,
            in_channels=512,
            output_channels=512,
            bottleneck_ratio=4,
            mlp_num=1,
            curve_config=curve_config[setting][3],
        )

        self.conv0 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=False),
        )
        self.conv1 = nn.Linear(1024 * 2, 512, bias=False)
        self.conv2 = nn.Linear(512, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)

    def forward(self, xyz):
        l0_points = self.lpfa(xyz, xyz)

        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)

        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)

        x = self.conv0(l4_points)
        x_max = F.adaptive_max_pool1d(x, 1)
        x_avg = F.adaptive_avg_pool1d(x, 1)

        x = torch.cat((x_max, x_avg), dim=1).squeeze(-1)
        x = F.relu(self.bn1(self.conv1(x).unsqueeze(-1)), inplace=False).squeeze(-1)
        x = self.dp1(x)
        x = self.conv2(x)
        return x

    def forward_to_global_vec(self, xyz):
        l0_points = self.lpfa(xyz, xyz)

        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)

        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)

        x = self.conv0(l4_points)
        x_max = F.adaptive_max_pool1d(x, 1)
        x_avg = F.adaptive_avg_pool1d(x, 1)

        x = torch.cat((x_max, x_avg), dim=1).squeeze(-1)
        return x


class CurveNetRWD_SEG(nn.Module):
    def __init__(self, num_classes=1, k=32, setting="default", **kwargs):
        super().__init__()

        category = 3
        assert setting in curve_config

        additional_channel = 32
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        # encoder
        self.cic11 = CIC(npoint=2048, radius=0.2, k=k, in_channels=additional_channel, output_channels=64, bottleneck_ratio=2, curve_config=curve_config[setting][0])
        self.cic12 = CIC(npoint=2048, radius=0.2, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4, curve_config=curve_config[setting][0])

        self.cic21 = CIC(npoint=512, radius=0.4, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2, curve_config=curve_config[setting][1])
        self.cic22 = CIC(npoint=512, radius=0.4, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4, curve_config=curve_config[setting][1])

        self.cic31 = CIC(npoint=128, radius=0.8, k=k, in_channels=128, output_channels=256, bottleneck_ratio=2, curve_config=curve_config[setting][2])
        self.cic32 = CIC(npoint=128, radius=0.8, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4, curve_config=curve_config[setting][2])

        self.cic41 = CIC(npoint=32, radius=1.2, k=31, in_channels=256, output_channels=512, bottleneck_ratio=2, curve_config=curve_config[setting][3])
        self.cic42 = CIC(npoint=32, radius=1.2, k=31, in_channels=512, output_channels=512, bottleneck_ratio=4, curve_config=curve_config[setting][3])

        self.cic51 = CIC(npoint=8, radius=2.0, k=7, in_channels=512, output_channels=1024, bottleneck_ratio=2, curve_config=curve_config[setting][4])
        self.cic52 = CIC(npoint=8, radius=2.0, k=7, in_channels=1024, output_channels=1024, bottleneck_ratio=4, curve_config=curve_config[setting][4])
        self.cic53 = CIC(npoint=8, radius=2.0, k=7, in_channels=1024, output_channels=1024, bottleneck_ratio=4, curve_config=curve_config[setting][4])

        # decoder
        self.fp4 = PointNetFeaturePropagation(in_channel=1024 + 512, mlp=[512, 512], att=[1024, 512, 256])
        self.up_cic5 = CIC(npoint=32, radius=1.2, k=31, in_channels=512, output_channels=512, bottleneck_ratio=4)

        self.fp3 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256], att=[512, 256, 128])
        self.up_cic4 = CIC(npoint=128, radius=0.8, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4)

        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[128, 128], att=[256, 128, 64])
        self.up_cic3 = CIC(npoint=512, radius=0.4, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4)

        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 64, mlp=[64, 64], att=[128, 64, 32])
        self.up_cic2 = CIC(npoint=2048, radius=0.2, k=k, in_channels=1603, output_channels=256, bottleneck_ratio=4)
        self.up_cic1 = CIC(npoint=2048, radius=0.2, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4)

        self.global_conv2 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.global_conv1 = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv1 = nn.Conv1d(256, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(256, num_classes, 1)
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(256, 256 // 8, 1, bias=False), nn.BatchNorm1d(256 // 8), nn.LeakyReLU(negative_slope=0.2), nn.Conv1d(256 // 8, 256, 1, bias=False), nn.Sigmoid())

    def forward(self, xyz):
        l0_points = self.lpfa(xyz, xyz)

        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)

        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)

        l5_xyz, l5_points = self.cic51(l4_xyz, l4_points)
        l5_xyz, l5_points = self.cic52(l5_xyz, l5_points)
        l5_xyz, l5_points = self.cic53(l5_xyz, l5_points)

        l4_points = self.global_conv2(l5_points) + self.global_conv1(l4_points)
        l3_points = self.fp4(l3_xyz, l5_xyz, l3_points, l4_points)
        l3_points = self.up_cic5(l3_xyz, l3_points)[1]

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points = self.up_cic4(l2_xyz, l2_points)[1]

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points = self.up_cic3(l1_xyz, l1_points)[1]

        l0_points = self.fp1(xyz, l1_xyz, l0_points, l1_points)
        l0_points = self.up_cic2(xyz, l0_points)[1]
        l0_points = self.up_cic1(xyz, l0_points)[1]

        x = self.conv1(l0_points)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.se(x) * x
        x = self.drop1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2).contiguous()

        return x

    def forward_to_global_vec(self, xyz):
        x = self.forward(xyz)
        return x


class CurveNetEnsemble(UniversalRWDModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.save_hyperparameters(logger=False)
        self.return_global_embedding = kwargs["return_global_embedding"]
        self.external = kwargs["external"]

        self.CurveNetModule = CurveNetRWD()

    def forward_to_global_feature_vec(self, x):
        feature_vec = self.CurveNetModule.forward_to_global_vec(x)
        feature_vec = self.MLP(feature_vec)
        return feature_vec


class CurveNetEnsemble_SEG(UniversalRWDModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.save_hyperparameters(logger=False)

        mlp_global = kwargs["mlp_global"]
        self.return_global_embedding = kwargs["return_global_embedding"]
        self.external = kwargs["external"]

        self.CurveNetModule = CurveNetRWD_SEG()

        self.MLP = MLP(
            input_size=mlp_global["input_size"],
            output_size=mlp_global["output_size"],
            hidden_sizes=mlp_global["hidden_sizes"],
            dropout_rate=mlp_global["dropout_rate"],
            normalisation_type=mlp_global["normalisation_type"],
            activation_type=mlp_global["activation_type"],
            use_dropout=mlp_global["use_dropout"],
        )

        self.scalar_rwd_head = MLP(
            input_size=self.global_feature_size,
            output_size=1,
            hidden_sizes=[256, 128],
            dropout_rate=kwargs["mlp_global"]["dropout_rate"],
            normalisation_type="none",
            activation_type="softplus",
            use_dropout=False,
        )

        self.scalar_rwd_head_BT = MLP(
            input_size=self.global_feature_size,
            output_size=1,
            hidden_sizes=[256, 128],
            dropout_rate=kwargs["mlp_global"]["dropout_rate"],
            normalisation_type="none",
            activation_type="softplus",
            use_dropout=False,
        )

        self.scalar_rwd_head_pairs = MLP(
            input_size=self.global_feature_size * 2,
            output_size=2,
            hidden_sizes=[512, 256, 128],
            dropout_rate=kwargs["mlp_global"]["dropout_rate"],
            normalisation_type="none",
            activation_type="softplus",
            use_dropout=False,
        )

    def forward_to_global_feature_vec(self, x):
        feature_vec = self.CurveNetModule.forward_to_global_vec(x)
        feature_vec = self.MLP(feature_vec)
        return feature_vec


__all__ = [
    "CurveNetRWD",
    "CurveNetRWD_SEG",
    "CurveNetEnsemble",
    "CurveNetEnsemble_SEG",
    "curve_config",
]
