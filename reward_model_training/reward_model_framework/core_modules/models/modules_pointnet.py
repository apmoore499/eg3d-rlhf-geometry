"""
PointNet/PointNet++ reward models extracted from the all_models monolith.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn import BatchNorm1d, InstanceNorm1d, LayerNorm, Linear as Lin, Sequential as Seq
from torch_geometric.data import Data
from torch_geometric.nn import PointNetConv, fps, radius
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add, scatter_max

from core_modules.models.base import UniversalRWDModel

from core_modules.models.utils_base import MLP, make_mlp, fourier_features, reduce_x


import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class PointNet2SAModule(torch.nn.Module):
    def __init__(self, sample_radio, radius, max_num_neighbors, mlp, add_self_loops=True, aggr="max"):
        super().__init__()
        self.sample_ratio = sample_radio
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors
        self.point_conv = PointNetConv(mlp, add_self_loops=add_self_loops, aggr=aggr)

    def forward(self, data):
        x, pos, batch = data
        idx = fps(pos, batch, ratio=self.sample_ratio)
        row, col = radius(pos, pos[idx], self.radius, batch, batch[idx], max_num_neighbors=self.max_num_neighbors)
        edge_index = torch.stack([col, row], dim=0)
        x1 = self.point_conv(x, (pos, pos[idx]), edge_index)
        pos1, batch1 = pos[idx], batch[idx]
        return x1, pos1, batch1


class PointNet2GlobalSAModule(torch.nn.Module):
    """One group with all input points, can be viewed as a simple PointNet module."""

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, data):
        x, pos, batch = data
        if x is not None:
            x = torch.cat([x, pos], dim=1)
        x1 = self.mlp(x)
        x1 = scatter_max(x1, batch, dim=0)[0]
        batch_size = x1.shape[0]
        pos1 = x1.new_zeros((batch_size, 3))
        batch1 = torch.arange(batch_size).to(batch.device, batch.dtype)
        return x1, pos1, batch1


class PointConvFP(MessagePassing):
    """Core layer of Feature propagtaion module."""

    def __init__(self, mlp=None):
        super().__init__()
        self.mlp = mlp
        self.aggr = "add"
        self.flow = "source_to_target"
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp)

    def forward(self, x, pos, edge_index):
        x_tmp = x[0] if x[1] is None else x
        aggr_out = self.propagate(edge_index, x=x_tmp, pos=pos)
        i, j = (0, 1) if self.flow == "target_to_source" else (1, 0)
        x_target, pos_target = x[i], pos[i]
        add = [pos_target] if x_target is None else [x_target, pos_target]
        aggr_out = torch.cat([aggr_out, *add], dim=1)
        if self.mlp is not None:
            aggr_out = self.mlp(aggr_out)
        return aggr_out

    def message(self, x_j, pos_j, pos_i, edge_index):
        dist = (pos_j - pos_i).pow(2).sum(dim=1).pow(0.5)
        dist = torch.max(dist, torch.Tensor([1e-10]).to(dist.device, dist.dtype))
        weight = 1.0 / dist
        row, col = edge_index
        index = col
        num_nodes = maybe_num_nodes(index, None)
        wsum = scatter_add(weight, col, dim=0, dim_size=num_nodes)[index] + 1e-16
        weight /= wsum
        return weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class PointNet2FPModule(torch.nn.Module):
    def __init__(self, knn_num, mlp):
        super().__init__()
        self.knn_num = knn_num
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = mlp[0]
        self.activate_function = torch.relu
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.fa1 = Attention(mlp[1], self.activate_function)
        self.fa2 = Attention(mlp[1], self.activate_function)
        self.fa3 = Attention(mlp[1], self.activate_function)
        self.fa4 = Attention(mlp[1], self.activate_function)
        self.fa5 = Attention(mlp[1], self.activate_function)

    def forward(self, xyz1, xyz2, points1, points2, fps_idx):
        _, _, C2 = points2.shape
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            new_points = interpolated_points.permute(0, 2, 1) + points1
        else:
            new_points = interpolated_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = self.activate_function(bn(conv(new_points)))
        return new_points


class Attention(nn.Module):
    def __init__(self, in_channel, activate_function):
        super().__init__()
        self.trans = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=in_channel // 4, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channel // 4),
            activate_function,
            nn.Conv1d(in_channels=in_channel // 4, out_channels=in_channel, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = x
        x_t = x.permute(0, 2, 1)
        att = torch.bmm(x, x_t)
        att = torch.sum(att, dim=-1, keepdim=True)
        att = self.trans(att)
        return y * att


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx.long(), :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :]
        centroid = centroid.view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def sample_and_group(npoint, radius, nsample, xyz, points, knn, csa_p):
    B, N, C = xyz.shape
    Bf, Nf, Cf = points.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz.contiguous(), npoint)
    new_xyz = index_points(xyz, fps_idx)
    if knn:
        dists = square_distance(new_xyz, xyz)
        idx = dists.argsort()[:, :, :nsample]
    else:
        idx = query_ball_point(radius, nsample, xyz.contiguous(), new_xyz.contiguous())
    grouped_points = index_points(points, idx)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if csa_p is not None:
        csa_position = index_points(csa_p, idx)
        csa_position = torch.cat([csa_position, grouped_xyz_norm, new_xyz.view(B, S, 1, C).expand_as(grouped_xyz_norm)], dim=-1)
    else:
        csa_position = grouped_xyz_norm
    csa_feature = torch.cat([grouped_points, grouped_xyz_norm, new_xyz.view(B, S, 1, C).expand_as(grouped_xyz_norm)], dim=-1)
    return new_xyz, csa_position, csa_feature, grouped_xyz_norm, fps_idx


class Pointnet2_RWD(torch.nn.Module):
    """
    Modified PointNet++ to produce global embeddings.
    """

    def __init__(
        self,
        add_self_loops=True,
        n_features=0,
        n_fourier_freq=0,
        aggr="max",
        use_residual_gconv_layers=False,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = 2
        self.model_class = "scalar_reward_pointnet"

        if not isinstance(aggr, (str, list)):
            aggr = OmegaConf.to_object(aggr)

        self.n_fourier_freq = n_fourier_freq
        self.use_residual_gconv_layers = use_residual_gconv_layers
        mlp_settings = kwargs["mlp_settings"]

        if n_features == 0:
            n_features = n_fourier_freq * 3 * 2
        else:
            raise AssertionError("error fix n features in code...")

        sa1_sample_ratio = 0.5
        sa1_radius = 0.2
        sa1_max_num_neighbours = 64
        sa1_mlp = make_mlp(3 + n_features, [64, 64, 128], **mlp_settings)
        self.sa1_module = PointNet2SAModule(sa1_sample_ratio, sa1_radius, sa1_max_num_neighbours, sa1_mlp, add_self_loops, aggr=aggr)

        sa2_sample_ratio = 0.25
        sa2_radius = 0.4
        sa2_max_num_neighbours = 64
        sa2_mlp = make_mlp(128 + 3, [128, 128, 256], **mlp_settings)
        self.sa2_module = PointNet2SAModule(sa2_sample_ratio, sa2_radius, sa2_max_num_neighbours, sa2_mlp, add_self_loops, aggr=aggr)

        if self.use_residual_gconv_layers:
            sa3_mlp = make_mlp(256 + 128 + 3, [256 + 128, 512, 1024], **mlp_settings)
        else:
            sa3_mlp = make_mlp(256 + 3, [256, 512, 1024], **mlp_settings)
        self.sa3_module = PointNet2GlobalSAModule(sa3_mlp)

        self.mlp_global = kwargs["mlp_global"]
        self.affine_offset = nn.Parameter(torch.tensor(0.0))
        self.affine_scale = nn.Parameter(torch.tensor(1.0))
        self.affine_offset.requires_grad = False
        self.affine_scale.requires_grad = False

    def forward(self, data):
        data_in = self.format_dense_input(data)
        sa1_out = self.sa1_module(data_in)
        sa2_out = self.sa2_module(sa1_out)
        if self.use_residual_gconv_layers:
            sa2_out[0] = torch.cat([sa2_out[0], sa1_out[0]], dim=1)
            sa2_out[2] = torch.hstack((sa1_out[2], sa2_out[2]))
        sa3_out = self.sa3_module(sa2_out)
        global_feature_vec = sa3_out[0]
        return global_feature_vec

    def format_dense_input(self, data):
        dense_input = True if isinstance(data, torch.Tensor) else False
        has_extra = False
        if dense_input:
            if data.shape[1] > 3:
                has_extra = True
                extra = data[:, 3:].transpose(1, 2).contiguous()
                extra = extra.view(-1, extra.shape[-1])
                data = data[:, :3]

            data = data.transpose(1, 2).contiguous()
            batch_size, N, _ = data.shape
            pos = data.view(batch_size * N, -1)
            batch = torch.zeros((batch_size, N), device=pos.device, dtype=torch.long)
            for i in range(batch_size):
                batch[i] = i
            batch = batch.view(-1)

            data = Data()
            data.pos, data.batch = pos, batch
            if has_extra:
                data.x = extra

        if not hasattr(data, "x"):
            data.x = None

        if self.n_fourier_freq > 0:
            fourier_features_ = fourier_features(data.pos, self.n_fourier_freq, 0, 1)
            if data.x is None:
                data.x = fourier_features_.reshape(-1, self.n_fourier_freq * 3 * 2)
            else:
                data.x = torch.cat([data.x, fourier_features_], dim=1)

        data_in = data.x, data.pos, data.batch
        return data_in

    def forward_to_global_vec(self, data):
        data_in = self.format_dense_input(data)
        sa1_out = self.sa1_module(data_in)
        sa2_out = self.sa2_module(sa1_out)
        sa3_out = self.sa3_module(sa2_out)
        global_feature_vec = sa3_out[0]
        return global_feature_vec


class PointNetTwo(UniversalRWDModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)
        self.return_global_embedding = kwargs["return_global_embedding"]
        self.external = None
        self.PointNet2Module = Pointnet2_RWD(**kwargs)

    def forward_to_global_feature_vec(self, x):
        if len(os.listdir(self.model_example_dir)) == 0:
            ex_output_fn = os.path.join(self.model_example_dir, "model_example_input.pt")
            torch.save(obj=x, f=ex_output_fn)
        feature_vec = self.PointNet2Module.forward_to_global_vec(x)
        feature_vec = self.MLP(feature_vec)
        return feature_vec


class PointNetOne(UniversalRWDModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.save_hyperparameters(logger=False)

        mlp_global = kwargs["mlp_global"]
        feature_transform = kwargs["feature_transform"]
        agg_type = kwargs["agg_type"]

        self.agg_type = agg_type
        # global_feature_size=kwargs['global_feature_size']

        spatial_transform_3d = kwargs["spatial_transform_3d"]
        self.return_global_embedding = kwargs["return_global_embedding"]
        self.spatial_transform_3d = spatial_transform_3d
        self.global_feature_size = mlp_global["output_size"]

        if spatial_transform_3d:
            # self.stn = STN3d(agg_type=agg_type)
            self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        normtype = kwargs["normtype"].lower()

        if normtype == "instance" or normtype == "instance_norm":
            # m = nn.InstanceNorm1d(100, affine=True)
            self.bn1 = nn.InstanceNorm1d(64, affine=True)
            self.bn2 = nn.InstanceNorm1d(128, affine=True)
            self.bn3 = nn.InstanceNorm1d(1024, affine=True)

        elif normtype == "batch_norm" or normtype == "batchnorm":
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)

        elif normtype == "weight_norm" or normtype == "weightnorm":
            self.conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(3, 64, 1), dim=0)
            self.conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(64, 128, 1), dim=0)
            self.conv3 = torch.nn.utils.weight_norm(torch.nn.Conv1d(128, 1024, 1), dim=0)

            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()

        elif normtype == "none":
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()

        # self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            # self.fstn = STNkd(k=64, agg_type=self.agg_type)
            self.fstn = STNkd(k=64)

        # mlp_global['input_size']
        # mlp_global['output_size']

        self.external = kwargs["external"]

        # self.MLP=MLP(input_size=128, output_size=1, hidden_sizes=[128,128,128],dropout_rate=0.2,normalisation_type='none',activation_type='softplus')

        # # forward_to_scalar_reward_from_single_global
        # def forward_to_scalar_reward_from_single_global(self, x):
        #     # scalar_rwd=torch.nn.functional.softplus(self.scalar_rwd_head(x))+1e-3, maybe too unstable
        #     scalar_rwd = self.scalar_rwd_head(x)
        #     return scalar_rwd

        #     # forward_to_scalar_reward_from_single_global

        # def forward_to_BT_lambda_from_single_global(
        #     self, x, mult=1.0
        # ):  # multiplier may need to be set to a small value at start of trianing to prevent numerical instability
        #     # scalar_rwd=torch.nn.functional.softplus(self.scalar_rwd_head(x))+1e-3, maybe too unstable
        #     scalar_rwd = torch.exp(self.scalar_rwd_head_BT(x) * mult)
        #     return scalar_rwd

        self.activation = F.softplus  # get_activation(mlp_global["activation_type"])
        # self.hparams.optimizer=hydra.utils.instantiate(kwargs['optimizer'])#,convert='partial')

    def forward_to_pre_aggregation(self, x):
        n_pts = x.size()[2]

        if self.spatial_transform_3d:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)

        x = self.activation(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        x = self.activation(self.bn2(self.conv2(x)))
        # x = F.softplus(self.bn3(self.conv3(x)))
        # x = F.softplus(self.bn3(self.conv3(x)))

        #
        x = self.activation(self.bn3(self.conv3(x)))

        return x

    def forward_and_return_reduction_argvals(self, x):
        x = self.forward_to_pre_aggregation(x)

        agg_features = reduce_x(x, self.agg_type, return_idx=True)

        vals, idx = agg_features

        return {"vals": vals, "idx": idx}

    def forward_to_global_feature_vec(self, x):
        x = self.forward_to_pre_aggregation(x)
        agg_features = reduce_x(x, self.agg_type)
        feature_vec = self.MLP(agg_features)

        return feature_vec


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


__all__ = [
    "PointNetOne",
    "PointNet2SAModule",
    "PointNet2GlobalSAModule",
    "PointConvFP",
    "PointNet2FPModule",
    "Pointnet2_RWD",
    "PointNetTwo",
    "Attention",
    "square_distance",
    "index_points",
    "query_ball_point",
    "farthest_point_sample",
    "sample_and_group",
    "STN3d",
    "STNkd",
]
