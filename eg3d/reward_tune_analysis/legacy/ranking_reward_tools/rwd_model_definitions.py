import argparse
import copy
import glob
import json
import os
import pickle
import random
import sys

import autoroot  # noqa: F401

# checking all unique combinations of flattened depth maps
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision
from pandas_ods_reader import read_ods
from pyexcel import get_book
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.autograd import Variable

# --------------------------------------------------------------------------------------------
# Load necessary Pytorch packages
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from camera_utils import FOV_to_intrinsics, LookAtPoseSampler

RLHF_DIR = "/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/000_RLHF_AM"
RLHF_DIR = RLHF_DIR.replace("##", "000")

from typing import Any, Dict, List, Optional, Tuple, Union

# for WFLW 98 landmarks...
rhs_lmks = [
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    57,
    58,
    59,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    97,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    90,
    91,
    92,
    93,
    94,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
]


lhs_lmks = [
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    96,
    76,
    77,
    78,
    79,
    85,
    86,
    87,
    88,
    89,
    90,
    95,
    94,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
]


# torch models


class dmap_stacked_chans(nn.Module):
    def __init__(self, dmap_outdim=128, dmap_stack=[8, 16, 32], nrs=64, drop1_pc=0.3, **kwargs):
        super().__init__()
        assert nrs == 64 or nrs == 128
        self.conv1 = nn.Conv2d(1, dmap_stack[0], kernel_size=(3, 3), stride=1, padding=1)  # previously, all channels set to 16 were 32, resulting connected fc4 of 127008 model param
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(drop1_pc)
        self.conv2 = nn.Conv2d(dmap_stack[1], dmap_stack[2], kernel_size=(5, 5), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        if nrs == 128:
            if dmap_stack[-1] == 32:
                lin_dim = 127008
            if dmap_stack[-1] == 16:
                lin_dim = 63504
            if dmap_stack[-1] == 8:
                lin_dim = 31752
            if dmap_stack[-1] == 4:
                lin_dim = 15876
            if dmap_stack[-1] == 2:
                lin_dim = 7938
            if dmap_stack[-1] == 1:
                lin_dim = 3969

        elif nrs == 64:
            if dmap_stack[-1] == 16:
                lin_dim = 15376

            if dmap_stack[-1] == 32:
                lin_dim = 30752

            if dmap_stack[-1] == 8:
                lin_dim = 7688

        self.flat = nn.Flatten()
        self.fc3 = nn.Linear(lin_dim, dmap_outdim)  # 30752 #127008
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.3)
        self.nrs = nrs

    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        x = self.pool2(x)
        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        # go from the big vector to the small vector
        x = self.fc3(x)
        return x


class rl_decoder_three_dmap(nn.Module):
    def __init__(self, dmap_stack=[8, 8, 32], dmap_outdim=32, drop1_pc=0.3, **kwargs):
        super().__init__()

        self.dmap_net_first = dmap_stacked_chans(dmap_outdim, dmap_stack=dmap_stack, drop1_pc=drop1_pc, **kwargs)
        self.dmap_net_second = dmap_stacked_chans(dmap_outdim, dmap_stack=dmap_stack, drop1_pc=drop1_pc, **kwargs)
        self.dmap_net_third = dmap_stacked_chans(dmap_outdim, dmap_stack=dmap_stack, drop1_pc=drop1_pc, **kwargs)

        self.fc1 = nn.Linear(dmap_outdim * 3, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.nrs = self.dmap_net_first.nrs

        self.reward_model_type = "depth_map_3"

        self.projection_head = create_mlp([dmap_outdim * 3, dmap_outdim * 3, dmap_outdim * 3])

        self.p1_layer = nn.Linear(dmap_outdim * 3, dmap_outdim * 3)
        self.p2_layer = nn.Linear(dmap_outdim * 3, dmap_outdim * 3)

        self.forward_layer = nn.Linear(dmap_outdim * 3, 1)

    def forward(self, z_features, dmap):
        # in this instance, z_features is prompt (ie x)
        dmap_f1 = self.dmap_net_first.forward(dmap[:, 0, :, :].unsqueeze(1))  # left hand side
        dmap_f2 = self.dmap_net_first.forward(dmap[:, 1, :, :].unsqueeze(1))  # canonical (front on)
        dmap_f3 = self.dmap_net_first.forward(dmap[:, 2, :, :].unsqueeze(1))  # right hand side

        fc_input = torch.cat((dmap_f1, dmap_f2, dmap_f3), dim=1)  # concatenate all output then feed into fc layers

        x = self.p1_layer(fc_input)
        x = self.relu(x)
        x = self.forward_layer(x)
        # x = self.relu(self.fc1(fc_input))
        # x = self.fc2(x)

        return x

    def forward_projection_simclr(self, z_features, dmap):
        # in this instance, z_features is prompt (ie x)
        dmap_f1 = self.dmap_net_first.forward(dmap[:, 0, :, :].unsqueeze(1))  # left hand side
        dmap_f2 = self.dmap_net_first.forward(dmap[:, 1, :, :].unsqueeze(1))  # canonical (front on)
        dmap_f3 = self.dmap_net_first.forward(dmap[:, 2, :, :].unsqueeze(1))  # right hand side

        fc_input = torch.cat((dmap_f1, dmap_f2, dmap_f3), dim=1)  # concatenate all output then feed into fc layers

        x = self.p1_layer(fc_input)

        x = self.relu(x)

        x = self.p2_layer(x)

        projection = x

        return projection


class rl_decoder_three_dmap_lrg(nn.Module):
    def __init__(self, dmap_stack=[8, 8, 32], dmap_outdim=32, drop1_pc=0.05, **kwargs):
        super().__init__()

        self.dmap_net_first = dmap_stacked_chans(dmap_outdim, dmap_stack=dmap_stack, drop1_pc=drop1_pc, **kwargs)
        self.dmap_net_second = dmap_stacked_chans(dmap_outdim, dmap_stack=dmap_stack, drop1_pc=drop1_pc, **kwargs)
        self.dmap_net_third = dmap_stacked_chans(dmap_outdim, dmap_stack=dmap_stack, drop1_pc=drop1_pc, **kwargs)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.nrs = self.dmap_net_first.nrs
        self.reward_model_type = "depth_map_3"
        self.projection_head = create_mlp([dmap_outdim * 3, dmap_outdim * 3, dmap_outdim * 3])
        self.reward_prediction_head = create_mlp([dmap_outdim * 3, dmap_outdim, 1])

    def forward(self, z_features, dmap):
        # in this instance, z_features is prompt (ie x)
        dmap_f1 = self.dmap_net_first.forward(dmap[:, 0, :, :].unsqueeze(1))  # left hand side
        dmap_f2 = self.dmap_net_first.forward(dmap[:, 1, :, :].unsqueeze(1))  # canonical (front on)
        dmap_f3 = self.dmap_net_first.forward(dmap[:, 2, :, :].unsqueeze(1))  # right hand side

        fc_input = torch.cat((dmap_f1, dmap_f2, dmap_f3), dim=1)  # concatenate all output then feed into fc layers

        x = self.reward_prediction_head(fc_input)
        return x

    def forward_projection_simclr(self, z_features, dmap):
        # in this instance, z_features is prompt (ie x)
        dmap_f1 = self.dmap_net_first.forward(dmap[:, 0, :, :].unsqueeze(1))  # left hand side
        dmap_f2 = self.dmap_net_first.forward(dmap[:, 1, :, :].unsqueeze(1))  # canonical (front on)
        dmap_f3 = self.dmap_net_first.forward(dmap[:, 2, :, :].unsqueeze(1))  # right hand side

        fc_input = torch.cat((dmap_f1, dmap_f2, dmap_f3), dim=1)  # concatenate all output then feed into fc layers
        projection = self.projection_head(fc_input)

        return projection


class dmap_classifier_triple(nn.Module):
    def __init__(self, outdim=128, dmap_stack=[8, 8, 32], nrs=128, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, dmap_stack[0], kernel_size=(3, 3), stride=1, padding=1)  # previously, all channels set to 16 were 32, resulting connected fc4 of 127008 model param
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(dmap_stack[1], dmap_stack[2], kernel_size=(5, 5), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        dmap_chans = dmap_stack[-1]
        if nrs == 128:
            if dmap_chans == 32:
                lin_dim = int(17672 * 2 * 2)
            if dmap_chans == 16:
                lin_dim = int(17672 * 2)
            if dmap_chans == 8:
                lin_dim = int(31752)
            if dmap_chans == 4:
                lin_dim = int(17672 / 2)
            if dmap_chans == 2:
                lin_dim = int(17672 / 4)
            if dmap_chans == 1:
                lin_dim = int(17672 / 8)

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(lin_dim, outdim)  # 30752 #127008
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.3)
        self.nrs = 128
        # self.fc4 = nn.Linear(512, 10)

    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        x = self.pool2(x)
        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        # go from the big vector to the small vector
        x = self.fc3(x)
        return x


class rl_decoder_triple(nn.Module):
    def __init__(
        self,
        conditional_model=True,
        dmap_only_mod=False,
        dmap_stack=[8, 8, 32],
        nrs=128,
        conditional_dim=512,
        dmap_outdim=128,
        decoder_layers=[128, 64, 1],
        projection_layers=[],
        **kwargs,
    ):
        super().__init__()

        if conditional_model:
            mlp_input_dim = conditional_dim + dmap_outdim
        else:
            mlp_input_dim = dmap_outdim
        if not dmap_only_mod:
            self.decoder = create_mlp(decoder_layers)
            self.dmap_net = dmap_classifier_triple(outdim=dmap_outdim, dmap_stack=dmap_stack, nrs=nrs)
        else:
            self.dmap_net = dmap_classifier_triple(outdim=1, dmap_stack=dmap_stack, nrs=nrs)

        if len(projection_layers) > 0:
            self.projection_head = create_mlp(projection_layers)

        self.conditional_model = conditional_model  # if we are using (x,y) or just (y) unconditional
        self.dmap_only_mod = dmap_only_mod  # like just the CNN direct into scalar

        self.reward_model_type = "depth_map"

    def forward(self, z_features, dmap):
        if not self.dmap_only_mod:
            dmap_features = self.dmap_net.forward(dmap)  # first do some conv on it
            if self.conditional_model:
                x = torch.cat((dmap_features, z_features), dim=1)
            else:
                x = dmap_features
            x = self.decoder(x)
            return x
        else:  # just using the small cnn
            x = self.dmap_net.forward(dmap)
            return x

    def forward_projection_simclr(self, z_features, dmap):
        if not self.dmap_only_mod:
            dmap_features = self.dmap_net.forward(dmap)  # first do some conv on it
            if self.conditional_model:
                x = torch.cat((dmap_features, z_features), dim=1)
            else:
                x = dmap_features
            projection = self.projection_head(x)
            return projection


class dmap_classifier_single(nn.Module):
    def __init__(self, outdim=128, dmap_stack=[8, 8, 32], nrs=128, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, dmap_stack[0], kernel_size=(3, 3), stride=1, padding=1)  # previously, all channels set to 16 were 32, resulting connected fc4 of 127008 model param
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(dmap_stack[1], dmap_stack[2], kernel_size=(5, 5), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        dmap_chans = dmap_stack[-1]
        if nrs == 128:
            if dmap_chans == 32:
                lin_dim = int(63504 * 2)
            if dmap_chans == 16:
                lin_dim = 63504
            if dmap_chans == 8:
                lin_dim = 31752
            if dmap_chans == 4:
                lin_dim = 15876
            if dmap_chans == 2:
                lin_dim = 3969
            if dmap_chans == 1:
                lin_dim = int(3969 / 2)

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(lin_dim, outdim)  # 30752 #127008
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.3)
        self.nrs = 128
        # self.fc4 = nn.Linear(512, 10)

    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        x = self.pool2(x)
        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        # go from the big vector to the small vector
        x = self.fc3(x)
        return x


class rl_decoder(nn.Module):
    def __init__(
        self,
        conditional_model=True,
        dmap_only_mod=False,
        dmap_stack=[8, 8, 32],
        nrs=128,
        conditional_dim=512,
        dmap_outdim=128,
        decoder_layers=[128, 64, 1],
        projection_layers=[],
        **kwargs,
    ):
        super().__init__()

        if conditional_model:
            mlp_input_dim = conditional_dim + dmap_outdim
        else:
            mlp_input_dim = dmap_outdim
        if not dmap_only_mod:
            self.decoder = create_mlp(decoder_layers)
            self.dmap_net = dmap_classifier_single(outdim=dmap_outdim, dmap_stack=dmap_stack, nrs=nrs)
        else:
            self.dmap_net = dmap_classifier_single(outdim=1, dmap_stack=dmap_stack, nrs=nrs)
        if len(projection_layers) > 0:
            self.projection_head = create_mlp(projection_layers)

        self.conditional_model = conditional_model  # if we are using (x,y) or just (y) unconditional
        self.dmap_only_mod = dmap_only_mod  # like just the CNN direct into scalar

        self.reward_model_type = "depth_map"

    def forward(self, z_features, dmap):
        if not self.dmap_only_mod:
            dmap_features = self.dmap_net.forward(dmap)  # first do some conv on it
            if self.conditional_model:
                x = torch.cat((dmap_features, z_features), dim=1)
            else:
                x = dmap_features
            x = self.decoder(x)
            return x
        else:  # just using the small cnn
            x = self.dmap_net.forward(dmap)
            return x

    def forward_projection_simclr(self, z_features, dmap):
        if not self.dmap_only_mod:
            dmap_features = self.dmap_net.forward(dmap)  # first do some conv on it
            if self.conditional_model:
                x = torch.cat((dmap_features, z_features), dim=1)
            else:
                x = dmap_features

            projection = self.projection_head(x)
            return projection


# to cmopute the feature vector for depth map, no gradients req
class vgg19_to_4096(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        model = torchvision.models.vgg19(weights="VGG19_Weights.IMAGENET1K_V1")
        model.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)])
        self.vgg_model = model

        self.freeze_classifier_layers()
        self.freeze_inet_layers()

    def freeze_classifier_layers(self):
        for param in self.vgg_model.classifier.parameters():
            param.requires_grad = False
        return self

    def freeze_inet_layers(self):
        for param in self.vgg_model.features.parameters():
            param.requires_grad = False
        return self

    def forward(self, dmap):
        bsize = dmap.shape[0]
        dmap_f1 = self.vgg_model(dmap[:, 0, :, :].unsqueeze(1).expand(bsize, 3, 224, 224))
        dmap_f2 = self.vgg_model(dmap[:, 1, :, :].unsqueeze(1).expand(bsize, 3, 224, 224))
        dmap_f3 = self.vgg_model(dmap[:, 2, :, :].unsqueeze(1).expand(bsize, 3, 224, 224))
        feature_vector = torch.cat((dmap_f1, dmap_f2, dmap_f3), dim=1)  # concatenate all output then feed into fc layers
        return feature_vector


# bigger model w more layers
class rwd_model_3dmap_vgg_minimal(nn.Module):
    def __init__(self, nrs=128, n_hidden=1, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(4096 * 3, 256)

        for n in range(n_hidden):
            setattr(self, f"fc{n+2}", nn.Linear(256, 256))

        self.total_hidden_layers = n_hidden

        self.reward_model_type = "rwd_model_3dmap_vgg_minimal"
        self.projection_head = nn.Linear(256, 256)
        self.rwd_model_head = nn.Linear(256, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.nrs = nrs

        # to be used when we want to scale the reward
        self.affine_offset = nn.Parameter(torch.tensor(0.0))
        self.affine_scale = nn.Parameter(torch.tensor(1.0))

        self.affine_offset.requires_grad = False
        self.affine_scale.requires_grad = False

    def forward(self, dmap_fc_input):
        x = self.fc1(dmap_fc_input)
        x = self.relu(x)

        for k in range(self.total_hidden_layers):
            x = getattr(self, f"fc{k+2}")(x)
            x = self.relu(x)

        x = self.rwd_model_head(x)

        x = (x - self.affine_offset) / self.affine_scale
        return x

    def forward_projection_simclr(self, dmap_fc_input):
        x = self.fc1(dmap_fc_input)
        x = self.relu(x)
        for k in range(self.total_hidden_layers):
            x = getattr(self, f"fc{k+2}")(x)
            x = self.relu(x)
        x = self.projection_head(x)
        projection = x
        return projection


# from facenet_pytorch import MTCNN

from facenet_pytorch import InceptionResnetV1 as inception_resnet_v1_fnet

# If required, create a face detection pipeline using MTCNN:
# mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)
# Create an inception resnet (in eval mode):


# arcface

sys.path.append("/media/krillman/1TB_DATA/codes/HFGI3D/inversion/scripts/")


# import id_loss_module_3DGIPO as id_loss_module_3DGIPO
# id_loss_face = id_loss_module_3DGIPO.IDLoss().cuda().eval()


# to cmopute the feature vector for depth map, no gradients req
# this one use facenet512 instead of the vgg module................
# facenet512 is vggface2 and inceptionv3 perhaps, not sure.
# inputs are in the range -1,1


def normfunc(dmap):
    retval = ((dmap - 2.25) / (3.3 - 2.25)) * 2 - 1
    retval[retval < -1.0] = -1.0
    retval[retval > 1.0] = 1.0

    return retval


# def normfunc(dmap):
#     return(dmap)

import torchvision.transforms as T

# #def opt_foo2(x, y):
#     a = torch.sin(x)
#     b = torch.cos(y)
#     return a + b


@torch.compile
def norm_dmap_min(dmap):
    dmap = ((dmap - 2.25) / (3.3 - 2.25)) * 2 - 1
    dmap[dmap < -1.0] = -1.0
    dmap[dmap > 1.0] = 1.0
    return dmap


class vggface2_to_512(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.vggface2 = inception_resnet_v1_fnet(pretrained="vggface2").eval()
        # self.upsample=T.Resize(size=(256, 256), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        # self.rcrop=T.RandomCrop(size=160)
        # self.ndm=norm_dmap_min

        # vggface2.cuda()
        # Features for target image.
        # target_images_for_id = target.unsqueeze(0).to(device).to(torch.float32)
        # target_images_for_id=((target_images_for_id/255.0)-0.5)*2 #need to rescale like so...

        # if target_images_for_id.shape[2] != 160:
        #    target_images_for_id = F.interpolate(target_images_for_id, size=(160, 160), mode='area')

        # with torch.no_grad():
        #    target_features = vggface2(target_images_for_id)

        # model = torchvision.models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1')
        # model.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)])
        # self.vgg_model=model

        self.freeze_vggface2_layers()
        # self.freeze_inet_layers()

    def freeze_vggface2_layers(self):
        for param in self.vggface2.parameters():
            param.requires_grad = False
        return self

    # def freeze_inet_layers(self):
    #     for param in self.vggface2.features.parameters():
    #         param.requires_grad = False
    #     return(self)

    def forward(self, dmap):
        with torch.no_grad():
            bsize = dmap.shape[0]
            # dmap=self.upsample(dmap)
            # dmap=self.rcrop(dmap)
            # dmap=self.ndm(dmap)
            dmap_f1 = self.vggface2(dmap[:, 0, :, :].unsqueeze(1).expand(bsize, 3, 160, 160))
            dmap_f2 = self.vggface2(dmap[:, 1, :, :].unsqueeze(1).expand(bsize, 3, 160, 160))
            dmap_f3 = self.vggface2(dmap[:, 2, :, :].unsqueeze(1).expand(bsize, 3, 160, 160))
            feature_vector = torch.cat((dmap_f1, dmap_f2, dmap_f3), dim=1)  # concatenate all output then feed into fc layers
            return feature_vector


# bigger model w more layers
class rwd_model_3dmap_vggface2_minimal(nn.Module):
    def __init__(self, nrs=128, n_hidden=1, h_layer_size=256, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(512 * 3, h_layer_size)

        assert n_hidden >= 1, "error you must have at least one hidden..."

        if n_hidden > 1:
            for n in range(1, n_hidden):
                setattr(self, f"fc{n+1}", nn.Linear(h_layer_size, h_layer_size))

        setattr(self, f"fc{n_hidden+1}", nn.Linear(h_layer_size, 256))

        self.total_hidden_layers = n_hidden

        self.reward_model_type = "rwd_model_3dmap_vggface2_minimal"
        self.projection_head = nn.Linear(256, 256)
        self.rwd_model_head = nn.Linear(256, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.nrs = nrs

        # to be used when we want to scale the reward
        self.affine_offset = nn.Parameter(torch.tensor(0.0))
        self.affine_scale = nn.Parameter(torch.tensor(1.0))

        self.affine_offset.requires_grad = False
        self.affine_scale.requires_grad = False

    def forward(self, dmap_fc_input):
        x = self.fc1(dmap_fc_input)
        x = self.relu(x)

        for k in range(self.total_hidden_layers):
            x = getattr(self, f"fc{k+2}")(x)
            x = self.relu(x)

        x = self.rwd_model_head(x)

        x = (x - self.affine_offset) / self.affine_scale
        return x

    def forward_projection_simclr(self, dmap_fc_input):
        x = self.fc1(dmap_fc_input)
        x = self.relu(x)
        for k in range(self.total_hidden_layers):
            x = getattr(self, f"fc{k+2}")(x)
            x = self.relu(x)
        x = self.projection_head(x)
        projection = x
        return projection

    def feature_embedding(self, dmap_fc_input):
        x = self.fc1(dmap_fc_input)
        x = self.relu(x)

        for k in range(self.total_hidden_layers):
            x = getattr(self, f"fc{k+2}")(x)
            x = self.relu(x)

        return x


# bigger model w more layers
class rwd_model_stylecode(nn.Module):
    def __init__(self, n_hidden=1, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        for n in range(n_hidden):
            setattr(self, f"fc{n+2}", nn.Linear(256, 256))
        self.total_hidden_layers = n_hidden
        self.rwd_model_head = nn.Linear(256, 1)
        self.reward_model_type = "rwd_model_stylecode"
        self.projection_head = nn.Linear(256, 256)

        # self.rwd_model_head=nn.Linear(256,1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        # to be used when we want to scale the reward
        self.affine_offset = nn.Parameter(torch.tensor(0.0))
        self.affine_scale = nn.Parameter(torch.tensor(1.0))

        self.affine_offset.requires_grad = False
        self.affine_scale.requires_grad = False

    def forward(self, w_features):
        x = self.fc1(w_features)
        x = self.relu(x)
        for k in range(self.total_hidden_layers):
            x = getattr(self, f"fc{k+2}")(x)
            x = self.relu(x)

        x = self.rwd_model_head(x)

        x = (x - self.affine_offset) / self.affine_scale
        return x

    def forward_projection_simclr(self, w_features):
        x = self.fc1(w_features)
        x = self.relu(x)
        for k in range(self.total_hidden_layers):
            x = getattr(self, f"fc{k+2}")(x)
            x = self.relu(x)

        x = self.projection_head(x)
        projection = x
        return projection


# using another id module...


# import id_loss_module_3DGIPO as id_loss_module_3DGIPO
# id_loss_face = id_loss_module_3DGIPO.IDLoss().cuda().eval()
# id_loss_face.extract_feats()


# feats=id_loss_face.extract_feats(F.interpolate(dmp_in, size=(112, 112), mode='nearest'))


# class rwd_2d_lmk_plus_vgg_triple_dmap(nn.Module):
#     def __init__(self,n_hidden=1,**kwargs):
#         super().__init__()

#         self.dmap_classifier=rwd_model_3dmap_vgg_minimal(n_hidden=n_hidden,**kwargs)

#         self.2d_lmks_98_classifier=rwd_model_2d_landmarks_98(n_hidden=n_hidden,**kwargs)

#         self.fc1=nn.Linear(256*2,256)

#         for n in range(n_hidden):
#             setattr(self, f"fc{n+2}", nn.Linear(256,256))
#         self.total_hidden_layers=n_hidden
#         self.rwd_model_head=nn.Linear(256,1)
#         self.reward_model_type='rwd_2d_lmk_plus_vgg_triple_dmap'
#         self.projection_head=nn.Linear(256,256)
#         self.tanh=nn.Tanh()
#         self.relu=nn.ReLU()

#         #to be used when we want to scale the reward
#         self.affine_offset=nn.Parameter(torch.tensor(0.0))
#         self.affine_scale=nn.Parameter(torch.tensor(1.0))

#         self.affine_offset.requires_grad=False
#         self.affine_scale.requires_grad=False

#     def forward(self, dmaps,lmks):

#         x_dmap=self.dmap_classifier(dmaps)


#         x = self.fc1(w_features)
#         x = self.relu(x)
#         for k in range(self.total_hidden_layers):
#             x = getattr(self, f"fc{k+2}")(x)
#             x = self.relu(x)

#         x = self.rwd_model_head(x)

#         x = (x - self.affine_offset)/self.affine_scale
#         return x

#     def forward_projection_simclr(self,w_features):
#         x = self.fc1(w_features)
#         x = self.relu(x)
#         for k in range(self.total_hidden_layers):
#             x = getattr(self, f"fc{k+2}")(x)
#             x = self.relu(x)

#         x = self.projection_head(x)
#         projection =x
#         return(projection)


class rwd_model_2d_landmarks_98(nn.Module):
    def __init__(self, n_hidden=1, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(2 * 98, 256)
        for n in range(n_hidden):
            setattr(self, f"fc{n+2}", nn.Linear(256, 256))
        self.total_hidden_layers = n_hidden
        self.rwd_model_head = nn.Linear(256, 1)
        self.reward_model_type = "rwd_model_2d_landmarks_98"
        self.projection_head = nn.Linear(256, 256)
        # self.projection_head_128=nn.Linear(256,128)
        # self.rwd_model_head=nn.Linear(256,1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        # to be used when we want to scale the reward
        self.affine_offset = nn.Parameter(torch.tensor(0.0))
        self.affine_scale = nn.Parameter(torch.tensor(1.0))

        self.affine_offset.requires_grad = False
        self.affine_scale.requires_grad = False

    def forward(self, w_features):
        x = self.fc1(w_features)
        x = self.relu(x)
        for k in range(self.total_hidden_layers):
            x = getattr(self, f"fc{k+2}")(x)
            x = self.relu(x)

        x = self.rwd_model_head(x)

        x = (x - self.affine_offset) / self.affine_scale
        return x

    def forward_projection_simclr(self, w_features):
        x = self.fc1(w_features)
        x = self.relu(x)
        for k in range(self.total_hidden_layers):
            x = getattr(self, f"fc{k+2}")(x)
            x = self.relu(x)

        x = self.projection_head(x)
        projection = x
        return projection

    # def forward_projection_128(self,w_features):
    #    x = self.fc1(w_features)
    #    x = self.relu(x)
    #    for k in range(self.total_hidden_layers):
    #       x = getattr(self, f"fc{k+2}")(x)
    #        x = self.relu(x)##

    #    x = self.projection_head_128(x)
    #    projection =x
    #    return(projection)


# 3 MLP for 2x98 lmks and then
# pass in 3x128 dim feature vec to this mlp


class rwd_model_2d_landmarks_98_for_triple(nn.Module):
    def __init__(self, n_hidden=1, act="relu", **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(2 * 98, 256)
        for n in range(n_hidden):
            setattr(self, f"fc{n+2}", nn.Linear(256, 256))
        self.total_hidden_layers = n_hidden
        self.rwd_model_head = nn.Linear(256, 1)
        self.reward_model_type = "rwd_model_2d_landmarks_98"
        self.projection_head = nn.Linear(256, 256)
        self.projection_head_128 = nn.Linear(256, 128)
        # self.rwd_model_head=nn.Linear(256,1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

        if act == "relu":
            self.act = self.relu
        if act == "tanh":
            self.act = self.tanh
        if act == "softplus":
            self.act = self.softplus

        # to be used when we want to scale the reward
        self.affine_offset = nn.Parameter(torch.tensor(0.0))
        self.affine_scale = nn.Parameter(torch.tensor(1.0))

        self.affine_offset.requires_grad = False
        self.affine_scale.requires_grad = False

    def forward(self, w_features):
        x = self.fc1(w_features)
        x = self.act(x)
        for k in range(self.total_hidden_layers):
            x = getattr(self, f"fc{k+2}")(x)
            x = self.act(x)

        x = self.rwd_model_head(x)

        x = (x - self.affine_offset) / self.affine_scale
        return x

    def forward_projection_simclr(self, w_features):
        x = self.fc1(w_features)
        x = self.act(x)
        for k in range(self.total_hidden_layers):
            x = getattr(self, f"fc{k+2}")(x)
            x = self.act(x)

        x = self.projection_head(x)
        projection = x
        return projection

    def forward_projection_128(self, w_features):
        x = self.fc1(w_features)
        x = self.act(x)
        for k in range(self.total_hidden_layers):
            x = getattr(self, f"fc{k+2}")(x)
            x = self.act(x)  ##

        x = self.projection_head_128(x)
        projection = x
        return projection


# 3 MLP for 2x98 lmks and then
# pass in 3x128 dim feature vec to this mlp


class rwd_model_2d_landmarks_98_triple(nn.Module):
    def __init__(self, n_hidden=1, n_hidden_single=1, act="relu", **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(128 * 3, 256)
        for n in range(n_hidden):
            setattr(self, f"fc{n+2}", nn.Linear(256, 256))
        self.total_hidden_layers = n_hidden
        self.rwd_model_head = nn.Linear(256, 1)
        self.reward_model_type = "rwd_model_2d_landmarks_98_triple"
        self.projection_head = nn.Linear(256, 256)
        # self.rwd_model_head=nn.Linear(256,1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

        if act == "relu":
            self.act = self.relu
        if act == "tanh":
            self.act = self.tanh
        if act == "softplus":
            self.act = self.softplus

        # to be used when we want to scale the reward
        self.affine_offset = nn.Parameter(torch.tensor(0.0))
        self.affine_scale = nn.Parameter(torch.tensor(1.0))

        self.affine_offset.requires_grad = False
        self.affine_scale.requires_grad = False

        self.first_lmk_head = rwd_model_2d_landmarks_98_for_triple(act=act, n_hidden=n_hidden_single)
        self.second_lmk_head = rwd_model_2d_landmarks_98_for_triple(act=act, n_hidden=n_hidden_single)
        self.third_lmk_head = rwd_model_2d_landmarks_98_for_triple(act=act, n_hidden=n_hidden_single)
        self.lhs_inverse_lmks = list(set(range(98)) - set(lhs_lmks))

        self.rhs_inverse_lmks = list(set(range(98)) - set(rhs_lmks))

    def forward(self, lmks):
        lmks_left = lmks[:, 0, :, :]
        lmks_right = lmks[:, 2, :, :]
        lmks_msk_left = torch.ones_like(lmks_left).clone()
        lmks_msk_right = torch.ones_like(lmks_right).clone()
        lmks_msk_left[:, :, self.lhs_inverse_lmks] = 0.0
        lmks_msk_right[:, :, self.rhs_inverse_lmks] = 0.0
        lmks_left_masked = lmks_left.masked_fill(lmks_msk_left == 0.0, 0.0)
        lmks_right_masked = lmks_right.masked_fill(lmks_msk_right == 0.0, 0.0)
        dmap_f1 = self.first_lmk_head.forward_projection_128(lmks_left_masked.view(-1, 2 * 98))
        dmap_f2 = self.second_lmk_head.forward_projection_128(lmks[:, 1, :, :].view(-1, 2 * 98))
        dmap_f3 = self.third_lmk_head.forward_projection_128(lmks_right_masked.view(-1, 2 * 98))

        feature_vector = torch.cat((dmap_f1, dmap_f2, dmap_f3), dim=1)  # concatenate all output then feed into fc layers

        x = self.fc1(feature_vector)
        x = self.act(x)
        for k in range(self.total_hidden_layers):
            x = getattr(self, f"fc{k+2}")(x)
            x = self.act(x)

        x = self.rwd_model_head(x)

        x = (x - self.affine_offset) / self.affine_scale
        return x

    def forward_projection_simclr(self, w_features):
        x = self.fc1(w_features)
        x = self.act(x)
        for k in range(self.total_hidden_layers):
            x = getattr(self, f"fc{k+2}")(x)
            x = self.act(x)

        x = self.projection_head(x)
        projection = x
        return projection
