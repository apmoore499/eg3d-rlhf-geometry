import argparse
import copy
import glob
import json
import os
import pickle
import random
import sys

# checking all unique combinations of flattened depth maps
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

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
import torchvision.transforms as T
from facenet_pytorch import InceptionResnetV1 as inception_resnet_v1_fnet
from IPython.core.debugger import set_trace
from pandas_ods_reader import read_ods
from pyexcel import get_book
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.autograd import Variable
from core_modules.data.aug_and_tforms.augtforms_dmap import (
    DepthMapPreprocessor,
    DepthMapTo160Transform,
    DepthMapTo224Transform,
    build_depthmap_preprocessors,
)


import random

import cv2
import numpy as np
import skimage.exposure as exposure
import torch
import numpy as np
import pywt
import torch

# --------------------------------------------------------------------------------------------
# Load necessary Pytorch packages
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Assuming 'image' is your input image tensor of shape (batch_size, 1, height, width)
# If your image is not in this shape, you'll need to reshape or unsqueeze it appropriately.

# Apply padding to keep the image size constant after convolution
# Use 'same' padding by calculating it based on the kernel size

# 'output' contains the result of the Laplace filter applied to the image


# image dim is B,C,W,H


def wavelet_decompose_reconstruct_extreme(image, level=1, detail_scale=2):
    # Perform wavelet decomposition
    coeffs = pywt.wavedec2(image, "haar", level=level)

    # Zero out the approximation coefficients for emphasis on details
    coeffs_high_freq = [coeffs[0]] + [(None, None, None) if i > 0 else coeff for i, coeff in enumerate(coeffs[1:])]

    # Ensure the first element (approximation) is replaced with zeros
    coeffs_high_freq[0] = np.zeros_like(coeffs_high_freq[0])

    # Amplify the detail coefficients
    for i in range(1, len(coeffs_high_freq)):
        if coeffs_high_freq[i] is not None:
            h, v, d = coeffs_high_freq[i]
            coeffs_high_freq[i] = (h * detail_scale, v * detail_scale, d * detail_scale)

    # Reconstruct the image using the modified coefficients
    high_freq_components = pywt.waverec2(coeffs_high_freq, "haar")
    return high_freq_components


class rgb_to_224_transform(nn.Module):
    def __init__(self, normalise=True):
        super().__init__()
        self.out_size = 224

        self.normalise = normalise

        self.norm_range = [0, 1]  # 0,1 for vgg

    def forward(self, dmap):
        # interpolate it....

        dmap = F.interpolate(dmap, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        # forward is in range -1,1 so convertt it...

        dmap = dmap / 2 + 0.5

        # dmap=dmap/dmap.max()

        return dmap


# to cmopute the feature vector for depth map, no gradients req
class vgg19_to_4096(nn.Module):
    def __init__(self, normalise_for_dmap, **kwargs):
        super().__init__()

        model = torchvision.models.vgg19(weights="VGG19_Weights.IMAGENET1K_V1")
        model.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)])
        self.vgg_model = model

        self.freeze_classifier_layers()
        self.freeze_inet_layers()

        self.embedding_size = 4096
        self.input_min = 0.0
        self.input_max = 1.0

        # self.dmap_to_224_transform=dmap_to_224_transform(normalise=normalise_for_dmap)

        self.dmap_to_224_transform = rgb_to_224_transform(normalise=True)

    def freeze_classifier_layers(self):
        for param in self.vgg_model.classifier.parameters():
            param.requires_grad = False
        return self

    def freeze_inet_layers(self):
        for param in self.vgg_model.features.parameters():
            param.requires_grad = False
        return self

    # def forward(self,dmap):
    #     bsize=dmap.shape[0]
    #     dmap_f1=self.vgg_model(dmap[:,0,:,:].unsqueeze(1).expand(bsize,3,224,224))
    #     dmap_f2=self.vgg_model(dmap[:,1,:,:].unsqueeze(1).expand(bsize,3,224,224))
    #     dmap_f3=self.vgg_model(dmap[:,2,:,:].unsqueeze(1).expand(bsize,3,224,224))
    #     feature_vector=torch.cat((dmap_f1,dmap_f2,dmap_f3),dim=1) #concatenate all output then feed into fc layers
    #     return feature_vector

    def forward(self, dmap):
        bsize = dmap.shape[0]
        n_maps = dmap.shape[1]

        retlist = []

        for n in range(n_maps):
            dmap_n = self.dmap_to_224_transform(dmap[:, n, :, :])  # will work for both rgb and dmap of size [B,C,H,W]
            # dmap_n=(dmap_n-self.input_min)/(d_max-self.input_max)

            dmap_f = self.vgg_model(dmap_n.expand(bsize, 3, 224, 224))
            retlist.append(dmap_f)  # featur rep
        # dmap=self.upsample(dmap)
        # dmap=self.rcrop(dmap)
        # dmap=self.ndm(dmap)
        # dmap_f1=self.vggface2(dmap[:,0,:,:].unsqueeze(1).expand(bsize,3,160,160))
        # dmap_f2=self.vggface2(dmap[:,1,:,:].unsqueeze(1).expand(bsize,3,160,160))
        # dmap_f3=self.vggface2(dmap[:,2,:,:].unsqueeze(1).expand(bsize,3,160,160))
        feature_vector = torch.cat(retlist, dim=1)  # concatenate all output then feed into fc layers
        return feature_vector


class vggface2_to_512(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.vggface2 = inception_resnet_v1_fnet(pretrained="vggface2").eval()
        # self.upsample=T.Resize(size=(256, 256), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        # self.rcrop=T.RandomCrop(size=160)
        # self.ndm=norm_dmap_min
        self.embedding_size = 512
        self.input_lims = [-1.0, 1.0]

        self.dmap_to_160_transform = DepthMapTo160Transform(normalize_range=True, invert=True)

        # vggface2.cuda()
        # Features for target image.
        # target_images_for_id = target.unsqueeze(0).to(device).to(torch.float32)
        # target_images_for_id=((target_images_for_id/255.0)-0.5)*2 #need to rescale like so...

        # if target_images_for_id.shape[2] != 160:
        #    target_images_for_id = F.interpolate(target_images_for_id, size=(160, 160), mode='area')

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
        bsize = dmap.shape[0]
        n_maps = dmap.shape[1]

        retlist = []

        for n in range(n_maps):
            dmap_n = self.dmap_to_160_transform(dmap[:, n, :, :])
            dmap_f = self.vggface2(dmap_n.expand(bsize, 3, 160, 160))

            retlist.append(dmap_f)
        # dmap=self.upsample(dmap)
        # dmap=self.rcrop(dmap)
        # dmap=self.ndm(dmap)
        # dmap_f1=self.vggface2(dmap[:,0,:,:].unsqueeze(1).expand(bsize,3,160,160))
        # dmap_f2=self.vggface2(dmap[:,1,:,:].unsqueeze(1).expand(bsize,3,160,160))
        # dmap_f3=self.vggface2(dmap[:,2,:,:].unsqueeze(1).expand(bsize,3,160,160))
        feature_vector = torch.cat(retlist, dim=1)  # concatenate all output then feed into fc layers
        return feature_vector

        #     n_maps=dmap.shape[1]

        #     retlist=[]

        #     for n in range(n_maps):
        #         #set_trace()
        #         dmap_n=self.dmap_to_160_transform(dmap[:,n,:,:]) #will work for both rgb and dmap of size [B,C,H,W]
        #         #dmap_n=(dmap_n-self.input_min)/(d_max-self.input_max)

        #         dmap_f=self.vgg_model(dmap_n.expand(bsize,3,224,224))
        #         retlist.append(dmap_f) #featur rep
        #     #dmap=self.upsample(dmap)
        #     #dmap=self.rcrop(dmap)
        #     #dmap=self.ndm(dmap)
        #     #dmap_f1=self.vggface2(dmap[:,0,:,:].unsqueeze(1).expand(bsize,3,160,160))
        #     #dmap_f2=self.vggface2(dmap[:,1,:,:].unsqueeze(1).expand(bsize,3,160,160))
        #     #dmap_f3=self.vggface2(dmap[:,2,:,:].unsqueeze(1).expand(bsize,3,160,160))
        #     feature_vector=torch.cat(retlist,dim=1) #concatenate all output then feed into fc layers
        #     return feature_vector


# https://pytorch.org/hub/pytorch_vision_resnet/

# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)


# Function to be called by the hook
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


from torchvision import transforms


class resnet50_to_2048_dmap(nn.Module):
    def __init__(self, transform_stats_to_resnet=False, **kwargs):
        super().__init__()

        self.resnet50 = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
        self.maps_transforms = kwargs["maps_transforms"]
        # self.upsample=T.Resize(size=(256, 256), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        # self.rcrop=T.RandomCrop(size=160)
        # self.ndm=norm_dmap_min
        self.embedding_size = 2048
        self.input_lims = [-1.0, 1.0]

        self.dmap_preprocessors = build_depthmap_preprocessors(
            self.maps_transforms,
            out_size=224,
            normalize_range=True,
            invert=True,
            lp_scale=5.0,
            hp_scale=1.0,
        ) or [DepthMapTo224Transform(normalize_range=True, invert=True, lp_scale=5.0, hp_scale=1.0)]

        # Define a hook function for the avgpool layer
        def avgpool_hook(module, input, output):
            # print("AvgPool hook triggered")
            # Optionally, store the output activations for later use
            self.avgpool_activations = output

        # Register the hook on the avgpool layer
        self.resnet50.avgpool.register_forward_hook(avgpool_hook)

        self.tf_std = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.activation = {}
        # self.resnet50.avgpool.register_forward_hook(get_activation('avgpool'))
        self.transform_stats_to_resnet = transform_stats_to_resnet

        self.maps_transforms = kwargs["maps_transforms"]

        self.freeze_resnet_layers()

    def freeze_resnet_layers(self):
        for param in self.resnet50.parameters():
            param.requires_grad = False
        return self

    def forward(self, dmap):
        self.avgpool_activations = None
        bsize = dmap.shape[0]
        n_maps = dmap.shape[1]

        retlist = []

        for n in range(n_maps):
            hipass_flags = self.maps_transforms.hipass
            laplace_flags = self.maps_transforms.laplace
            crop_flags = self.maps_transforms.normalise_sides_crop
            lowpass_flags = self.maps_transforms.run_lowpass_sides

            hipass = hipass_flags[n if n < len(hipass_flags) else -1]
            laplace = laplace_flags[n if n < len(laplace_flags) else -1]
            normalise_sides_crop = crop_flags[n if n < len(crop_flags) else -1]
            run_lowpass_sides = lowpass_flags[n if n < len(lowpass_flags) else -1]
            preproc = self.dmap_preprocessors[n if n < len(self.dmap_preprocessors) else -1]
            dmap_n = preproc(
                dmap[:, n, :, :],
                hipass=hipass,
                laplace=laplace,
                Llp=5.0,
                Lhp=1.0,
                normalise_sides_crop=normalise_sides_crop,
                run_lowpass_sides=run_lowpass_sides,
            ).expand(bsize, 3, 224, 224)

            if self.transform_stats_to_resnet:
                dmap_n = self.tf_std(dmap_n)
            dmap_f = self.resnet50(dmap_n)

            # retlist.append(dmap_f)

            retlist.append(self.avgpool_activations if self.avgpool_activations is not None else dmap_f)

        feature_vector = torch.cat(retlist, dim=1)  # concatenate all output then feed into fc layers
        if feature_vector.dim() > 2:
            return feature_vector.squeeze(-2, -1)
        return feature_vector


class resnet152_to_1000_dmap(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.resnet152 = torch.hub.load("pytorch/vision:v0.10.0", "resnet152", pretrained=True)
        self.embedding_size = 512
        self.input_lims = [-1.0, 1.0]

        self.dmap_to_160_transform = DepthMapTo160Transform(normalize_range=True, invert=True)

        self.freeze_resnet152_layers()

    def freeze_resnet152_layers(self):
        for param in self.resnet152.parameters():
            param.requires_grad = False
        return self

    def forward(self, dmap):
        bsize = dmap.shape[0]
        n_maps = dmap.shape[1]

        retlist = []

        for n in range(n_maps):
            dmap_n = self.dmap_to_160_transform(dmap[:, n, :, :])
            dmap_f = self.resnet152(dmap_n.expand(bsize, 3, 160, 160))

            retlist.append(dmap_f)
        feature_vector = torch.cat(retlist, dim=1)  # concatenate all output then feed into fc layers
        return feature_vector


class vggface2_to_512_rgb_dmap(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # takes in rgb and a dmap in canonical view
        self.vggface2 = inception_resnet_v1_fnet(pretrained="vggface2").eval()
        self.embedding_size = 512
        self.input_lims = [-1.0, 1.0]

        self.dmap_to_160_transform = DepthMapTo160Transform(normalize_range=True, invert=True)

        self.freeze_vggface2_layers()
        # self.freeze_inet_layers()

    def freeze_vggface2_layers(self):
        for param in self.vggface2.parameters():
            param.requires_grad = False
        return self

    def forward(self, rgb_dmap):
        bsize = rgb_dmap.shape[0]
        n_maps = rgb_dmap.shape[1]

        retlist = []

        assert n_maps == 1, "error only implement thise for rgb dmap for canonical view, no multi dmaps"

        for n in range(n_maps):
            rgb = rgb_dmap[:, :, :3, :, :]

            rgb_n = self.dmap_to_160_transform(rgb.squeeze(1))
            rgb_f = self.vggface2(rgb_n.expand(bsize, 3, 160, 160))

            dmap = rgb_dmap[:, :, 3, :, :]

            dmap_n = self.dmap_to_160_transform(dmap)
            dmap_f = self.vggface2(dmap_n.expand(bsize, 3, 160, 160))

            dmap_f = torch.cat((rgb_f, dmap_f), dim=1)

            retlist.append(dmap_f)
        feature_vector = torch.cat(retlist, dim=1)  # concatenate all output then feed into fc layers
        return feature_vector
