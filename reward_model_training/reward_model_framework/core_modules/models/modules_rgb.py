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
from pandas_ods_reader import read_ods
from pyexcel import get_book
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.autograd import Variable

# --------------------------------------------------------------------------------------------
# Load necessary Pytorch packages
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# bigger model w more layers
class scalar_reward_rgb_triple(nn.Module):
    def __init__(self, nrs=128, input_embedding_size=512, n_hidden=1, h_layer_size=256, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(input_embedding_size * 3, h_layer_size)

        assert n_hidden >= 1, "error you must have at least one hidden..."

        if n_hidden > 1:
            for n in range(1, n_hidden):
                setattr(self, f"fc{n + 1}", nn.Linear(h_layer_size, h_layer_size))

        setattr(self, f"fc{n_hidden + 1}", nn.Linear(h_layer_size, 256))

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
            x = getattr(self, f"fc{k + 2}")(x)
            x = self.relu(x)

        x = self.rwd_model_head(x)

        x = (x - self.affine_offset) / self.affine_scale
        return x

    def forward_projection_simclr(self, dmap_fc_input):
        x = self.fc1(dmap_fc_input)
        x = self.relu(x)
        for k in range(self.total_hidden_layers):
            x = getattr(self, f"fc{k + 2}")(x)
            x = self.relu(x)
        x = self.projection_head(x)
        projection = x
        return projection

    def feature_embedding(self, dmap_fc_input):
        x = self.fc1(dmap_fc_input)
        x = self.relu(x)

        for k in range(self.total_hidden_layers):
            x = getattr(self, f"fc{k + 2}")(x)
            x = self.relu(x)

        return x


# bigger model w more layers
class scalar_reward_rgb_single(nn.Module):
    def __init__(self, nrs=128, input_embedding_size=4096, n_hidden=1, **kwargs):
        super().__init__()
        print(input_embedding_size)
        print(type(input_embedding_size))
        self.fc1 = nn.Linear(input_embedding_size, 256)

        for n in range(n_hidden):
            setattr(self, f"fc{n + 2}", nn.Linear(256, 256))

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
            x = getattr(self, f"fc{k + 2}")(x)
            x = self.relu(x)

        x = self.rwd_model_head(x)

        x = (x - self.affine_offset) / self.affine_scale
        return x

    def forward_projection_simclr(self, dmap_fc_input):
        x = self.fc1(dmap_fc_input)
        x = self.relu(x)
        for k in range(self.total_hidden_layers):
            x = getattr(self, f"fc{k + 2}")(x)
            x = self.relu(x)
        x = self.projection_head(x)
        projection = x
        return projection
