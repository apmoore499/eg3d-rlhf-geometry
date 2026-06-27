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

from core_modules.models.base import UniversalRWDModel


class DepthMap(UniversalRWDModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_to_global_feature_vec(self, x):
        if len(os.listdir(self.model_example_dir)) == 0:
            self.save_model_example_input(x)
        feature_vec = self.MLP(x)
        return feature_vec

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            if self.external is not None:
                self.external = torch.compile(self.external)
            self.MLP = torch.compile(self.MLP)
            self.scalar_rwd_head = torch.compile(self.scalar_rwd_head)
            self.scalar_rwd_head_BT = torch.compile(self.scalar_rwd_head_BT)
            self.scalar_rwd_head_pairs = torch.compile(self.scalar_rwd_head_pairs)

    def on_save_checkpoint(self, checkpoint):
        external_keys = [k for k in checkpoint["state_dict"].keys() if "external." in k]
        for key in external_keys:
            if key in checkpoint["state_dict"]:
                del checkpoint["state_dict"][key]
        return checkpoint

    # def return_swa_weights(self):
    #     if self.swa_model is None:
    #         log.info("no SWA model used, returning None")
    #         return None
    #     swa_sd = copy.deepcopy(self.swa_model)
    #     external_keys = [k for k in swa_sd.keys() if "external." in k]
    #     for key in external_keys:
    #         if key in swa_sd:
    #             del swa_sd[key]
    #     return swa_sd

    def load_model_state_dict(self, checkpoint) -> None:
        self.remove_external()
        assert self.lightning_module is not None
        self.lightning_module.load_state_dict(checkpoint["state_dict"])
        self.set_external()


__all__ = [
    "DepthMap",
]
