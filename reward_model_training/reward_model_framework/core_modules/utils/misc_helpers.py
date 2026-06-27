import os
import random

import ml_collections as mlc
import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split

# general utils not doing anything important
# --------------------------------------


def print_break(message=""):
    # print("\n\n")
    # print(message)
    # retval = "#---------------------------------------------------------\n\n"
    # print(retval)
    return


# read in yaml as ml_collections configdict


def read_yaml_as_mlconfigdict(yaml_fn):
    with open(yaml_fn) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = mlc.ConfigDict(config)
    return config


from IPython.core.debugger import set_trace


def split_dset_partitions(data, n_train, n_val, n_test):
    train, test_val = train_test_split(data, test_size=n_val + n_test, train_size=n_train, random_state=42)
    val, test = train_test_split(
        test_val,
        test_size=n_test / (n_test + n_val),
        train_size=n_val / (n_test + n_val),
        random_state=42,
    )

    return (train, val, test)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
