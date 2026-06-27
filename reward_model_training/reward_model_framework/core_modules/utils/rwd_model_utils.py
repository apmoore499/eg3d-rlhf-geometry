import itertools
import os
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib

matplotlib.use("agg")

# import data_rwd_training.collate_data as dc
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import PIL
import seaborn as sns
import torch
import torchvision
import tqdm
from core_modules.data import io_geometry_utils as io_utils

# from torch import multiprocessing
# from tsnecuda import TSNE as TSNE_c
import wandb

# from data_rwd_training import rld
# import pylogger_c
# import rich_utils
# from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, ListConfig

# Seeds reserved as "good" demos (used as positive anchors during training).
# Final-metrics reporting computes accuracies both with and without these.
GOODSEEDS = [i for i in range(100000, 101000)]

# Centralized helpers (aliases keep legacy names intact)
create_pt_fn = io_utils.create_pt_fn
get_canonical_dmap_cams = io_utils.get_canonical_dmap_cams
imd_to_xyz_with_radius_cutoff = io_utils.imd_to_xyz_with_radius_cutoff
return_lmks_mask = io_utils.return_lmks_mask_aw98
from PIL import Image

# import utils.small_data_tools as sdt


from core_modules.utils import pylogger_c, rich_utils



# root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)


# ----------------------------------------------


log = pylogger_c.RankedLogger(__name__, rank_zero_only=True)

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
REWARD_MODEL_TRAINING_DIR = PROJECT_ROOT / "reward_model_training"
PRECOMPUTED_DIR = REWARD_MODEL_TRAINING_DIR / "precomputed"
COMPRESSION_TMP_DIR = REWARD_MODEL_TRAINING_DIR / "compression_dl" / "tmp"


def _precomputed_path(filename: str) -> str:
    return str(PRECOMPUTED_DIR / filename)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(f"Metric value not found! <metric_name={metric_name}>\nMake sure metric name logged in LightningModule is correct!\nMake sure `optimized_metric` name in `hparams_search` config is correct!")

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


import pathlib


def create_img_of_ranked_meshes(df, seed_func, ddir_func, which_seeds="best", n_meshes=10):
    if which_seeds == "best":
        # sort the dataframe by the mean column in descending order
        sorted_joined_all = df.sort_values(by="rwd_val", ascending=False)

    elif which_seeds == "worst":
        sorted_joined_all = df.sort_values(by="rwd_val", ascending=True)

    # extract the top n_meshes seeds and loss values
    sel_seeds = sorted_joined_all["seed"].head(n_meshes).tolist()
    sel_seeds = [int(i) for i in sel_seeds]
    # meshdir = "/path/to/eg3d-rlhf-geometry/reward_model_training/notebooks/legacy/03122025_98lmks_fix/visualised_meshes"
    meshdir = "/home/user/Documents/eg3dredo_data/visualisations"
    meshdir = pathlib.Path(meshdir)
    seedmeshes = [meshdir.joinpath(f"mesh_cat_s_{seed_func(s)}.jpg") for s in sel_seeds]
    pics = []
    for s in sel_seeds:
        # Prefer PT-loaded canonical RGB to avoid jpg dependency
        try:
            rgb = assemble_single_rgb(seed_func(s), ddir_func(s)).squeeze(0).squeeze(0)  # 3,H,W in [-1,1]
            rgb = ((rgb + 1.0) / 2.0).clamp(0, 1)  # to [0,1]
            arr = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pics.append(Image.fromarray(arr))
        except Exception:
            pics.append(os.path.join(ddir_func(s), f"triple_rgb_s_{seed_func(s)}_1.jpg"))
    overall_images = plot_overall_images(seedmeshes, pics)
    grid_img = torchvision.utils.make_grid(overall_images, nrow=10)

    return grid_img


# trainer.datamodule.data_train.ddir_func


def plot_tsne_colour_rwd(X_embedded, rwds, subset_bins=None, top_bottom_middle_ten=False, uselims=True):
    df = pd.DataFrame()
    df["y"] = rwds  # torch.tensor(rwds).flatten().detach().cpu().numpy()
    df.head()
    df["comp-1"] = X_embedded[:, 0]
    df["comp-2"] = X_embedded[:, 1]
    n_bins = 10
    df["rwd_bins"] = pd.qcut(df["y"], q=n_bins, labels=False)
    mean_rwd = df.groupby("rwd_bins").mean().y
    rwds_labels = [f"mean_rwd: {i:.3f}" for i in mean_rwd]
    df["mean_rwd"] = df["rwd_bins"].apply(lambda x: rwds_labels[x])
    sns.set(rc={"figure.figsize": (15.0, 15.0)})

    df = df.sort_values(by="rwd_bins", ascending=True)

    if top_bottom_middle_ten:
        bottom = df["y"].sort_values(ascending=True)[:10]
        top = df["y"].sort_values(ascending=False)[:10]

        mid_point = int(df["y"].shape[0] / 2)
        middle = df["y"].sort_values(ascending=True)[mid_point - 5 : mid_point + 5]

        all_index = list(set(bottom.index).union(set(top.index)).union(set(middle.index)))

        if len(all_index) != 30:
            log.warning("Unable to return top/bottom/middle 10 selections; falling back to empty figure.")
            fig = plt.figure()
            fig.set_size_inches(15.0, 15.0)  # You can set the size of the figure (width, height in inches)

            # Optionally, you can set other properties of the figure, like background color
            fig.patch.set_facecolor("white")

            return fig
        else:
            df["rwd_bins_str"] = ""

            df.loc[bottom.index, "rwd_bins_str"] = "bottom_10"
            df.loc[top.index, "rwd_bins_str"] = "top_10"
            df.loc[middle.index, "rwd_bins_str"] = "middle_10"
            df = df[df.index.isin(all_index)]
            df.mean_rwd = df.rwd_bins

            ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.rwd_bins_str.tolist(), palette=sns.color_palette("hls", 3), data=df).set(title="Rewards T-SNE projection, top, bottom, middle 10")

            if uselims:
                ax[0].axes.set_xlim([-40, 40])
                ax[0].axes.set_ylim([-40, 40])

            fig = ax[0].get_figure()

            return fig

    if subset_bins is not None:
        # subset_bins should be which bins to take, ie [0,5,9]
        subset_bins = [int(s) for s in subset_bins]
        df = df[df.rwd_bins.isin(subset_bins)]

        ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.mean_rwd.tolist(), palette=sns.color_palette("hls", n_bins), data=df).set(title="Rewards T-SNE projection, subset to bins 0,5,9")

    else:
        ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.mean_rwd.tolist(), palette=sns.color_palette("hls", n_bins), data=df).set(title="Rewards T-SNE projection, 10 bins based on decile")

    if uselims:
        ax[0].axes.set_xlim([-40, 40])
        ax[0].axes.set_ylim([-40, 40])

    figure = ax[0].get_figure()
    return figure


def log_best_worst_meshes(df, ddir_func, seed_func, epoch, save_dir, n_meshes=10, remove_good=False, using_wandb=False, log_thumbnail_only=True, exp="val", comparison_type="", dset_version="", thumbnail_size=10000):
    rg_str = f"remove_good_{remove_good}"

    if remove_good:
        df = df[~df.seed.astype(int).isin(GOODSEEDS)]

    # log best ones
    grid_img = create_img_of_ranked_meshes(df, which_seeds="best", n_meshes=n_meshes, seed_func=seed_func, ddir_func=ddir_func)
    out_fn = os.path.join(save_dir, f"{exp}_top_10_mesh_{epoch}_{rg_str}_exp_{exp}.jpg")
    PIL.Image.fromarray(grid_img.numpy().transpose(1, 2, 0)).save(out_fn, quality=95)

    if using_wandb:
        im = PIL.Image.open(out_fn)

        if not log_thumbnail_only:
            images = wandb.Image(im, mode="RGB", caption="Top 10 Mesh")
            wandb.log({f"{exp}/dset_{dset_version}/{comparison_type}/top_10_mesh_exp_{comparison_type}": images, "epoch": epoch})

        im.thumbnail(size=(thumbnail_size, thumbnail_size))

        images = wandb.Image(im, mode="RGB", caption="Top 10 Mesh (Thumbnail)")

        wandb.log({f"{exp}/dset_{dset_version}/{comparison_type}/top_10_mesh_thumb_{rg_str}_exp_{comparison_type}": images, "epoch": epoch})

        os.remove(out_fn)

    # log worst ones
    grid_img = create_img_of_ranked_meshes(df, seed_func=seed_func, ddir_func=ddir_func, which_seeds="worst", n_meshes=n_meshes)
    out_fn = os.path.join(save_dir, f"{exp}_bottom_10_mesh_{epoch}_{rg_str}_exp_{exp}.jpg")
    PIL.Image.fromarray(grid_img.numpy().transpose(1, 2, 0)).save(out_fn, quality=95)

    if using_wandb:
        im = PIL.Image.open(out_fn)

        if not log_thumbnail_only:
            images = wandb.Image(im, mode="RGB", caption="Bottom 10 Mesh")
            wandb.log({f"{exp}/dset_{dset_version}/{comparison_type}/bottom_10_mesh_exp_{comparison_type}": images, "epoch": epoch})

        im.thumbnail(size=(thumbnail_size, thumbnail_size))

        images = wandb.Image(im, mode="RGB", caption="Bottom 10 Mesh (Thumbnail)")
        wandb.log({f"{exp}/dset_{dset_version}/{comparison_type}/bottom_10_mesh_thumb_{rg_str}_exp_{comparison_type}": images, "epoch": epoch})

        os.remove(out_fn)


def plot_overall_images(seedmeshes, pics):
    overall_images = []
    max_w = 0
    max_h = 0

    for s, p in zip(seedmeshes, pics):
        # open and resize the images
        try:
            img1 = Image.open(s)  # .resize((200, 200))
        except:  # noqa: E722, W0702
            # meshdir = "/path/to/eg3d-rlhf-geometry/reward_model_training/notebooks/legacy/03122025_98lmks_fix/visualised_meshes"
            meshdir = "/home/user/Documents/eg3dredo_data/visualisations"
            gg = list(pathlib.Path(meshdir).glob("mesh_cat_s_*.jpg"))
            # gg = #glob.glob("/path/to/eg3d-rlhf-geometry/000_RLHF_AM/rlhf_meshes/*/mesh_cat_s_*.jpg")
            img1 = Image.open(gg[0])  # .resize((200, 200))
            iii = np.array(img1)
            iii = np.ones_like(iii) * 255
            img1 = Image.fromarray(iii)

            # img1 = Image.open(p)#.resize((200, 200))

        if isinstance(p, (str, pathlib.Path)):
            img2 = Image.open(p)  # .resize((200, 300))
        elif isinstance(p, Image.Image):
            img2 = p
        elif torch.is_tensor(p):
            arr = p.detach().cpu()
            if arr.ndim == 4:
                arr = arr.squeeze(0)
            if arr.shape[0] == 3:
                arr = arr.permute(1, 2, 0)
            img2 = Image.fromarray(arr.numpy().astype(np.uint8))
        elif isinstance(p, np.ndarray):
            img2 = Image.fromarray(p)
        else:
            raise TypeError(f"Unsupported pic type: {type(p)}")

        # get the dimensions of the images
        w1, h1 = img1.size
        w2, h2 = img2.size

        # compute the difference in height
        dh = abs(h1 - h2)

        # create a new image with the maximum width and the sum of the heights
        new_img = Image.new("RGB", (max(w1, w2), h1 + h2), (255, 255, 255))

        # paste the first image at the top
        new_img.paste(img1, (0, 0))

        # paste the second image at the bottom, with padding if necessary
        if h1 > h2:
            new_img.paste(img2, (0, h1))
        else:
            new_img.paste(img2, (0, h1 - dh))

        max_w = max(max_w, new_img.size[0])
        max_h = max(max_h, new_img.size[1])
        overall_images.append(new_img)

    # pad each composite to common size so cat succeeds
    padded_tensors = []
    for img in overall_images:
        if img.size != (max_w, max_h):
            padded = Image.new("RGB", (max_w, max_h), (255, 255, 255))
            padded.paste(img, (0, 0))
            img = padded
        tensor = torch.from_numpy(np.array(img)).unsqueeze(0)
        padded_tensors.append(tensor)

    overall_images = torch.cat(padded_tensors, 0).permute(0, 3, 1, 2)

    return overall_images


# mlflow.autolog()

# python train.py --multirun data.dset_dict.n_point_samples_per_pcd_batch=2048,4096,8192 optimizer.lr=0.001,0.005,0.0005 -training.n_epochs=8
# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/
# https://hydra.cc/docs/advanced/override_grammar/basic/
# https://dagshub.com/blog/best-8-experiment-tracking-tools-for-machine-learning-2023/

# https://medium.com/optuna/easy-hyperparameter-management-with-hydra-mlflow-and-optuna-783730700e7d

# https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/


# code from here:
# https://medium.com/optuna/easy-hyperparameter-management-with-hydra-mlflow-and-optuna-783730700e7d
def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f"{parent_name}.{k}", v)
            else:
                mlflow.log_param(f"{parent_name}.{k}", v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f"{parent_name}.{i}", v)


# mlflow.set_experiment("mlflow-experiment")


def log_params_from_omegaconf_dict_wandb(params):
    for param_name, element in params.items():
        _explore_recursive_wandb(param_name, element)


def _explore_recursive_wandb(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive_wandb(f"{parent_name}.{k}", v)
            else:
                wandb.log({f"{parent_name}.{k}": v})

    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            wandb.log({f"{parent_name}.{i}": v})


def reorder_pair_by_idx(pair, idx):
    out_tuple = (pair[idx[0]], pair[idx[1]])
    return out_tuple


def get_rand_reordered_pair(p=0.5):
    sel_p = np.random.rand()
    in_tuple = (0, 1)
    if sel_p > p:
        out_tuple = (in_tuple[1], in_tuple[0])
    else:
        out_tuple = in_tuple
    return out_tuple


# Open an Image
# get the permutation
# Open an Image
import copy
import glob
import numbers
import sys

import cv2
import matplotlib.image as mpimg
import open3d as o3d
import torch_geometric
import torchvision.transforms.v2 as v2
import trimesh
import autoroot  # noqa: F401

# sys.path.append("/path/to/eg3d-rlhf-geometry/eg3d")


from pandas_ods_reader import read_ods
from PIL import ImageDraw, ImageFont, ImageOps
from eg3d.training.volumetric_rendering.ray_sampler import RaySampler

# from .misc_helpers import *
# from .rlhf_data_utils import *
# from .small_data_tools import *


# from data_loading import *

idx_for_grey = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
}


seed_letter_concordance = {
    "A": "seed1",
    "B": "seed2",
    "C": "seed3",
    "D": "seed4",
    "E": "seed5",
    "F": "seed6",
}


def add_alpha_channel_to_im_as_np_array(iic):
    iic = copy.deepcopy(iic)
    iic_tp = np.dstack((iic, np.zeros((iic.shape[0], iic.shape[1]))))
    iic_tp[:, :, 3] = 225
    iic_tpi = iic_tp.astype(np.uint8)
    return iic_tpi


def assemble_triple_rgb(seed, data_dir):
    seed_targets = []
    read_ims = []
    for k in range(3):
        target_fn = os.path.join(data_dir, "triple_rgb_s_" + str(seed) + f"_{k}.jpg")
        seed_targets.append(target_fn)
        read_im = mpimg.imread(target_fn)
        read_ims.append(read_im)

    im_array = np.hstack(read_ims)
    return im_array


def combine_vertically(im1, im2):
    im1_height = im1.shape[0]
    im2_height = im2.shape[0]

    if im1_height > im2_height:
        im2 = im2[:im1_height, :, :]
    elif im2_height > im1_height:
        im1 = im1[:im2_height, :, :]

    combined_im = np.vstack([im1, im2])
    return combined_im


def compose_vertical_from_np(mesh_file, image_file):
    im1 = Image.fromarray(mesh_file)
    im2 = Image.fromarray(image_file)
    mesh_file_width, mesh_file_height = im1.size
    image_file_width, image_file_height = im2.size

    ratio_for_image_file_height = mesh_file_height / image_file_height
    im2 = im2.resize(
        (
            int(image_file_width * ratio_for_image_file_height),
            int(image_file_height * ratio_for_image_file_height),
        )
    )

    im3 = Image.new("RGB", (max(im1.width, im2.width), im1.height + im2.height))

    im3.paste(im1, (0, 0))
    starting_position = int(im3.width / 2 - im2.width / 2)
    im3.paste(im2, (starting_position, im1.height))
    return im3


def convert_seed_to_path(seed):
    if seed is None or str(seed).lower() == "None" or np.isnan(seed):
        return None
    seed = int(seed)
    return os.path.join(composed_dir, "composed_s_" + str(seed) + ".jpg")


def combine_image_stack(iic):
    res_4k = {
        "height": 2160,
        "width": 3840,
    }

    iic_height = iic.shape[0]
    iic_width = iic.shape[1]
    new_height = res_4k["height"]
    ratio = new_height / iic_height
    new_width = int(iic_width * ratio)

    new_width, new_height = get_new_dims(iic, res_4k["height"])
    iic = PIL.Image.fromarray(iic).resize((new_width, new_height))

    return iic


def combine_row(row):
    # row=row[['seed1','seed2','seed3','seed4','seed5','seed6','seed7']]
    read_ims = read_in_row(row)
    read_ims = [put_red_border_around_img(ri) for ri in read_ims]
    list_of_seeds = ["A", "B", "C", "D", "E", "F", "G"]
    seed_names = [list_of_seeds[i] for i in range(len(read_ims))]
    read_ims = [text_on_image(ri, sn) for ri, sn in zip(read_ims, seed_names)]

    list_of_ims = [copy.deepcopy(read_ims[0]) for i in range(6)]
    for l in list_of_ims:
        l.fill(0)

    for i in range(len(read_ims)):
        list_of_ims[i] = read_ims[i]

    extr = extract_ranks(row)

    for letter in extr:
        idx = idx_for_grey[letter]
        list_of_ims[idx] = return_greyed_image(list_of_ims[idx])

    for k, i in enumerate(list_of_ims):
        if i.shape[2] != 4:
            i = add_alpha_channel_to_im_as_np_array(i)
            list_of_ims[k] = i

    combined_h1 = np.hstack(list_of_ims[:2])
    combined_h2 = np.hstack(list_of_ims[2:4])
    combined_h3 = np.hstack(list_of_ims[4:])

    combined_v = np.vstack([combined_h1, combined_h2, combined_h3])

    return combined_v


def fix_mesh(crop_mesh):
    tmesh = trimesh.Trimesh(vertices=np.asarray(crop_mesh.vertices), faces=crop_mesh.triangles)
    trimesh.repair.fix_inversion(tmesh)
    tmd = torch_geometric.utils.from_trimesh(tmesh)
    return tmd


def convert_pcd_to_mesh_using_rolling_ball(pcd):
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radii = [0.005, 0.01, 0.02, 0.04, 0.1, 0.4, 0.001]
    pcd.estimate_normals()
    mesh_ball = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    # o3d.visualization.draw_geometries([mesh_ball])


def convert_dmaps_to_pcd_for_checking_rwd_dist():
    # save out..............
    fn_list_dict = {
        "rndm_truncate_2.0": _precomputed_path("pretrained_pkl_dmaps_1000_128_rndm_noise_t2p0_dmaps_condensed.pt"),
        "const_truncate0.25": _precomputed_path("pretrained_pkl_dmaps_1000_128_const_noise_t025_dmaps_condensed.pt"),
        "rwds_unseen": _precomputed_path("pretrained_pkl_dmaps_unseen_200k_to_300k_128_dmaps_condensed.pt"),
        "seen": _precomputed_path("pretrained_pkl_dmaps_1000_128_const_noise_dmaps_condensed.pt"),
    }

    pcd_processed = {}

    for k in fn_list_dict.keys():
        rnd = torch.load(fn_list_dict[k])
        dmp_list = [r[1] for r in rnd]
        list_of_pcd_for_testing = []

        for dmp in tqdm.tqdm(dmp_list):
            rs = RaySampler()
            ray_origins, ray_directions = rs(cam2world_matrix, intrinsics, neural_rendering_resolution)
            pcd_points, img = imd_to_xyz_with_radius_cutoff(
                image_depth=dmp,
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                neural_rendering_resolution=neural_rendering_resolution,
            )
            indices = torch.randperm(pcd_points.shape[1])[:2048]
            list_of_pcd_for_testing.append(pcd_points[:, indices, :].view(1, 3, 2048).cpu())

        list_of_pcd_for_testing = torch.cat(list_of_pcd_for_testing, dim=0)
        pcd_processed[k] = list_of_pcd_for_testing

    fnld = fn_list_dict

    new_names_for_pcd = {k: fnld[k].replace("dmaps", "pcd_pts") for k in fnld.keys()}

    for k in fnld.keys():
        obj = pcd_processed[k]
        fn = new_names_for_pcd[k]
        torch.save(obj=obj, f=fn)
        log.info(f"saved processed pcd for key {k}")


# ----------------------------------------------------


# get n permutation per row
def get_n_combinations(n):
    retval = len(list(itertools.combinations(range(n), 2)))
    return retval


def get_row_ranks(index, ranked_c):
    row_rank = ranked_c.loc[index][["rank1", "rank2", "rank3", "rank4", "rank5", "rank6", "rank7"]]
    n_in_row = ranked_c.loc[index].n_in_row.item()
    return dict(n_in_row=n_in_row, row_rank=row_rank)


def get_just_ranks(df_row):
    row_rank = df_row[["rank1", "rank2", "rank3", "rank4", "rank5", "rank6", "rank7"]]
    return row_rank


def get_mesh_im(seed, data_dir):
    mesh_fn = os.path.join(data_dir, "mesh_cat_s_" + str(seed) + ".jpg")
    read_im = mpimg.imread(mesh_fn)
    return read_im


def get_new_dims(im, new_height):
    iic_height = im.shape[0]
    iic_width = im.shape[1]
    ratio = new_height / iic_height
    new_width = int(iic_width * ratio)
    return (new_width, new_height)


def globdir(in_path, extension):
    search_str = os.path.join(in_path, extension)
    return glob.glob(search_str)


# -----------------------------------------


def put_red_border_around_img(current_img_np_array):
    current_img = PIL.Image.fromarray(current_img_np_array)
    img_with_border = ImageOps.expand(current_img, border=15, fill="red")
    img_with_border = np.asarray(img_with_border)
    return img_with_border


def text_on_image(img, text, fontsize=100):
    img = Image.fromarray(img)
    im_width, im_height = img.size
    font_location = (int(4 * im_width / 5) + 100, int(im_height - 160))
    I1 = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf", fontsize)
    I1.text(font_location, text, fill=(255, 0, 0), font=font)
    img = np.asarray(img)
    return img


# record a list of meshes to exclude, based on eyeballing
def setup_seeds_for_good_meshes():
    entire_spec_fn = "/path/to/eg3d-rlhf-geometry/000_RLHF_AM/mesh_outputs_spec_20_09_2023.ods"
    output_types = read_ods(entire_spec_fn, sheet="outputs_basic", headers=True).output_type.values
    conditions_to_synth = ["rlhf_meshes_ffhqrebalanced512-128_tpsi_025"]

    cs = conditions_to_synth[0]

    conditions = read_ods(entire_spec_fn, sheet="eg3d_model_experiment_settings", headers=True).set_index("condition")
    condition_dict = conditions.loc[cs].to_dict()
    sstart = int(condition_dict["seed_start"])
    send = int(condition_dict["seed_end"])
    seeds_good = [s for s in range(sstart, send)]

    log.info("good seeds")
    log.debug(seeds_good)

    exclude_seeds = [100037, 100066]  # glasses look ambiguous
    seeds_good_filtered = [s for s in seeds_good if s not in exclude_seeds]
    log.info("n good mesh")
    log.info(len(seeds_good_filtered))
    seeds_good_fn = "/path/to/eg3d-rlhf-geometry/000_RLHF_AM/rlhf_meshes/rlhf_meshes_ffhqrebalanced512-128_tpsi_025_good_meshes_04_10_2023.pt"
    torch.save(obj=seeds_good_filtered, f=seeds_good_fn)


# runs transfropms loads .pt files and saves into a single .gz file
def precompute_dataset_to_npgz(names_of_pt):
    working_dir = str(COMPRESSION_TMP_DIR)

    os.makedirs(working_dir, exist_ok=True)
    for k in names_of_pt.keys():
        log.info(f"processing key: {k}")

        train_loader = torch.utils.data.DataLoader(torch.load(names_of_pt[k]))

        dmaps_list = []
        blens_list = []

        entire_files_list = []
        for batch in tqdm(train_loader):
            dummy = batch

            if len(entire_files_list) == 0:
                keys = [k for k in dummy[0][1].keys()]

                for kk in range(len(keys)):
                    entire_files_list.append([])

            for i, kk in enumerate(keys):
                # Eif 'len' not in k:
                dk = dummy[0][1][kk]
                if len(dk) == 0:
                    current_f = torch.hstack([d for d in dummy[0][1][kk]]).unsqueeze(0)  # .half()

                else:
                    # blen=torch.hstack([d for d in dummy[0][1]['batch_len']]).unsqueeze(0)
                    current_f = torch.cat([d for d in dummy[0][1][k]]).unsqueeze(0).half()

                entire_files_list[i].append(current_f)

            # dmap=torch.cat([d for d in dummy[0][1]['files']]).unsqueeze(0)#.half()
            # blen=torch.hstack([d for d in dummy[0][1]['batch_len']]).unsqueeze(0)#.to(torch.*.ByteTensor)
            # dmaps_list.append(dmap)
            # blens_list.append(blen)
        for i in range(len(keys)):
            # current_f=torch.cat([d for d in dummy[0][1][k]]).unsqueeze(0).half()
            entire_files_list[k] = torch.cat(entire_files_list[k])
        log.info("concatenating entire dset...")

        new_save_dir = os.path.join(working_dir, k)

        os.makedirs(new_save_dir, exist_ok=True)
        dmaps_cat = dmaps_cat.detach().cpu().numpy()

        import gzip

        out_fn = os.path.join(new_save_dir, "dmaps_cat.npy")  # ,dmaps_cat

        # f = gzip.GzipFile(out_fn.replace('.npy','.npy.gz'), "w")
        # np.save(file=f, arr=dmaps_cat)
        # f.close()

        out_fn = os.path.join(new_save_dir, "blens_list.npy")  # ,dmaps_cat

        f = gzip.GzipFile(out_fn.replace(".npy", ".npy.gz"), "w")
        np.save(file=f, arr=blens_list)
        f.close()

        # dt=
        # numpy.save(out_fn,dmaps_cat)#'/path/to/eg3d-rlhf-geometry/reward_model_training/compression_dl/data.npy', dmaps_cat)
        # print(os.path.getsize('/path/to/eg3d-rlhf-geometry/reward_model_training/compression_dl/data.npy'))
        # 4000080 uncompressed size
        # subprocess.call(f'xz -9 --threads=8 {out_fn}', shell=True)

        # numpy.save(out_fn,blens_list)#'/path/to/eg3d-rlhf-geometry/reward_model_training/compression_dl/data.npy', dmaps_cat)
        # print(os.path.getsize('/path/to/eg3d-rlhf-geometry/reward_model_training/compression_dl/data.npy'))
        # 4000080 uncompressed size
    # subprocess.call(f'xz -9 --threads=8 {out_fn}', shell=True)

    # dataset = TensorDataset(dmaps_cat, blens_cat)
    # print('saving...')
    # torch.save(obj=dataset,f=f'{names_of_pt[k]}'.replace('.pt','_PRECOMPUTED.pt')) #953mb...
    # print('done....')


def precompute_dataset_to_pt(names_of_pt):
    working_dir = str(COMPRESSION_TMP_DIR)

    os.makedirs(working_dir, exist_ok=True)
    for partition in names_of_pt.keys():
        log.info(f"processing key: {partition}")

        train_loader = torch.utils.data.DataLoader(torch.load(names_of_pt[partition]))

        dmaps_list = []
        blens_list = []

        entire_files_list = []
        for batch in tqdm(train_loader):
            dummy = batch

            if len(entire_files_list) == 0:
                keys = [k for k in dummy[0][1].keys()]

                for kk in keys:
                    log.debug(f"found key {kk}")
                    entire_files_list.append([])

            for i, k in enumerate(keys):
                # Eif 'len' not in k:
                dk = dummy[0][1][k]
                # print(dk)
                # print(len(dk[0].shape))
                if len(dk[0].shape) == 0:
                    current_f = torch.hstack([d for d in dummy[0][1][k]]).unsqueeze(0)  # .half()
                else:
                    current_f = torch.cat([d for d in dummy[0][1][k]]).unsqueeze(0).half()
                entire_files_list[i].append(current_f)

            # dmap=torch.cat([d for d in dummy[0][1]['files']]).unsqueeze(0)#.half()
            # blen=torch.hstack([d for d in dummy[0][1]['batch_len']]).unsqueeze(0)#.to(torch.*.ByteTensor)
            # dmaps_list.append(dmap)
            # blens_list.append(blen)
        for i in range(len(keys)):
            # current_f=torch.cat([d for d in dummy[0][1][k]]).unsqueeze(0).half()
            entire_files_list[i] = torch.cat(entire_files_list[i])

        log.info("concatenating entire dset...")

        dataset = TensorDataset(*entire_files_list)
        log.info("saving precomputed dataset")
        torch.save(obj=dataset, f=f"{names_of_pt[partition]}".replace(".pt", "_PRECOMPUTED.pt"))  # 953mb...
        log.info("save complete")

    return


# expecting points shape as N, dim
def jitter_points_uniform(points, size=0.001):
    pos = points
    orig_shape = pos.shape

    assert len(pos.shape) == 2, "error u need the 2 dim positions for rotation thing"
    if pos.shape[-1] != 3:
        pos = pos.permute(1, 0)

    (n, dim), t = pos.size(), size
    if isinstance(t, numbers.Number):
        t = list(repeat(t, times=dim))
    assert len(t) == dim

    ts = []
    for d in range(dim):
        ts.append(torch.empty_like(pos[:, 0]).uniform_(-abs(t[d]), abs(t[d])))

    pos = pos + torch.stack(ts, dim=-1)
    return pos.reshape(orig_shape)


def mean_scale_pts(ttl):
    ttl_c = ttl - ttl.mean(dim=0, keepdim=True)
    scale = (1 / ttl_c.abs().max()) * 0.999999
    ttl_c = ttl_c * scale
    return ttl_c


def downsample_pcd_points(ttl, n_points=5000):
    perm = torch.randperm(ttl.size(0))
    idx = perm[: min(ttl.size(0), n_points)]
    samples = ttl[idx]
    return samples


def random_translate_points(ttl, dist=0.2):
    translation = torch.empty(1, 3).uniform_(-dist, dist).expand(ttl.shape)
    ttl_c = ttl + translation
    return ttl_c


def random_scale_points_along_axes(ttl, margins=0.05):
    translation = torch.empty(1, 3).uniform_(1 - margins, 1 + margins).expand(ttl.shape)
    ttl_c = ttl * translation
    return ttl_c


def center_points(ttl):
    ttl_c = ttl - ttl.mean(dim=0, keepdim=True)
    return ttl_c


def get_upsample_normalise_as_v2_transform():
    upsample_normalise = v2.Compose(
        [
            v2.Resize(
                size=(256, 256),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),  # Or Resize(antialias=True)
            # v2.RandomCrop(size=160),
            v2.Lambda(normfunc),
        ]
    )
    return upsample_normalise


def create_dataloader_from_types_and_dsets_dict(dict_of_dsets, selected_dtypes):
    sel_dsets_dict = {k: dict_of_dsets[k] for k in selected_dtypes}
    sel_dsets = [sel_dsets_dict[i] for i in sel_dsets_dict.keys()]
    concat_dataset = dset_smulti_stream(sel_dsets)
    combined_dloader = torch.utils.data.DataLoader(concat_dataset, batch_size=64, shuffle=True, num_workers=0)
    return combined_dloader


def create_dset_from_types_and_dsets_dict(dict_of_dsets, selected_dtypes):
    sel_dsets_dict = {k: dict_of_dsets[k] for k in selected_dtypes}
    sel_dsets = [sel_dsets_dict[i] for i in sel_dsets_dict.keys()]
    concat_dataset = dset_smulti_stream(sel_dsets)
    return concat_dataset


def check_row(idx_of, ranked_c):
    possible_rank_letters = np.array(["A", "B", "C", "D", "E", "F"])
    rr = get_row_ranks(idx_of, ranked_c)
    vv = rr["row_rank"].values.astype("str")
    vv = vv[vv != "nan"]
    all_in_row = len(vv) == rr["n_in_row"]
    ranks = rr["row_rank"]
    necessary_rank_letters = possible_rank_letters[: rr["n_in_row"]]
    ranked_cases = ranks[: rr["n_in_row"]]
    length_and_unique_check = set(necessary_rank_letters.tolist()) == set(ranked_cases.tolist())
    if all_in_row and length_and_unique_check:
        return True
    else:
        return False


# def get_dsets_dict_for_list_of_rankings_minimal(list_of_rankings, ddir_func, using_transforms=False):
#     dict_of_dsets = {}
#     for dt in [
#         "triple_dmap",
#         "single_dmap",
#         "ws_code_view_conditioned",
#         "ws_code_unconditioned",
#         "triple_rgb_lmks_98",
#         "canonical_rgb_lmks_98",
#         "triple_rgb",
#         "canonical_rgb",
#         "mesh_crop_ptgeom",
#         "pcd",
#         "pcd_as_pt",
#         "lmks_pcd",
#     ]:
#         dset = dset_single_stream_ordered_minimal(
#             all_combined_rankings=list_of_rankings,
#             dtype=dt,
#             ddir_func=ddir_func,
#             using_transforms=using_transforms,
#         )
#         dict_of_dsets[dt] = dset

#     return dict_of_dsets


def get_dsets_dict_for_list_of_rankings_minimal(list_of_rankings, ddir_func, using_transforms=False):
    """
    Canonical wrapper; delegates to the implementation in core_modules.data.ranking_datasets to avoid divergence.
    """
    from core_modules.data import ranking_datasets

    return ranking_datasets.get_dsets_dict_for_list_of_rankings_minimal(list_of_rankings, ddir_func, using_transforms)


def append_good_meshes_to_ranked_seeds(ranked_seeds, goodmesh_idx_list):
    np.random.seed(42)

    rankings_idx = np.arange(len(ranked_seeds))
    np.random.shuffle(rankings_idx)
    ranks_split = np.array_split(rankings_idx, len(goodmesh_idx_list))

    new_list_of_comparisons = []

    for i in np.arange(len(ranks_split)):
        current_good_mesh_idx = goodmesh_idx_list[i]
        current_splits = [ranked_seeds[k] for k in ranks_split[i]]

        appended_list_of_rankings = []

        for s in current_splits:
            us = torch.from_numpy(np.unique(s))  # get unique seed in batch
            reshape_unique = us.unsqueeze(1)
            superior_mesh = torch.ones_like(reshape_unique) * current_good_mesh_idx
            extra_comparison = torch.cat([superior_mesh, reshape_unique], dim=1)
            np_comparison = extra_comparison.detach().cpu().numpy()
            new_comparison = np.concatenate([np_comparison, s], axis=0)
            appended_list_of_rankings.append(new_comparison)

        p_add = 0.4

        if np.random.sample() < p_add:
            new_list_of_comparisons.append(appended_list_of_rankings)
        else:
            new_list_of_comparisons.append(current_splits)

    lll_reduced = list(itertools.chain.from_iterable(new_list_of_comparisons))

    return lll_reduced


def normfunc(dmap, lower=2.25, upper=2.95):
    retval = ((dmap - lower) / (upper - lower)) * 2 - 1
    retval[retval < -1.0] = -1.0
    retval[retval > 1.0] = 1.0
    return retval


def append_good_meshes_to_ranked_seeds_minimal(ranked_seeds_minimal, goodmesh_idx_list, p_add=0.4):
    ranked_seeds_minimal = torch.from_numpy(ranked_seeds_minimal)
    neg_ones = torch.ones_like(ranked_seeds_minimal[:, -1:]) * -1
    ranked_w_dummy = torch.cat([ranked_seeds_minimal, neg_ones], dim=1)

    np.random.seed(42)

    rankings_idx = np.arange(len(ranked_w_dummy))
    np.random.shuffle(rankings_idx)
    ranks_split = np.array_split(rankings_idx, len(goodmesh_idx_list))

    new_list_of_comparisons = []

    for i in np.arange(len(ranks_split)):
        current_good_mesh_idx = goodmesh_idx_list[i]

        current_gmesh_for_insert = torch.ones([1]) * current_good_mesh_idx
        current_splits = [ranked_w_dummy[k] for k in ranks_split[i]]

        appended_list_of_rankings = []

        for s in current_splits:
            if np.random.sample() < p_add:
                new_combo = torch.cat([current_gmesh_for_insert, s[1:]])
                new_list_of_comparisons.append(new_combo)
            else:
                new_list_of_comparisons.append(s)

    new_list_of_comparisons = [nlc.unsqueeze(0) for nlc in new_list_of_comparisons]

    new_list_of_comparisons = torch.cat(new_list_of_comparisons, dim=0).cpu().detach().numpy().astype(np.int32)

    return new_list_of_comparisons


def get_seed_from_composed(in_str):
    retval = in_str.split("/")[-1].split("_s_")[-1].split(".")[0]
    return retval


def insert_seed_to_rankings(row_for_clone):
    row_of_it_all = row_for_clone.copy()
    n_in_row = row_of_it_all.n_in_row
    ranks = get_just_ranks(row_of_it_all)

    for idx in range(n_in_row):
        current_rank = ranks[f"rank{idx + 1}"]
        current_seed_rank = seed_letter_concordance[current_rank]
        current_np_seed = get_seed_from_composed(row_of_it_all[current_seed_rank])
        row_of_it_all[f"rank{idx + 1}"] = current_np_seed

    return row_of_it_all


# get the listofrankings for the dataloader....
def get_list_of_rankings(ranked_seeds_for_dataloader):
    list_of_rankings = []
    for idx in ranked_seeds_for_dataloader.index:
        list_of_seeds = ranked_seeds_for_dataloader.loc[idx].dropna().astype(np.int32).values
        n_meshes = len(list_of_seeds)
        ordered_pairs = np.array(list(itertools.combinations(range(n_meshes), 2)))  # 2 for pairwise...
        ordered_seeds = np.array(list_of_seeds)[ordered_pairs]
        list_of_rankings.append(ordered_seeds)

    assert len(list_of_rankings) == ranked_seeds_for_dataloader.shape[0], "error list of rankigns not same as n row in ranked seeds"
    return list_of_rankings


def dataframe_to_ranked_seeds(ranked_c_df):
    ranked_c_for_seeds = ranked_c_df.copy()
    for idx in ranked_c_for_seeds.index:
        ranked_c_for_seeds.loc[idx] = insert_seed_to_rankings(ranked_c_for_seeds.loc[idx])

    ranked_seeds_for_dataloader = ranked_c_for_seeds[["rank1", "rank2", "rank3", "rank4", "rank5", "rank6", "rank7"]]
    return ranked_seeds_for_dataloader


def create_list_of_rankings(ranked_c_df):
    ranked_seeds_for_dataloader = dataframe_to_ranked_seeds(ranked_c_df)
    list_of_rankings = get_list_of_rankings(ranked_seeds_for_dataloader)
    return list_of_rankings


# get the listofrankings for the dataloader....
def get_list_of_rankings_minimal_rows(ranked_seeds_for_dataloader):
    list_of_rankings = []
    for idx in ranked_seeds_for_dataloader.index:
        list_of_seeds = ranked_seeds_for_dataloader.loc[idx].fillna(-1).astype(np.int32).values
        # n_meshes=len(list_of_seeds)
        # ordered_pairs=np.array(list(itertools.combinations(range(n_meshes), 2))) #2 for pairwise...
        ordered_seeds = np.array(list_of_seeds)  # [ordered_pairs]
        list_of_rankings.append(ordered_seeds)

    assert len(list_of_rankings) == ranked_seeds_for_dataloader.shape[0], "error list of rankigns not same as n row in ranked seeds"
    return np.array(list_of_rankings)


def create_list_of_rankings_minimal(ranked_c_df):
    ranked_seeds_for_dataloader = dataframe_to_ranked_seeds(ranked_c_df)
    list_of_rankings = get_list_of_rankings_minimal_rows(ranked_seeds_for_dataloader)
    return list_of_rankings


# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel


def rescale_im(dmap):
    rmin = 2.25
    rmax = 3.3
    dm_min = -1.0
    dmap = (((dmap - rmin) / (rmax - rmin)) * 2) - 1
    dmap[dmap < dm_min] = dm_min
    dmap[dmap > 1.0] = 1.0
    return dmap


def rescale_im_dmp_for_lmk(dmap):
    rmin = 2.25
    rmax = 3.3
    dm_min = -1.0
    dmap = (((dmap - rmin) / (rmax - rmin)) * 2) - 1
    dmap[dmap < dm_min] = dm_min
    dmap[dmap > 1.0] = 1.0
    return dmap


def rescale_dmap_to_raymarching_limits(dmap):
    rmin = 2.25
    rmax = 3.3
    dm_min = -1.0
    dmap = (((dmap - rmin) / (rmax - rmin)) * 2) - 1
    dmap[dmap < dm_min] = dm_min
    dmap[dmap > 1.0] = 1.0
    return dmap


def read_in_row(row):
    read_ims = []
    for c in row.index:
        if c.startswith("seed"):
            if type(row[c]) == str and len(row[c]) > 0:
                im = mpimg.imread(row[c])
                read_ims.append(im)
            else:
                pass
    return read_ims


def extract_ranks(current_row):
    rank_cols = [c for c in current_row.index if "rank" in c]
    rankings_from_row = current_row[rank_cols]
    ranked_letters = np.array(rankings_from_row)[np.array([is_not_none(k) for k in rankings_from_row.values])]
    return ranked_letters


def return_greyed_image(image_as_np_array):
    iic = image_as_np_array
    myBackgroundImage = PIL.Image.fromarray(iic)
    iic = copy.deepcopy(iic)

    iic.fill(100)

    iic_tp = np.dstack((iic, np.zeros((iic.shape[0], iic.shape[1]))))
    iic_tp[:, :, 3] = 225
    iic_tpi = PIL.Image.fromarray(iic_tp.astype(np.uint8))

    myForegroundImage = iic_tpi
    myMerged_image = Image.new("RGBA", myBackgroundImage.size)
    myMerged_image.paste(myBackgroundImage, (0, 0))
    _, _, _, mask = myForegroundImage.split()
    myMerged_image.paste(myForegroundImage, (0, 0), mask)

    return np.asarray(myMerged_image)


def is_not_none(val):
    if type(val) == str and len(val) > 0:
        return True
    else:
        return False


def compose_meshes_for_ranking():
    composed_dir = os.path.join(data_dir, "composed_for_ranking")

    seeds = return_ffhq_seeds()

    ims = globdir(composed_dir, "composed_s_*.jpg")

    s_completed = [s.split("composed_s_")[-1].split(".jpg")[0] for s in ims]

    sc = [int(s) for s in s_completed]

    seeds_remain = [s for s in seeds if s not in sc]
    log.info("seeds remaining to synthesise")
    log.info(len(seeds_remain))

    for s in tqdm.tqdm(seeds):
        a = assemble_triple_rgb(s, data_dir)
        b = get_mesh_im(s, data_dir)
        c = compose_vertical_from_np(b, a)
        c.save(os.path.join(composed_dir, "composed_s_" + str(s) + ".jpg"))


def format_rankings_csv_for_seeds(data_dir, saving_file=False):
    rankings_dir = os.path.join(data_dir, "rankings_data")
    records_save_name = os.path.join(rankings_dir, "mesh_seed_choices.csv")
    path_seed_choices = pd.read_csv(records_save_name).set_index("index")  # .head(10)
    for c in path_seed_choices.columns:
        path_seed_choices[c] = path_seed_choices[c].apply(convert_seed_to_path)

    path_seed_choices.head(10)
    path_seed_choices_records_name = os.path.join(rankings_dir, "path_seed_choices.csv")
    path_seed_choices.to_csv(path_seed_choices_records_name)
    pdr = pd.read_csv(path_seed_choices_records_name).set_index("index")
    pdr.head()

    pdr_rankings = pdr.copy()

    pdr_rankings["rank1"] = None
    pdr_rankings["rank2"] = None
    pdr_rankings["rank3"] = None
    pdr_rankings["rank4"] = None
    pdr_rankings["rank5"] = None
    pdr_rankings["rank6"] = None
    pdr_rankings["rank7"] = None
    pdr_rankings["completed"] = False

    pdr_rankings["n_in_row"] = None
    n_in_each_row = []

    for i in pdr_rankings.index:
        rowvals = pdr_rankings.loc[i].values
        n_in_row = sum([is_not_none(k) for k in rowvals])
        pdr_rankings.loc[i, "n_in_row"] = n_in_row

    n_in_each_row = pdr_rankings.n_in_row.values
    min_n_in_each_row = min(n_in_each_row)
    log.info(f"min seeds per row: {min_n_in_each_row}")
    max_n_in_each_row = max(n_in_each_row)
    log.info(f"max seeds per row: {max_n_in_each_row}")

    # pdr_rankings.drop(columns=['seed1','seed2','seed3','seed4','seed5','seed6','seed7'],inplace=True)

    rankings_records_name = os.path.join(rankings_dir, "rankings_records.csv")

    if not os.path.isfile(rankings_records_name) and saving_file:
        pdr_rankings.to_csv(rankings_records_name)


def list_all_files_in_dir_and_create_composed(data_dir):
    all_in_dir = os.listdir(data_dir)
    all_in_dir = [a for a in all_in_dir if a.endswith(".jpg")]
    log.info("n files in current data dir")
    log.info(len(all_in_dir))

    composed_dir = os.path.join(data_dir, "composed_for_ranking")
    os.makedirs(composed_dir, exist_ok=True)


def join_all_ims_together(pdr_rankings, data_dir):
    pdr_rankings_c = pdr_rankings.copy()
    os.makedirs(os.path.join(data_dir, "rankings", "joined_ims"), exist_ok=True)

    for i in tqdm.tqdm(pdr_rankings_c.index):
        current_im = combine_row(pdr_rankings_c.loc[i])
        out_fn = os.path.join(data_dir, "rankings", "joined_ims", "joined_idx_" + str(i) + ".jpg")
        current_im = PIL.Image.fromarray(current_im).convert("RGB")
        current_im.save(out_fn)


def get_rankings_from_pdr(pdr_rankings):
    starting_idx = 10000
    current_row = pdr_rankings.loc[current_idx]  # .n_in_row
    n_to_rank = current_row.n_in_row
    rank_cols = [c for c in current_row.index if "rank" in c]
    rankings_from_row = current_row[rank_cols]
    n_ranked = sum([is_not_none(k) for k in rankings_from_row.values])
    log.info(f"rankings completed in current row: {n_ranked}")


def view_first_example_of_joined_im(data_dir):
    tims = glob.glob(os.path.join(data_dir, "rankings", "joined_ims", "joined_idx_*.jpg"))
    if not tims:
        log.warning("no joined images found to view")
        return
    log.info(tims[0])
    tt = PIL.Image.open(tims[0])
    tt.show()


# -----------------------------------------


# ---------------------------------------------------
# Point Clouds
# ---------------------------------------------------


class universal_mesh_format:
    def __init__(self, mesh_object):
        if type(mesh_object) == DracoPy.DracoMesh:
            self.points = mesh_object.points
            self.faces = mesh_object.faces
            # self.normals=mesh_object.normals

        if type(mesh_object) == open3d.cuda.pybind.geometry.TriangleMesh or type(mesh_object) == open3d.geometry.TriangleMesh:
            self.points = np.asarray(mesh_object.vertices)
            self.faces = np.asarray(mesh_object.triangles)
            # self.normals=mesh_object.normals
        if type(mesh_object) == trimesh.base.Trimesh:
            self.points = np.asarray(mesh_object.vertices)
            self.faces = np.asarray(mesh_object.faces)

        if type(mesh_object) == torch_geometric.data.data.Data:
            self.points = mesh_object.pos.numpy()
            self.faces = mesh_object.face.numpy().transpose()

    def as_dracopy(self):
        log.warning("as_dracopy not implemented")

    def as_open3d(self):
        retval = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(self.points),
            triangles=o3d.utility.Vector3iVector(self.faces),
        )
        return retval

    def as_trimesh(self):
        retval = trimesh.Trimesh(vertices=self.points, faces=self.faces)
        return retval

    def as_ptg_data(self):
        retval = trimesh.Trimesh(vertices=self.points, faces=self.faces)
        trimesh.repair.fix_inversion(retval)
        retval = torch_geometric.utils.from_trimesh(retval)
        return retval

    def as_o3d_pcd(self):
        retval = o3d.geometry.TriangleMesh(points=o3d.utility.Vector3dVector(self.points))
        return retval

    def visualise_points(self):
        retval = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(self.points))
        o3d.visualization.draw_geometries([retval])


def get_canon_dmap(fn):
    tens = torch.load(fn)
    return tens[1]


def convert_draco_mesh_to_croppped_pt_geom_mesh_and_save(draco_fn):
    with open(draco_fn, "rb") as draco_file:
        dmesh = DracoPy.decode(draco_file.read())
    # print(f"number of points: {len(dmesh.points)}")
    # print(f"number of faces: {len(dmesh.faces)}")
    # print(f"number of normals: {len(dmesh.normals)}")

    cms = get_canonical_dmap_cams
    cam2world_matrix = cms["cam2world_matrix"]
    intrinsics = cms["intrinsics"]
    neural_rendering_resolution = 128
    pcd = o3d.geometry.PointCloud()

    # draco mesh to o3d mesh
    # -------------------------------
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(dmesh.points),
        triangles=o3d.utility.Vector3iVector(dmesh.faces),
    )

    # read in point cloud from depth map
    # -------------------------------
    dmap_fn = draco_fn.replace("/mesh_s_", "/triple_dmap_s_").replace(".drc", ".pt")
    dmp = get_canon_dmap(dmap_fn)  # .squeeze(0)
    ray_origins, ray_directions = rs(cam2world_matrix, intrinsics, neural_rendering_resolution)
    dd, imd_greater = imd_to_xyz_with_radius_cutoff(
        image_depth=dmp,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        neural_rendering_resolution=neural_rendering_resolution,
    )
    pcd.points = o3d.utility.Vector3dVector(dd[:, imd_greater[0], :].cpu().numpy().reshape(-1, 3))
    pcd.estimate_normals()
    new_fn = sel_fn.replace("/triple_dmap_s_", "/pcd_s_").replace(".pt", ".pcd")
    o3d.io.write_point_cloud(filename=new_fn, pointcloud=pcd)

    # -------------------------------------
    # corresponding pcd from depth map, get bounding box
    bbb = pcd.get_axis_aligned_bounding_box()
    corners = np.asarray(bbb.get_box_points())

    # -------------------------------------
    # Convert the corners array to have type float64
    bounding_polygon = corners.astype("float64")
    bounding_polygon *= 1.05
    # Create a SelectionPolygonVolume
    vol = o3d.visualization.SelectionPolygonVolume()
    # polygon vertices.
    vol.orthogonal_axis = "Y"
    vol.axis_max = np.max(bounding_polygon[:, 1])
    vol.axis_min = np.min(bounding_polygon[:, 1])

    # Set all the Y values to 0 (they aren't needed since we specified what they
    # should be using just vol.axis_max and vol.axis_min).
    bounding_polygon[:, 1] = 0

    # Convert the np.array to a Vector3dVector
    vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)

    # rotate o3d mesh
    # ---------------------------------
    mesh_r = copy.deepcopy(mesh)
    R = mesh.get_rotation_matrix_from_xyz((0, -np.pi / 2, 0))
    mesh_r.rotate(R, center=(0, 0, 0))
    mesh_r.translate([-0.1, 0.11, 0.06])
    # o3d.visualization.draw_geometries([mesh_r])
    crop_mesh = vol.crop_triangle_mesh(mesh_r)

    # reduce n vertices in the mesh
    # ---------------------------------
    pcd_p = np.asarray(pcd.points).shape[0]
    cm_p = np.asarray(crop_mesh.vertices).shape[0]
    ratio = pcd_p / cm_p
    cm_f = np.asarray(crop_mesh.triangles).shape[0]
    tf = int(ratio * cm_f)
    cme = crop_mesh.simplify_quadric_decimation(tf)
    cme.compute_vertex_normals()

    # convert to trimesh
    # ---------------------------------
    tmesh_cme = trimesh.Trimesh(vertices=np.asarray(cme.vertices), faces=cme.triangles)
    trimesh.repair.fix_inversion(tmesh_cme)
    tmd = torch_geometric.utils.from_trimesh(tmesh_cme)
    pt_geom_mesh_fn = draco_fn.replace("/mesh_s_", "/mesh_crop_ptgeom_s_").replace(".drc", ".pt")
    torch.save(obj=tmd, f=pt_geom_mesh_fn)


def create_point_clouds_from_dmap_and_save(nrs=256):
    ddir_list = [
        "/path/to/eg3d-rlhf-geometry/000_RLHF_AM/rlhf_meshes/rlhf_meshes_ffhqrebalanced512-128_tpsi_025",
        "/path/to/eg3d-rlhf-geometry/000_RLHF_AM/rlhf_meshes/rlhf_meshes_ffhq512-128_const_noise_t1_augment",
    ]

    for data_dir in ddir_list:
        all_draco_fn = glob.glob(os.path.join(data_dir, "*.drc"))
        all_draco_fn = random.sample(all_draco_fn, k=len(all_draco_fn))
        # read in point cloud from depth map directly
        # -------------------------------
        for draco_fn in tqdm.tqdm(all_draco_fn):
            dmap_fn = draco_fn.replace("/mesh_s_", "/triple_dmap_s_").replace(".drc", ".pt")
            dmp = torch.nn.functional.interpolate(get_canon_dmap(dmap_fn).unsqueeze(0), size=(nrs, nrs), mode="bilinear")  # .squeeze(0)
            ray_origins, ray_directions = rs(cam2world_matrix, intrinsics, nrs)
            dd, imd = imd_to_xyz_with_radius_cutoff(
                image_depth=dmp,
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                neural_rendering_resolution=nrs,
                radius_cutoff=2.7,
            )  # 2.7 LAST..., #2.60 can get it working, try with more, 2.7=more signal, 2.9 too much
            dd = dd[:, imd[0], :].reshape(-1, 3)
            pt_pcd_fn = draco_fn.replace("/mesh_s_", "/pcd_as_pt_s_").replace(".drc", ".pt")
            torch.save(obj=dd, f=pt_pcd_fn)


import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def get_xyz_from_lmk(ll):
    xlmk = [l.x for l in ll]
    ylmk = [l.y for l in ll]
    zlmk = [l.z for l in ll]
    return np.array([xlmk, ylmk, zlmk]).T


def create_export_mediapipe_lmks_as_pt():
    ddir_list = [
        "/path/to/eg3d-rlhf-geometry/000_RLHF_AM/rlhf_meshes/rlhf_meshes_ffhqrebalanced512-128_tpsi_025",
        "/path/to/eg3d-rlhf-geometry/000_RLHF_AM/rlhf_meshes/rlhf_meshes_ffhq512-128_const_noise_t1_augment",
    ]

    ddir = ddir_list[0]

    for ddir in ddir_list:
        gg = globdir(ddir, "triple_rgb_s_*_1.jpg")

        file_list = gg

        annotated_images = []
        create_img = False

        # For static images:
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        with mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5) as face_mesh:
            for file in tqdm.tqdm(file_list):
                image = cv2.imread(file)
                # Convert the BGR image to RGB before processing.
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Print and draw face mesh landmarks on the image.
                if not results.multi_face_landmarks:
                    continue
                annotated_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
                # print('len results:', len(results.multi_face_landmarks[0]))

                out_fn = file.replace("triple_rgb_s_", "mediapipe_468_lmk_s_").replace(".jpg", ".pt")

                # print(out_fn)

                if not results.multi_face_landmarks:
                    log.error(f"no landmarks for file {file}")
                    assert False, "error not landmarks for some exaample"

                if not len(ll) == 468:
                    log.error(f"expected 468 landmarks for file {file}")
                    assert False, "error len of lmk not 468 for some exaample"
                # print()

                if create_img:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=annotated_image,
                            landmark_list=face_landmarks,
                            # connections=mp_face_mesh.FACE_CONNECTIONS,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec,
                        )

                    annotated_images.append(annotated_image)

                ll = results.multi_face_landmarks[0].landmark

                xyz = get_xyz_from_lmk(ll)
                dtype = torch.float32
                xyz = torch.tensor(xyz, dtype=dtype)

                torch.save(obj=xyz, f=out_fn)


def return_checked_combos(data_dir):
    ranked = pd.read_csv(os.path.join(data_dir, "rankings_data/rankings_records.csv")).set_index("index")

    ranked_c = ranked[ranked.completed == True]
    ranked.head()

    all_idx = [i for i in ranked_c.index]
    checked_rows = [check_row(idx, ranked_c) for idx in all_idx]
    # pull out offending idx
    invert_idx = [not c for c in checked_rows]
    idx_with_mistakes = np.array(all_idx)[invert_idx]
    log.info("mistake rankings idx")
    log.info(len(idx_with_mistakes))
    # for i in idx_with_mistakes:
    #    print(i)

    checked_rankings = ranked_c[checked_rows]

    # ranked_c[checked_rows].n_in_row
    total_combos = [get_n_combinations(nrow) for nrow in checked_rankings.n_in_row]
    log.info(f"total n combo for thise: {np.sum(total_combos)}")

    # checked_rankings=checked_rankings.sample(n=100)

    # ranked_c[checked_rows].n_in_row
    total_combos = [get_n_combinations(nrow) for nrow in checked_rankings.n_in_row]
    log.info(f"total n combo for thise: {np.sum(total_combos)}")

    return checked_rankings


# printing module info:


# from torchinfo import summary

# model = rm
# model.return_global_only=True
# batch_size = 1
# summary(model, input_size=(batch_size, 129, 141,128),    dtypes=[torch.float],
#     verbose=2,
#     col_width=30,
#     col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
#     row_settings=["var_names"],
# )
