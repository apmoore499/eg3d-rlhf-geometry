# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Train a GAN using the techniques described in the paper "Efficient Geometry-aware 3D Generative Adversarial Networks.".

Code adapted from "Alias-Free Generative Adversarial Networks".
"""

import copy
import json
import os
import re
import tempfile
import autoroot

import click
import dnnlib
import hydra
import torch
from hydra import compose, initialize
from metrics import metric_main
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch_utils import custom_ops, training_stats
from training import training_loop

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

try:
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
except:
    next

# ----------------------------------------------------------------------------


def subprocess_fn(rank, c, temp_dir, hydra_cfg=None):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, "log.txt"), file_mode="a", should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, ".torch_distributed_init"))
        if os.name == "nt":
            init_method = "file:///" + init_file.replace("\\", "/")
            torch.distributed.init_process_group(backend="gloo", init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f"file://{init_file}"
            torch.distributed.init_process_group(backend="nccl", init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device("cuda", rank) if c.num_gpus > 1 else None

    if torch.multiprocessing.get_start_method(allow_none=True) != "spawn":
        training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)

    if rank != 0:
        custom_ops.verbosity = "none"

    # Execute training loop.
    training_loop.training_loop(rank=rank, hydra_cfg=hydra_cfg, **c)


# ----------------------------------------------------------------------------


def launch_training_rlhf(c, desc, outdir, dry_run, hydra_cfg):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r"^\d+", x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1

    # ^^ we don't use their old directory naming structure
    # rts=dnnlib.EasyDict(c.loss_kwargs.rlhf_opts.regularisation_terms)
    # lrwd,lD,lG,lGcanon=rts.lambda_reward_tune,rts.lambda_D_gain,rts.lambda_G_gain,rts.lambda_G_canonical

    # lD_l1=rts.lambda_dmap_shift_l1
    # lD_mse=rts.lambda_dmap_shift_mse
    # assert lD_l1 ==0 or lD_mse==0, 'error cant have both l1 and mse for dmap shift'

    # l#dmap=float(lD_l1)+float(lD_mse)

    # ptmd=c.loss_kwargs.rlhf_opts.reward_model_name.replace('.pkl','')

    # mdn=ptmd.split('/')[-1]

    # @TODO: fix resume pkl saving

    # rlhf_rdir=f'm_{mdn}_lams_dmap_{ldmap}_rwd_{lrwd}_D_{lD}_G_{lG}_Gcanon_{lGcanon}'

    # if c.loss_kwargs.rlhf_opts.resume_pkl is not None:
    #     pkl_name=c.loss_kwargs.rlhf_opts.resume_pkl.split('/')[-1].split('.')[0]
    # else:
    #     pkl_name='NONE'

    # pkl_str=f'_PKL_{pkl_name}_'

    c.run_dir = os.path.join(outdir, f"{cur_run_id:05d}-{desc}")  # +pkl_str+rlhf_rdir)

    # c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print("Training options:")

    if c.get("metrics") is not None:
        cmg = c.get("metrics")
        cmg = [i for i in cmg]  # convert listconfig to list
        c.metrics = cmg

    c_train = copy.deepcopy(c)

    # c_train=c

    # c_train.loss_kwargs.pop('reward_model')
    print(json.dumps(c_train, indent=2))
    print()
    print(f"Output directory:    {c.run_dir}")
    print(f"Number of GPUs:      {c.num_gpus}")
    print(f"Batch size:          {c.batch_size} images")
    print(f"Training duration:   {c.total_kimg} kimg")
    print(f"Dataset path:        {c.training_set_kwargs.path}")
    print(f"Dataset size:        {c.training_set_kwargs.max_size} images")
    print(f"Dataset resolution:  {c.training_set_kwargs.resolution}")
    print(f"Dataset labels:      {c.training_set_kwargs.use_labels}")
    print(f"Dataset x-flips:     {c.training_set_kwargs.xflip}")
    print()

    # Dry run?
    if dry_run:
        print("Dry run; exiting.")
        return

    # Create output directory.
    print("Creating output directory...")
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, "training_options.json"), "w") as f:
        json.dump(c_train, f, indent=2)

    # set hydra main dir to that dir

    # hydra_cfg.hydra.main_dir=c.run_dir

    # hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # hydra_cfg.using_wandb
    cfg = hydra_cfg

    cfg_container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    if cfg.using_wandb:
        if wandb is None:
            raise ImportError("using_wandb=true but wandb is not installed. Install wandb or set using_wandb=false.")
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            config=cfg_container,
            name=f"{c.run_dir}".split("/")[-1],
        )
        # wandb.config.update(cfg)
        print("run id is :")
        print(wandb.run.id)
        cfg.wandb.run_id = wandb.run.id

    # save it out,

    rundir_fn = os.path.join(c.run_dir, "hydra_cfg.yaml")

    with open(rundir_fn, "w") as f:
        OmegaConf.save(cfg, f.name)

    # cfg.hydra.main_dir=c.run_dir

    # cfg_container = OmegaConf.to_container(
    #     cfg, resolve=True, throw_on_missing=True
    # )
    # hydra_cfg.legacy_run_dir=

    # hydra.core.hydra_config.HydraConfig

    # Launch processes.
    print("Launching processes...")

    try:
        if torch.multiprocessing.get_start_method(allow_none=True) != "spawn":
            torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Context already set, ignore
        pass
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir, hydra_cfg=hydra_cfg)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir, hydra_cfg), nprocs=c.num_gpus)


# ----------------------------------------------------------------------------


def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(
            class_name="training.dataset.ImageFolderDataset",
            path=data,
            use_labels=True,
            max_size=None,
            xflip=False,
        )
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution  # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels  # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj)  # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except OSError as err:
        raise click.ClickException(f"--data: {err}")


# ----------------------------------------------------------------------------


def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == "none" or s == "":
        return []
    return s.split(",")


# ----------------------------------------------------------------------------

# class def for pointnet mod


# ----------------------------------------------------------------------------

# put in the dmap model def

# import os

# import torch
# from rlhf_imports import *


# def load_reward_model_dmap(model_dir):


#     # model dir contains saved pkl, and should also contain dict of mparams to init the model

#     mparams_fn=os.path.join(model_dir,'model_init_params.json')

#     with open(mparams_fn) as f:
#         mparams=json.load(f)


#     #dmap_chans=32
#     #mparams=dict(conditional_model=False, dmap_only_mod=True, dmap_chans=dmap_chans)


#     # Find the best model
#     best_model_path = None
#     best_val_loss = float('inf')
#     for filename in os.listdir(model_dir):
#         if filename.startswith('model_losses_') and filename.endswith('.pth'):
#             epoch = int(filename[len('model_losses_'):-len('.pth')])
#             checkpoint = torch.load(os.path.join(model_dir, filename))
#             val_loss = checkpoint['val_loss']
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_model_path = os.path.join(model_dir, 'model_state_dict_{}.pth'.format(epoch))

#     # Load the best model
#     if best_model_path is not None:
#         #classifier = rl_decoder(conditional_model=True, dmap_only_mod=False, dmap_chans=1)
#         classifier = rl_decoder(**mparams)
#         classifier.load_state_dict(torch.load(best_model_path))
#         classifier.eval()
#         print(f'Loaded best model from {best_model_path}')
#         return classifier
#     else:
#         print('No model found in model_dir')


# def load_reward_model_dmap3(model_dir):


#     # model dir contains saved pkl, and should also contain dict of mparams to init the model

#     mparams_fn=os.path.join(model_dir,'model_init_params.json')

#     with open(mparams_fn) as f:
#         mparams=json.load(f)


#     #dmap_chans=32
#     #mparams=dict(conditional_model=False, dmap_only_mod=True, dmap_chans=dmap_chans)


#     # Find the best model
#     best_model_path = None
#     best_val_loss = float('inf')
#     for filename in os.listdir(model_dir):
#         if filename.startswith('model_losses_') and filename.endswith('.pth'):
#             epoch = int(filename[len('model_losses_'):-len('.pth')])
#             checkpoint = torch.load(os.path.join(model_dir, filename))
#             val_loss = checkpoint['val_loss']
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_model_path = os.path.join(model_dir, 'model_state_dict_{}.pth'.format(epoch))

#     # Load the best model
#     if best_model_path is not None:
#         #classifier = rl_decoder(conditional_model=True, dmap_only_mod=False, dmap_chans=1)
#         classifier = rl_decoder_three_dmap(**mparams)
#         classifier.load_state_dict(torch.load(best_model_path))
#         classifier.eval()
#         print(f'Loaded best model from {best_model_path}')
#         return classifier
#     else:
#         print('No model found in model_dir')


# def load_reward_model_pnet(model_dir):

#     classifier = PointNetRankMesh(feature_transform=True)

#     #model_dir = '/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/eg3d/models_rlhf_23_06_2023_rxy_1'
#     #model_dir = '/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/eg3d/models_rlhf_02_07_2023_pnet_bk/'

#     #mparams=dict(conditional_model=False, dmap_only_mod=True, dmap_chans=1)


#     import os

#     import torch

#     #model_dir = './models_rlhf_23_06_2023_rxy_16'
#     # Find the best model
#     best_model_path = None
#     best_val_loss = float('inf')
#     for filename in os.listdir(model_dir):
#         if filename.startswith('model_losses_') and filename.endswith('.pth'):
#             epoch = int(filename[len('model_losses_'):-len('.pth')])
#             checkpoint = torch.load(os.path.join(model_dir, filename))
#             val_loss = checkpoint['val_loss']
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_model_path = os.path.join(model_dir, 'model_state_dict_{}.pth'.format(epoch))

#     # Load the best model
#     if best_model_path is not None:
#         #classifier = rl_decoder(conditional_model=True, dmap_only_mod=False, dmap_chans=16)
#         classifier.load_state_dict(torch.load(best_model_path))
#         classifier.eval()
#         print(f'Loaded best model from {best_model_path}')
#     else:
#         print('No model found in model_dir')


#     return classifier


# ----------------------------------------------------------------------------

# @click.command()

# # Required.
# @click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
# @click.option('--cfg',          help='Base configuration',                                      type=str, required=True)
# @click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
# @click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
# @click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
# @click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0), required=True)

# # Optional features.
# @click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=True, show_default=True)
# @click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
# @click.option('--aug',          help='Augmentation mode',                                       type=click.Choice(['noaug', 'ada', 'fixed']), default='noaug', show_default=True)
# @click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
# @click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)

# # Misc hyperparameters.
# @click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
# @click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
# @click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
# @click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
# @click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
# @click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
# @click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True)
# @click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1), default=2, show_default=True)
# @click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)

# # Misc settings.
# @click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
# @click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
# @click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
# @click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
# @click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=10, show_default=True)
# @click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
# # @click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
# @click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
# @click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
# @click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

# # @click.option('--sr_module',    help='Superresolution module', metavar='STR',  type=str, required=True)
# @click.option('--neural_rendering_resolution_initial', help='Resolution to render at', metavar='INT',  type=click.IntRange(min=1), default=64, required=False)
# @click.option('--neural_rendering_resolution_final', help='Final resolution to render at, if blending', metavar='INT',  type=click.IntRange(min=1), required=False, default=None)
# @click.option('--neural_rendering_resolution_fade_kimg', help='Kimg to blend resolution over', metavar='INT',  type=click.IntRange(min=0), required=False, default=1000, show_default=True)

# @click.option('--blur_fade_kimg', help='Blur over how many', metavar='INT',  type=click.IntRange(min=1), required=False, default=200)
# @click.option('--gen_pose_cond', help='If true, enable generator pose conditioning.', metavar='BOOL',  type=bool, required=False, default=False)
# @click.option('--c-scale', help='Scale factor for generator pose conditioning.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=1)
# @click.option('--c-noise', help='Add noise for generator pose conditioning.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=0)
# @click.option('--gpc_reg_prob', help='Strength of swapping regularization. None means no generator pose conditioning, i.e. condition with zeros.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=0.5)
# @click.option('--gpc_reg_fade_kimg', help='Length of swapping prob fade', metavar='INT',  type=click.IntRange(min=0), required=False, default=1000)
# @click.option('--disc_c_noise', help='Strength of discriminator pose conditioning regularization, in standard deviations.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=0)
# @click.option('--sr_noise_mode', help='Type of noise for superresolution', metavar='STR',  type=click.Choice(['random', 'none']), required=False, default='none')
# @click.option('--resume_blur', help='Enable to blur even on resume', metavar='BOOL',  type=bool, required=False, default=False)
# @click.option('--sr_num_fp16_res',    help='Number of fp16 layers in superresolution', metavar='INT', type=click.IntRange(min=0), default=4, required=False, show_default=True)
# @click.option('--g_num_fp16_res',    help='Number of fp16 layers in generator', metavar='INT', type=click.IntRange(min=0), default=0, required=False, show_default=True)
# @click.option('--d_num_fp16_res',    help='Number of fp16 layers in discriminator', metavar='INT', type=click.IntRange(min=0), default=4, required=False, show_default=True)
# @click.option('--sr_first_cutoff',    help='First cutoff for AF superresolution', metavar='INT', type=click.IntRange(min=2), default=2, required=False, show_default=True)
# @click.option('--sr_first_stopband',    help='First cutoff for AF superresolution', metavar='FLOAT', type=click.FloatRange(min=2), default=2**2.1, required=False, show_default=True)
# @click.option('--style_mixing_prob',    help='Style-mixing regularization probability for training.', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0, required=False, show_default=True)
# @click.option('--sr-module',    help='Superresolution module override', metavar='STR',  type=str, required=False, default=None)
# @click.option('--density_reg',    help='Density regularization strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.25, required=False, show_default=True)
# @click.option('--density_reg_every',    help='lazy density reg', metavar='int', type=click.FloatRange(min=1), default=4, required=False, show_default=True)
# @click.option('--density_reg_p_dist',    help='density regularization strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.004, required=False, show_default=True)
# @click.option('--reg_type', help='Type of regularization', metavar='STR',  type=click.Choice(['l1', 'l1-alt', 'monotonic-detach', 'monotonic-fixed', 'total-variation']), required=False, default='l1')
# @click.option('--decoder_lr_mul',    help='decoder learning rate multiplier.', metavar='FLOAT', type=click.FloatRange(min=0), default=1, required=False, show_default=True)
# @click.option('--rlhf_config_fn',    help='RLHF config file', metavar='STR',  type=str, required=True, default=None)
# @click.option('--resume_kimg',    help='resuming number image', metavar='INT',  type=click.IntRange(min=0), required=False, default=0)


def main_legacy_click(**kwargs):
    """Train a GAN using the techniques described in the paper "Alias-Free Generative Adversarial Networks".

    Examples:

    \b
    # Train StyleGAN3-T for AFHQv2 using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/afhqv2-512x512.zip \\
        --gpus=8 --batch=32 --gamma=8.2 --mirror=1

    \b
    # Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.
    python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/metfacesu-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \\
        --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl

    \b
    # Train StyleGAN2 for FFHQ at 1024x1024 resolution using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan2 --data=~/datasets/ffhq-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=noaug
    """

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs)  # Command line arguments.
    c = dnnlib.EasyDict()  # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(
        class_name="training.networks_stylegan2.Discriminator",
        block_kwargs=dnnlib.EasyDict(),
        mapping_kwargs=dnnlib.EasyDict(),
        epilogue_kwargs=dnnlib.EasyDict(),
    )
    c.G_opt_kwargs = dnnlib.EasyDict(class_name="torch.optim.Adam", betas=[0, 0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name="torch.optim.Adam", betas=[0, 0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name="training.loss.StyleGAN2Loss_with_RLHF_pnet")

    # load in point net....

    # load the pointnet model

    # ---------------------------------

    # load reward model

    # torch.save(classifier.state_dict(), )#classifier_state_dict.pth')

    # POINT NET CLASSIFIER

    # classifier = PointNetRankMesh(feature_transform=True)
    # classifier.load_state_dict(torch.load('/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/eg3d/pointnet_rankmesh_BEST_21_06_2023.PTH'))
    # classifier.eval()

    # CONDITIONAL DMAP CLASSIFIER

    #     import pickle

    #     import yaml

    #     with open(opts.rlhf_config_fn, 'r') as fin:
    #         rlhf_opts = yaml.load(fin, Loader=yaml.FullLoader)
    #     rlhf_opts=dnnlib.EasyDict(rlhf_opts)

    #     #if rlhf_opts.reward_model_type=='depth_map':

    #         #reward_model_name: models_rlhf_11_07_2023_first_3dmap_nrs_64.pkl     #directory where state dict stored
    # #reward_models_dir: /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/##_RLHF_AM/optimal_reward_models/

    #     full_model_path=os.path.join(rlhf_opts.reward_models_dir,rlhf_opts.reward_model_name)

    #     with open(full_model_path, 'rb') as fin:
    #         full_model=pickle.load(fin)

    #     m_init_params=full_model['m_init_params']

    #     reward_model=eval(m_init_params['MODEL_CLASS'])(**m_init_params) #this is reward model....
    #     reward_model.load_state_dict(full_model['model_state_dict'])
    #     reward_model.eval()
    #     reward_model.cuda()

    #     rlhf_opts.reward_model_type=full_model['reward_model_type']

    #     if 'normalisation_scalar' in m_init_params.keys():
    #         rlhf_opts.normalisation_scalar=m_init_params['normalisation_scalar']

    #    classifier=load_reward_model_dmap(rlhf_opts.pretrained_model_dir)
    #    classifier.eval()
    #    classifier.cuda()

    # elif rlhf_opts.reward_model_type=='depth_map_3':
    #    classifier=load_reward_model_dmap3(rlhf_opts.pretrained_model_dir)
    #    classifier.eval()
    #    classifier.cuda()

    # else:
    #    print('reward model type not supported')

    #   assert 1==0

    # ---------------------------------

    # now put rlhf opts into c.loss_kwargs

    # if 'resume' in opts.keys():

    #     rlhf_opts['resume_pkl']=opts.resume
    # else:
    #     rlhf_opts['resume_pkl']=''
    # c.loss_kwargs.rlhf_opts=rlhf_opts

    # c.loss_kwargs.reward_model=reward_model
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException("--cond=True requires labels specified in dataset.json")
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = opts.map_depth
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.loss_kwargs.r1_gamma = opts.gamma
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == "stylegan2" else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException("--batch must be a multiple of --gpus")
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException("--batch must be a multiple of --gpus times --batch-gpu")
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException("--batch-gpu cannot be smaller than --mbstd")
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException("\n".join(["--metrics can only contain the following values:"] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    c.G_kwargs.class_name = "training.triplane.TriPlaneGenerator"
    c.D_kwargs.class_name = "training.dual_discriminator.DualDiscriminator"
    c.G_kwargs.fused_modconv_default = "inference_only"  # Speed up training by using regular convolutions instead of grouped convolutions.
    c.loss_kwargs.filter_mode = "antialiased"  # Filter mode for raw images ['antialiased', 'none', float [0-1]]
    c.D_kwargs.disc_c_noise = opts.disc_c_noise  # Regularization for discriminator pose conditioning

    if c.training_set_kwargs.resolution == 512:
        sr_module = "training.superresolution.SuperresolutionHybrid8XDC"
    elif c.training_set_kwargs.resolution == 256:
        sr_module = "training.superresolution.SuperresolutionHybrid4X"
    elif c.training_set_kwargs.resolution == 128:
        sr_module = "training.superresolution.SuperresolutionHybrid2X"
    else:
        assert False, f"Unsupported resolution {c.training_set_kwargs.resolution}; make a new superresolution module"

    if opts.sr_module != None:
        sr_module = opts.sr_module

    rendering_options = {
        "image_resolution": c.training_set_kwargs.resolution,
        "disparity_space_sampling": False,
        "clamp_mode": "softplus",
        "superresolution_module": sr_module,
        "c_gen_conditioning_zero": not opts.gen_pose_cond,  # if true, fill generator pose conditioning label with dummy zero vector
        "gpc_reg_prob": opts.gpc_reg_prob if opts.gen_pose_cond else None,
        "c_scale": opts.c_scale,  # mutliplier for generator pose conditioning label
        "superresolution_noise_mode": opts.sr_noise_mode,  # [random or none], whether to inject pixel noise into super-resolution layers
        "density_reg": opts.density_reg,  # strength of density regularization
        "density_reg_p_dist": opts.density_reg_p_dist,  # distance at which to sample perturbed points for density regularization
        "reg_type": opts.reg_type,  # for experimenting with variations on density regularization
        "decoder_lr_mul": opts.decoder_lr_mul,  # learning rate multiplier for decoder
        "sr_antialias": True,
    }

    if opts.cfg == "ffhq":
        rendering_options.update(
            {
                "depth_resolution": 48,  # number of uniform samples to take per ray.
                "depth_resolution_importance": 48,  # number of importance samples to take per ray.
                "ray_start": 2.25,  # near point along each ray to start taking samples.
                "ray_end": 3.3,  # far point along each ray to stop taking samples.
                "box_warp": 1,  # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
                "avg_camera_radius": 2.7,  # used only in the visualizer to specify camera orbit radius.
                "avg_camera_pivot": [
                    0,
                    0,
                    0.2,
                ],  # used only in the visualizer to control center of camera rotation.
            }
        )
    elif opts.cfg == "afhq":
        rendering_options.update(
            {
                "depth_resolution": 48,
                "depth_resolution_importance": 48,
                "ray_start": 2.25,
                "ray_end": 3.3,
                "box_warp": 1,
                "avg_camera_radius": 2.7,
                "avg_camera_pivot": [0, 0, -0.06],
            }
        )
    elif opts.cfg == "shapenet":
        rendering_options.update(
            {
                "depth_resolution": 64,
                "depth_resolution_importance": 64,
                "ray_start": 0.1,
                "ray_end": 2.6,
                "box_warp": 1.6,
                "white_back": True,
                "avg_camera_radius": 1.7,
                "avg_camera_pivot": [0, 0, 0],
            }
        )
    else:
        assert False, "Need to specify config"

    # ------------------------------------------------------------------
    # PanoHead arch branch (additive). Defaults to "eg3d" so that every
    # existing config -- which never sets `arch` -- runs the unchanged
    # EG3D path (class names + rendering_options assigned above). Only
    # when arch == "panohead" do we swap to the vendored PanoHead nets
    # and merge the PanoHead-specific rendering / discriminator kwargs.
    arch = opts.get("arch", "eg3d")
    if arch == "panohead":
        c.G_kwargs.class_name = "training.panohead_nets.triplane.TriPlaneGenerator"
        c.D_kwargs.class_name = "training.panohead_nets.dual_discriminator.MaskDualDiscriminatorV2"
        # Vendored superresolution module (isolated PanoHead package).
        rendering_options["superresolution_module"] = "training.panohead_nets.superresolution.SuperresolutionHybrid8XDC"
        # PanoHead rendering kwargs taken from the checkpoint init_kwargs.
        rendering_options.update(
            {
                "clamp_mode": "softplus",
                "c_gen_conditioning_zero": False,
                "gpc_reg_prob": 0.8,
                "density_reg": 0.0,
                "density_reg_p_dist": 0.004,
                "reg_type": "l1",
                "decoder_lr_mul": 1.0,
                "decoder_activation": "none",
                "use_torgb_raw": True,
                "triplane_size": 256,
                "triplane_depth": 3,
                "trans_reg": 10.0,
                "use_background": True,
                "sr_antialias": True,
                "depth_resolution": 48,
                "depth_resolution_importance": 48,
                "ray_start": 2.25,
                "ray_end": 3.3,
                "box_warp": 1,
                "avg_camera_radius": 2.7,
                # PanoHead centers its canonical head at the ORIGIN and its own
                # gen_samples.py hardcodes the camera look-at to [0,0,0]. The
                # pkl's avg_camera_pivot=[0,0,0.2] is a vestigial EG3D value;
                # using it aims the viz camera 0.2 above the head (eyes too
                # high / nose too low / face too far). Use PanoHead's native 0.
                "avg_camera_pivot": [0, 0, 0],
                "image_resolution": 512,
            }
        )
        # MaskDualDiscriminatorV2 needs seg kwargs; values match the pkl.
        c.D_kwargs.seg_resolution = 128
        c.D_kwargs.seg_channels = 1

    if opts.density_reg > 0:
        c.G_reg_interval = opts.density_reg_every

    if "G_reg_interval" in opts.keys():
        c.G_reg_interval = opts.G_reg_interval
    c.G_kwargs.rendering_kwargs = rendering_options
    c.G_kwargs.num_fp16_res = 0
    c.loss_kwargs.blur_init_sigma = 10  # Blur the images seen by the discriminator.
    c.loss_kwargs.blur_fade_kimg = c.batch_size * opts.blur_fade_kimg / 32  # Fade out the blur during the first N kimg.

    c.loss_kwargs.gpc_reg_prob = opts.gpc_reg_prob if opts.gen_pose_cond else None
    c.loss_kwargs.gpc_reg_fade_kimg = opts.gpc_reg_fade_kimg
    c.loss_kwargs.dual_discrimination = True
    c.loss_kwargs.neural_rendering_resolution_initial = opts.neural_rendering_resolution_initial
    c.loss_kwargs.neural_rendering_resolution_final = opts.neural_rendering_resolution_final
    c.loss_kwargs.neural_rendering_resolution_fade_kimg = opts.neural_rendering_resolution_fade_kimg
    c.G_kwargs.sr_num_fp16_res = opts.sr_num_fp16_res

    c.G_kwargs.sr_kwargs = dnnlib.EasyDict(channel_base=opts.cbase, channel_max=opts.cmax, fused_modconv_default="inference_only")

    c.loss_kwargs.style_mixing_prob = opts.style_mixing_prob

    # Augmentation.
    if opts.aug != "noaug":
        c.augment_kwargs = dnnlib.EasyDict(
            class_name="training.augment.AugmentPipe",
            xflip=1,
            rotate90=1,
            xint=1,
            scale=1,
            rotate=1,
            aniso=1,
            xfrac=1,
            brightness=1,
            contrast=1,
            lumaflip=1,
            hue=1,
            saturation=1,
        )
        if opts.aug == "ada":
            c.ada_target = opts.target
        if opts.aug == "fixed":
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100  # Make ADA react faster at the beginning.
        c.ema_rampup = None  # Disable EMA rampup.
        c.resume_kimg = opts.resume_kimg
        if not opts.resume_blur:
            c.loss_kwargs.blur_init_sigma = 0  # Disable blur rampup.
            c.loss_kwargs.gpc_reg_fade_kimg = 0  # Disable swapping rampup

    # Performance-related toggles.
    # if opts.fp32:
    #     c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
    #     c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    c.G_kwargs.num_fp16_res = opts.g_num_fp16_res
    c.G_kwargs.conv_clamp = 256 if opts.g_num_fp16_res > 0 else None  # try g_num_fp16_res=4?
    c.D_kwargs.num_fp16_res = opts.d_num_fp16_res
    c.D_kwargs.conv_clamp = 256 if opts.d_num_fp16_res > 0 else None

    if opts.nobench:
        c.cudnn_benchmark = False

    # opts.

    # Description string.
    desc = f"{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}"
    if opts.desc is not None:
        desc += f"-{opts.desc}"

    return (c, desc, opts.dry_run)


# @hydra.main(config_path="training/cfg_rlhf_tune_AM/rlhf_tune.yaml",strict=False)


@hydra.main(version_base=None, config_path="training/rlhf_tune_configs", config_name="base_config.yaml")
def main_hydra(hydracfg: DictConfig) -> None:
    # legacy_args=[
    #                     "--outdir",
    #                     "/media/krillman/240GB_DATA/training_runs_2",
    #                     "--cfg",
    #                     "ffhq",
    #                     "--data",
    #                     "/media/krillman/DISK5_1TB/t2_ffhq/eg3d_for_dataset/dataset_preprocessing/ffhq/FFHQ_512_4995.zip",
    #                     "--gpus",
    #                     "1",
    #                     "--batch",
    #                     "4",
    #                     "--gamma",
    #                     "25",
    #                     "--rlhf_config_fn",
    #                     "/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/eg3d_rlhf_hpms_dm3.yaml",
    #                     "--gen_pose_cond",
    #                     "True",
    #                     "--mbstd-group",
    #                     "1",
    #                     "--tick",
    #                     "1",
    #                     "--resume",
    #                     "/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/pkl_pt/eg3d_1/ffhq512-128.pkl",
    #                     "--neural_rendering_resolution_final",
    #                     "128",
    #                     "--resume_kimg",
    #                     "2028"
    #                 ]
    # #dict_of_args={legacy_args[i]:legacy_args[i+1] for i in range(0,len(legacy_args),2)}

    dict_of_args = hydracfg.click_legacy_args
    outdir = hydracfg.click_legacy_args.outdir

    orig_legacy_args = {
        "cond": True,
        "mirror": False,
        "aug": "noaug",
        "resume": None,
        "freezed": 0,
        "p": 0.2,
        "target": 0.6,
        "batch_gpu": None,
        "cbase": 32768,
        "cmax": 512,
        "glr": None,
        "dlr": 0.002,
        "map-depth": 2,
        "mbstd-group": 4,
        "desc": None,
        # "metrics": ["fid50k_full"],
        "metrics": ["fid5k_partial"],  # change to partial AM 20122025 to save time thru train run
        "kimg": 25000,
        "tick": 4,
        "snap": 10,
        "seed": 0,
        "nobench": False,
        "workers": 3,
        "dry-run": False,
        "neural_rendering_resolution_initial": 64,
        "neural_rendering_resolution_final": None,
        "neural_rendering_resolution_fade_kimg": 1000,
        "blur_fade_kimg": 200,
        "gen_pose_cond": False,
        "c-scale": 1,
        "c-noise": 0,
        "gpc_reg_prob": 0.5,
        "gpc_reg_fade_kimg": 1000,
        "disc_c_noise": 0,
        "sr_noise_mode": "none",
        "resume_blur": False,
        "sr_num_fp16_res": 4,
        "g_num_fp16_res": 0,
        "d_num_fp16_res": 4,
        "sr_first_cutoff": 2,
        "sr_first_stopband": 2**2.1,
        "style_mixing_prob": 0,
        "sr-module": None,
        "density_reg": 0.25,
        "density_reg_every": 4,
        "density_reg_p_dist": 0.004,
        "reg_type": "l1",
        "decoder_lr_mul": 1,
        "rlhf_config_fn": None,
        "resume_kimg": 0,
    }

    renamed_orig = {}

    for k in orig_legacy_args.keys():
        renamed_orig[k.replace("-", "_")] = orig_legacy_args[k]  # G_reg_interval

    orig_legacy_args = renamed_orig
    orig_legacy_args.update(dict_of_args)

    c, desc, dry_run = main_legacy_click(**orig_legacy_args)  # pylint: disable=no-value-for-parameter

    # ------------------------------------------------------------------

    cfg = hydracfg

    outdir = cfg.click_legacy_args.outdir

    # Optional consistency regularisers (depth-map + LPIPS) need old_G / old_G_ema
    # references so loss.py can compare the tuned generator against the frozen one.
    if any(
        [
            cfg.rlhf_tune_hpms.lambda_dmap_l1 != 0.0,
            cfg.rlhf_tune_hpms.lambda_dmap_mse != 0.0,
            cfg.rlhf_tune_hpms.get("lambda_dmap_forward_mse", 0.0) != 0.0,
            cfg.rlhf_tune_hpms.lambda_lpips != 0.0,
        ]
    ):
        cfg.pretrained_modules.old_G = True
        cfg.pretrained_modules.old_G_ema = True

    if cfg.rlhf_tune_hpms.lambda_lpips != 0.0:
        cfg.pretrained_modules.LPIPS = True

    launch_training_rlhf(c=c, desc=desc, outdir=outdir, dry_run=dry_run, hydra_cfg=cfg)


# ----------------------------------------------------------------------------

#     import hydra
#     #from hydra import compose, initialize
#     from omegaconf import OmegaConf

#         # initialize the Hydra subsystem.
#         # This is needed for apps that cannot have a standard @hydra.main() entry point
#     hydra_cfg_path="training/cfg_rlhf_tune_AM"#/rlhf_tune.yaml"

#     if hydra_cfg_path is None:
#         assert False, 'hydra_cfg_path is None'
#         #/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/eg3d/training/cfg_rlhf_tune_AM/rlhf_tune.yaml
#         #/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/eg3d/training/cfg_rlhf_tune_AM/rlhf_tune.yaml
#     hydra.initialize(config_path=hydra_cfg_path, job_name="rlhf_tune",version_base=None)
#     cfg = hydra.compose("rlhf_tune.yaml")#, overrides=["db=mysql", "db.user=${oc.env:USER}"])
#     print(OmegaConf.to_yaml(cfg, resolve=True))
# #@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")

# Launch.
# Launch.


# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main_hydra()


# ----------------------------------------------------------------------------


# launch_training_rlhf(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run,hydra_cfg=cfg)
