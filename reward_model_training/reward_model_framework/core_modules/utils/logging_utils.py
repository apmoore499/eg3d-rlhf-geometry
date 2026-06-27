import os
import sys
from typing import Any, Dict
import autoroot  # noqa: F401
import shutil


# ----------------------------------

import hydra
import matplotlib.pyplot as plt

# import models
import numpy as np
import pandas as pd
import torch
import mrcfile
import tqdm

from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import uniform_filter

import wandb

import logging

# back_up.py
import pathlib
from pathlib import Path

# some_module.py
from lightning.pytorch.callbacks import Callback
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from core_modules.utils import pylogger_c
from core_modules.utils.rwd_model_utils import log_best_worst_meshes

log = pylogger_c.RankedLogger(__name__, rank_zero_only=True)

# ------------------


def stack_ims_horizontally(image_names):
    # Load images
    images = [Image.open(name) for name in image_names]

    # Calculate total width and maximum height
    total_width = sum(im.width for im in images)
    max_height = max(im.height for im in images)

    # Create a new image with the appropriate size
    new_im = Image.new("RGB", (total_width, max_height))

    # Paste images into the new image
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.width

    return new_im


def stack_ims_vertically_with_labels(image_names, labels):
    # Assuming a common font size
    font_size = 20
    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()  # size=80)

    # Load images
    images = [(Image.open(name), label) for name, label in zip(image_names, labels)]

    # Calculate the maximum width (considering label space) and total height
    max_width = max(im.width for im, _ in images) + 200  # Adding space for labels on each image
    total_height = sum(im.height for im, _ in images)

    # Create a new image with the appropriate size
    new_im = Image.new("RGB", (max_width, total_height), (255, 255, 255))  # White background

    # Paste images and add labels vertically
    y_offset = 0
    for im, label in images:
        # Create label area
        label_area_width = max_width - im.width
        label_area_height = im.height
        label_area = Image.new("RGB", (label_area_width, label_area_height), (255, 255, 255))
        d = ImageDraw.Draw(label_area)
        # This attempts to center the label vertically in the available label area

        text_height = d.textlength(label, font=font)
        d.text((10, (label_area_height - text_height) / 2), label, fill=(0, 0, 0), font=font)

        # Paste label area to the left of the image
        new_im.paste(label_area, (0, y_offset))
        # Paste image next to its label
        new_im.paste(im, (label_area_width, y_offset))

        # Update y_offset for the next image
        y_offset += im.height

    return new_im


# Example usage
# image_names = ['image1.jpg', 'image2.jpg', 'image3.jpg']
# labels = ['Label 1', 'Label 2', 'Label 3']
# result_image = stack_ims_vertically_with_labels(image_names, labels)
# result_image.show()


def stack_ims_horizontally_with_labels(image_names, labels):
    # Assuming a common font size
    font_size = 20
    # Load a font - this path might need to be adjusted based on the system or environment
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Load images and calculate total width (with added space for labels) and maximum height
    images = [(Image.open(name), label) for name, label in zip(image_names, labels)]
    total_width = sum(im.width for im, _ in images) + 200 * len(images)  # Adding space for labels
    max_height = max(im.height for im, _ in images)

    # Create a new image with the appropriate size
    new_im = Image.new("RGB", (total_width, max_height), (255, 255, 255))  # White background

    # Paste images and add labels
    x_offset = 0
    for im, label in images:
        # Adjusting offset for label space
        label_area = Image.new("RGB", (200, max_height), (255, 255, 255))
        d = ImageDraw.Draw(label_area)
        # This attempts to center the label vertically
        text_height = d.textsize(label, font=font)[1]
        d.text((10, (max_height - text_height) / 2), label, fill=(0, 0, 0), font=font)

        new_im.paste(label_area, (x_offset, 0))
        x_offset += 200  # Move offset after the label area
        new_im.paste(im, (x_offset, 0))
        x_offset += im.width

    return new_im


def activation_maps_across_heads(trainer, comparison_meshes_stack, comparison_meshes_idx, winning_seed_mesh, swaporder=False):
    comparison_stack = comparison_meshes_stack[comparison_meshes_idx, ...]
    comp_dset = torch.utils.data.TensorDataset(comparison_stack)
    dl_comp = torch.utils.data.DataLoader(comp_dset, batch_size=4, shuffle=False, drop_last=False)

    activations_mean = None
    totlen = 0

    for batch in tqdm.tqdm(iter(dl_comp)):
        batch_gfeature_stack = batch[0]

        if len(batch_gfeature_stack.shape) == 3:
            batch_gv_test = winning_seed_mesh.reshape(1, batch_gfeature_stack.shape[1], batch_gfeature_stack.shape[2]).expand(batch_gfeature_stack.shape[0], -1, -1)
        elif len(batch_gfeature_stack.shape) == 2:
            batch_gv_test = winning_seed_mesh.reshape(1, batch_gfeature_stack.shape[1]).expand(batch_gfeature_stack.shape[0], -1)

        if not swaporder:
            activations_fwd = trainer.model.get_activation_maps_from_embedded(batch_gv_test, batch_gfeature_stack)["attn_seq1_maps"]
        else:
            activations_fwd = trainer.model.get_activation_maps_from_embedded(batch_gfeature_stack, batch_gv_test)["attn_seq1_maps"]

        activations_fwd_mean = torch.mean(activations_fwd, 0, keepdim=True)

        if activations_mean is None:
            activations_mean = torch.zeros_like(activations_fwd_mean)

        activations_mean += activations_fwd_mean
        totlen += activations_fwd_mean.shape[0]

    activations_mean = activations_mean / totlen
    return activations_mean


# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html#Transformer-Encoder
# NB modified for pairwise comparisons
def plot_attention_maps_seq1_seq2(input_data, attn_maps, idx=0):
    if input_data is not None:
        input_data = input_data[idx].detach().cpu().numpy()
    else:
        input_data = np.arange(attn_maps[0][idx].shape[-1])

    attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]

    # attn_mapse = [m[0].detach().cpu().numpy() for m in attn_maps]

    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 15  # if num_heads == 1 else 3
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads * fig_size, num_layers * fig_size))
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    for row in range(num_layers):
        for column in range(num_heads):
            ax[row][column].imshow(attn_maps[row][column], origin="lower", vmin=0)
            ax[row][column].set_xticks(list(range(seq_len)))
            ax[row][column].set_xticklabels(input_data.tolist(), fontsize=5)
            ax[row][column].set_yticks(list(range(seq_len)))
            ax[row][column].set_yticklabels(input_data.tolist(), fontsize=5)
            ax[row][column].set_title("Layer %i, Head %i" % (row + 1, column + 1))
    # fig.subplots_adjust(hspace=0.5)
    fig.tight_layout()

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image_from_plot


def visualise_heatmaps_for_nn_patches(activation_maps, heads, nn, plot_v1=True):
    # Adjust the reshape to account for the heads dimension
    activation_map_reshaped = activation_maps.reshape((heads, nn, nn, nn, nn))

    # Define the kernel size
    kernel_size = 5

    # Prepare heatmaps for all heads
    heatmaps = np.zeros((heads, nn, nn))

    for h in range(heads):
        for i in range(32):
            for j in range(32):
                if plot_v1:
                    # Compute heatmap for v1 across all heads
                    heatmaps[h, i, j] = uniform_filter(activation_map_reshaped[h, i, j, :, :], size=kernel_size, mode="nearest").mean()
                else:
                    # Compute heatmap for v2 across all heads
                    heatmaps[h, i, j] = uniform_filter(activation_map_reshaped[h, :, :, i, j], size=kernel_size, mode="nearest").mean()

    # Plotting
    fig, axs = plt.subplots(1, heads, figsize=(12, 4))

    for h in range(heads):
        axs[h].imshow(heatmaps[h], cmap="viridis", interpolation="nearest")
        axs[h].set_title(f"Head {h + 1}")
        axs[h].axis("off")

    fig.tight_layout()

    # Draw the figure in memory and retrieve the image as an array
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)  # Close the figure to prevent it from displaying in the notebook/output

    return image_from_plot


log = pylogger_c.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


class LogImagesCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation batch ends."""

        # newdict_keys=pl_module.validation_step_preds_all[-1].keys()
        # collate_dict={k:[] for k in newdict_keys}

        dset_version = trainer.datamodule.dset_version

        print("------------------------------------")
        print("dset version:")
        print(dset_version)
        print("------------------------------------")

        newdict_keys = pl_module.validation_step_preds_all[-1].keys()
        global_features = {}

        for ll in pl_module.validation_step_preds_all:
            global_features.update(ll["dict_of_global_feature"])

        global_features = {int(k): v.cpu().numpy().reshape(-1) for k, v in global_features.items()}
        # global_features = {int(k): v.cpu().numpy().view(-1) for k, v in global_features.items()}

        # wandb.log({"val/embeddings/global_embeddings": gf,"epoch": pl_module.current_epoch,}, commit=False)#, on_epoch=True, on_step=False)

        df_of = pd.DataFrame.from_dict(global_features, orient="index")

        df_of["epoch"] = pl_module.current_epoch

        import pathlib

        sdir = pathlib.Path(os.environ["RWD_MODELS_DIR"]) / str(wandb.run.id)
        sdir.mkdir(parents=True, exist_ok=True)
        o_fn = sdir.joinpath(f"val_gv_epoch_{pl_module.current_epoch}.csv")
        df_of.to_csv(o_fn)

        # wandb.log(
        #     {
        #         "val/embeddings/global_embeddings": wandb.Table(df_of),
        #         "epoch": pl_module.current_epoch,
        #     },
        #     commit=False,
        # )  # , on_epoch=True, on_step=False)

        # -------------------------------------------

        if pl_module.hparams.loss.lambda_pairs != 0.0:
            # pairs win / lose

            pred_logits_list_first = []
            pred_logits_list_second = []

            for ll in pl_module.validation_step_preds_all:
                logits_pairs = ll["pred_logits"]
                lp_size = logits_pairs.shape[0]

                lps = int(lp_size / 2)
                val_logits_orig = logits_pairs[:lps, :]
                val_logits_reversed = logits_pairs[lps:, :]

                first_pred = torch.nn.functional.softmax(val_logits_orig, -1)
                second_pred = torch.nn.functional.softmax(val_logits_reversed, -1)

                # second_pred=second_pred.flip(-1)\

                # fps=(first_pred+second_pred)/2  ##change from multiply to add, take mean

                # if len(fps.shape)==3:
                #    fps=fps.mean(1)

                pred_logits_list_first.append(first_pred)
                pred_logits_list_second.append(second_pred)

            # fps=torch.vstack(pred_logits_list_first)

            fps_first = torch.vstack(pred_logits_list_first)

            fps_second = torch.vstack(pred_logits_list_second)

            n_correct_val = torch.sum(fps_first[:, 0] > fps_first[:, 1]).detach().cpu().numpy()
            pc_correct_val = n_correct_val / fps_first.shape[0]

            pc_correct_val_first = pc_correct_val

            n_correct_val = torch.sum(fps_second[:, 0] > fps_second[:, 1]).detach().cpu().numpy()
            pc_correct_val = n_correct_val / fps_second.shape[0]

            pc_correct_val_second = pc_correct_val

            log.info(f"Paired rwd correct first/second: {pc_correct_val_first:.4f}/{pc_correct_val_second:.4f}")

            pc_correct_val = (pc_correct_val_first + pc_correct_val_second) / 2
            # print(pc_correct_val)

            current_pred = {"type": "pairs", "epoch": pl_module.current_epoch, "pc_correct": pc_correct_val, "data": "val"}

            pl_module.epo_correct_list.append(current_pred)

            # pl_module.log(f"val/pairs/pc_correct",
            #             pc_correct_val,
            #             on_step=False,
            #             on_epoch=True,
            #             prog_bar=True,
            #             batch_size=1)

            wandb.log(
                {
                    "val/pairs/pc_correct": pc_correct_val,
                    "epoch": pl_module.current_epoch,
                }
            )  # , commit=False)#, on_epoch=True, on_step=False)

            # print('paired rwd correct')
            # print(pc_correct_val)

        # -----------------------------------------------------------------------

        if pl_module.hparams.loss.lambda_BT != 0.0:
            # BRADLEY-TERRY LOSSES

            newdict_keys = pl_module.validation_step_preds_all[-1].keys()

            BT_pairwise = []
            for llll in pl_module.validation_step_preds_all:
                llcomp = llll["BT_comparison_values"]
                # if len(llcomp)==1 and type(llcomp[0])==list:
                #    llcomp=llcomp[0]
                BT_pairwise.append(torch.hstack(llcomp).flatten())

            all_BT_comparison = torch.hstack(BT_pairwise).flatten()
            pc_correct_val_BT = torch.mean((all_BT_comparison > 0.5).float()).detach().cpu().numpy()
            wandb.log(
                {
                    "val/BT/pc_correct": pc_correct_val_BT,
                    "epoch": pl_module.current_epoch,
                }
            )  # , commit=False)#, on_epoch=True, on_step=False)

            # len(pl_module.validation_step_preds_all[0]['BT_comparison_values'][4])

            current_pred = {"type": "BT", "epoch": pl_module.current_epoch, "pc_correct": pc_correct_val_BT, "data": "val"}

            pl_module.epo_correct_list.append(current_pred)

            BT_vals = {}

            for ll in pl_module.validation_step_preds_all:
                BT_vals.update(ll["rwds_dist_dict_BT"])

            BT_values = {int(k): v.cpu().numpy() for k, v in BT_vals.items()}

            BT_df = pd.DataFrame.from_dict(BT_values, orient="index").reset_index(drop=False)

            BT_df.columns = ["seed", "rwd_val"]
            BT_df["rwd_val"] = pd.Series([x for x in BT_df["rwd_val"]])
            # BT_values=torch.hstack([v for v in BT_vals.values()]).detach().cpu().numpy()
            # OrderedSeeds=torch.hstack([v for v in BT_vals.values()]).detach().cpu().numpy()
            BT_df = BT_df.groupby("seed").mean().reset_index(drop=False)

            wandb.log(
                {
                    "val/BT/vals_histogram": wandb.Histogram(BT_df.rwd_val),
                    "epoch": pl_module.current_epoch,
                }
            )  # , on_epoch=True, on_step=False)
            wandb.log(
                {
                    "val/BT/vals": BT_values,
                    "epoch": pl_module.current_epoch,
                }
            )  # , on_epoch=True, on_step=False)

            # BT_df_rwd=torch.hstack(BT_df.rwd_val.values)

            save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            ddir_func = trainer.datamodule.data_val.ddir_func
            seed_func = trainer.datamodule.data_val.seed_func

            log.info(f"BT rwd correct: {pc_correct_val_BT}")

            BT_df["epoch"] = pl_module.current_epoch

            wandb.log(
                {
                    f"val/dset_{dset_version}/BT_rwd_df/": wandb.Table(dataframe=BT_df),
                    "epoch": pl_module.current_epoch,
                }
            )

        # -----------------------------------------------------------------------

        # SCALAR LOSSES

        if pl_module.hparams.loss.lambda_scalar_rwd != 0.0:
            newdict_keys = pl_module.validation_step_preds_all[-1].keys()

            scalar_rwd_pairwise = []
            for ll in pl_module.validation_step_preds_all:
                scalar_rwd_pairwise.append(torch.hstack(ll["sigmoid_comparison_values"]).flatten())

            all_scalar_rwd_comparison = torch.hstack(scalar_rwd_pairwise).flatten()
            pc_correct_val_scalar = torch.mean((all_scalar_rwd_comparison > 0.0).float()).detach().cpu().numpy()

            scalar_rwd_vals = {}

            for ll in pl_module.validation_step_preds_all:
                scalar_rwd_vals.update(ll["scalar_rwds_dist_dict_scalar"])

            scalar_rwd_values = {int(k): v.cpu().numpy() for k, v in scalar_rwd_vals.items()}
            scalar_rwd_df = pd.DataFrame.from_dict(scalar_rwd_values, orient="index").reset_index(drop=False)
            scalar_rwd_df.columns = ["seed", "rwd_val"]
            # Assuming df is your DataFrame and 'tensor_column' is the name of the column with the tensors
            scalar_rwd_df["rwd_val"] = pd.Series([x for x in scalar_rwd_df["rwd_val"]])

            scalar_rwd_df = scalar_rwd_df.groupby("seed").mean().reset_index(drop=False)
            log.info(f"scalar_rwd_corect: {pc_correct_val_scalar}")

            current_pred = {"type": "scalar", "epoch": pl_module.current_epoch, "pc_correct": pc_correct_val_scalar, "data": "val"}
            pl_module.epo_correct_list.append(current_pred)
            scalar_rwd_df["epoch"] = pl_module.current_epoch

            wandb.log(
                {
                    "val/scalar_rwd/vals_histogram": wandb.Histogram(scalar_rwd_df.rwd_val),
                    "epoch": pl_module.current_epoch,
                }
            )  # , on_epoch=True, on_step=False)
            wandb.log(
                {
                    "val/scalar_rwd/vals": scalar_rwd_values,
                    "epoch": pl_module.current_epoch,
                }
            )  # , on_epoch=True, on_step=False)
            wandb.log(
                {
                    "val/scalar_rwd/pc_correct": pc_correct_val_scalar,
                    "epoch": pl_module.current_epoch,
                }
            )  # , commit=False)#, on_epoch=True, on_step=False)
            wandb.log(
                {
                    f"val/dset_{dset_version}/scalar_rwd_df/": wandb.Table(dataframe=scalar_rwd_df),
                    "epoch": pl_module.current_epoch,
                }
            )

        if "reconstruction_errors_dict_unet3d" in pl_module.validation_step_preds_all[0].keys() and len(pl_module.validation_step_preds_all[0]["reconstruction_errors_dict_unet3d"]) > 0:
            reconstruction_errors_dict_unet3d = {}

            for ll in pl_module.validation_step_preds_all:
                reconstruction_errors_dict_unet3d.update(ll["reconstruction_errors_dict_unet3d"])

            reconstruction_errors_dict_unet3d = {int(k): v.cpu().numpy() for k, v in reconstruction_errors_dict_unet3d.items()}
            reconstruction_errors_dict_unet3d = pd.DataFrame.from_dict(reconstruction_errors_dict_unet3d, orient="index").reset_index(drop=False)
            reconstruction_errors_dict_unet3d.columns = ["seed", "recon_loss"]
            reconstruction_errors_dict_unet3d = reconstruction_errors_dict_unet3d.groupby("seed").mean().reset_index(drop=False)
            log.info(f"mean recon error over all example in val set: {reconstruction_errors_dict_unet3d.recon_loss.mean()}")
            reconstruction_errors_dict_unet3d["epoch"] = pl_module.current_epoch
            wandb.log(
                {
                    f"val/dset_{dset_version}/reconstruction_errors_dict_unet3d/": wandb.Table(dataframe=reconstruction_errors_dict_unet3d),
                    "epoch": pl_module.current_epoch,
                }
            )

            # then also get min and max seed, reconstrtuct it...

        if trainer.current_epoch in [0, 1, 2, 3, 4, trainer.max_epochs - 1, trainer.max_epochs] and type(trainer.model).__name__ == "Conv3DNetworkEnsemble" and not trainer.model.return_global_only:
            # get min and max seeds....
            reconstruction_errors_dict_unet3d = reconstruction_errors_dict_unet3d.sort_values(by="recon_loss", ascending=False)
            reconstruction_errors_dict_unet3d = reconstruction_errors_dict_unet3d[reconstruction_errors_dict_unet3d.seed > 0]

            best_seed = reconstruction_errors_dict_unet3d.seed.head(1).item()
            best_example = trainer.datamodule.data_test.return_single_data(best_seed, augmentation=torch.nn.Identity()).cuda().unsqueeze(0)

            worst_seed = reconstruction_errors_dict_unet3d.seed.tail(1).item()
            worst_example = trainer.datamodule.data_test.return_single_data(worst_seed, augmentation=torch.nn.Identity()).cuda().unsqueeze(0)

            trainer.model.return_global_only = False

            ld = trainer.log_dir
            os.makedirs(ld, exist_ok=True)

            bb = trainer.model.forward_to_global_feature_vec(best_example)

            orig_x = bb["orig_x"][0].to(torch.float16)
            recon_x = bb["recon_x"][0].to(torch.float16)

            if trainer.model.hparams.loss.normalise_recon == True:
                orig_max = orig_x.max()
                orig_min = orig_x.min()
                rkx = recon_x  # [k]
                rkx = (rkx - rkx.min()) / (rkx.max() - rkx.min()) * (orig_max - orig_min) + orig_min

                recon_x = rkx

            orig_x_best_fn = os.path.join(ld, f"val_best_reconstructed_example_ORIGINAL_epoch_{trainer.current_epoch}_run_{wandb.run.id}.pt")
            recon_x_best_fn = os.path.join(ld, f"val_best_reconstructed_example_RECONSTRUCTED_epoch_{trainer.current_epoch}_run_{wandb.run.id}.pt")

            torch.save(obj=orig_x, f=orig_x_best_fn)
            torch.save(obj=recon_x, f=recon_x_best_fn)

            with mrcfile.new_mmap(orig_x_best_fn.replace(".pt", ".mrc"), overwrite=True, shape=orig_x.squeeze(0, 1).shape, mrc_mode=2) as mrc:
                mrc.data[:] = orig_x.squeeze(0, 1).cpu().numpy()

            with mrcfile.new_mmap(recon_x_best_fn.replace(".pt", ".mrc"), overwrite=True, shape=recon_x.squeeze(0, 1).shape, mrc_mode=2) as mrc:
                mrc.data[:] = recon_x.squeeze(0, 1).cpu().numpy()

            ww = trainer.model.forward_to_global_feature_vec(worst_example)

            orig_x = ww["orig_x"][0].to(torch.float16)
            recon_x = ww["recon_x"][0].to(torch.float16)

            if trainer.model.hparams.loss.normalise_recon == True:
                orig_max = orig_x.max()
                orig_min = orig_x.min()
                rkx = recon_x  # [k]
                rkx = (rkx - rkx.min()) / (rkx.max() - rkx.min()) * (orig_max - orig_min) + orig_min

                recon_x = rkx

            orig_x_best_fn = os.path.join(ld, f"val_worst_reconstructed_example_ORIGINAL_epoch_{trainer.current_epoch}_run_{wandb.run.id}.pt")
            recon_x_best_fn = os.path.join(ld, f"val_worst_reconstructed_example_RECONSTRUCTED_epoch_{trainer.current_epoch}_run_{wandb.run.id}.pt")

            torch.save(obj=orig_x, f=orig_x_best_fn)
            torch.save(obj=recon_x, f=recon_x_best_fn)

            with mrcfile.new_mmap(orig_x_best_fn.replace(".pt", ".mrc"), overwrite=True, shape=orig_x.squeeze(0, 1).shape, mrc_mode=2) as mrc:
                mrc.data[:] = orig_x.squeeze(0, 1).cpu().numpy()

            with mrcfile.new_mmap(recon_x_best_fn.replace(".pt", ".mrc"), overwrite=True, shape=recon_x.squeeze(0, 1).shape, mrc_mode=2) as mrc:
                mrc.data[:] = recon_x.squeeze(0, 1).cpu().numpy()

        return

    def on_test_epoch_end(self, trainer, pl_module):
        dset_version = trainer.datamodule.dset_version

        print("------------------------------------")
        print("dset version:")
        print(dset_version)
        print("------------------------------------")

        newdict_keys = pl_module.test_step_preds_all[-1].keys()
        global_features = {}

        for ll in pl_module.test_step_preds_all:
            global_features.update(ll["dict_of_global_feature"])

        global_features = {int(k): v.cpu().numpy() for k, v in global_features.items()}

        gf_seeds = [k for k in global_features.keys()]

        # wandb.log({"test/embeddings/global_embeddings": gf,"epoch": pl_module.current_epoch,}, commit=False)#, on_epoch=True, on_step=False)

        wandb.log(
            {
                "test/embeddings/global_embeddings": global_features,  # pd.DataFrame.from_dict(global_features,orient='index'),
                "epoch": pl_module.current_epoch,
            },
            commit=False,
        )  # , on_epoch=True, on_step=False)

        # -------------------------------------------

        if pl_module.hparams.loss.lambda_pairs != 0.0:
            # pairs win / lose

            pred_logits_list = []

            for ll in pl_module.test_step_preds_all:
                logits_pairs = ll["pred_logits"]
                lp_size = logits_pairs.shape[0]

                lps = int(lp_size / 2)
                val_logits_orig = logits_pairs[:lps, :]
                val_logits_reversed = logits_pairs[lps:, :]

                first_pred = torch.nn.functional.softmax(val_logits_orig, -1)
                second_pred = torch.nn.functional.softmax(val_logits_reversed, -1)

                # second_pred=second_pred.flip(-1)
                fps = (first_pred + second_pred) / 2

                if len(fps.shape) == 3:
                    fps = fps.mean(1)

                pred_logits_list.append(fps)

            fps = torch.vstack(pred_logits_list)

            n_correct_test = torch.sum(fps[:, 0] > fps[:, 1]).detach().cpu().numpy()
            pc_correct_test = n_correct_test / fps.shape[0]

            # test_step_preds_all

            wandb.log(
                {
                    "test/pairs/pc_correct": pc_correct_test,
                    "epoch": pl_module.current_epoch,
                }
            )  # , commit=False)#, on_epoch=True, on_step=False)

            print("paired rwd correct")
            print(pc_correct_test)

            current_pred = {"type": "pairs", "epoch": pl_module.current_epoch, "pc_correct": pc_correct_test, "data": "test", "model_weights": pl_module.weights_type, "dset_version": dset_version}

            # current_pred={"type":"pairs","epoch":pl_module.current_epoch,"pc_correct":pc_correct_test,"type":"test"}

            # pl_module.epo_correct_list.append(current_pred)
            pl_module.test_correct_list.append(current_pred)

            print("calculating n correct pairs for reference meshes")

            # here you can calculate loss with respect to pairs from the dloader.................
            N_REFERENCE_SEEDS = 200
            reference_seeds = torch.tensor([120000 + i for i in range(N_REFERENCE_SEEDS)])

            ss = reference_seeds[0]  # get the init shape of dataaa
            COMPARE_TO_REFERENCE_SEEDS = False  # very long time to go...
            if trainer.datamodule.dset_version == "third" and COMPARE_TO_REFERENCE_SEEDS:
                do = trainer.datamodule.data_test.return_single_data(ss, augmentation=torch.nn.Identity()).cuda().unsqueeze(0)

                out_shape = do.shape

                NO_SQUEEZE = False
                if len(out_shape) == 6:
                    NO_SQUEEZE = True

                bsize_comp = 8
                import tqdm

                gfeature_stack = []
                # return the data items and run forward pass thru them
                trainer.model.return_global_only = True

                with torch.no_grad():
                    for slist in tqdm.tqdm(reference_seeds.split(bsize_comp)):
                        if NO_SQUEEZE:
                            data_oute = [trainer.datamodule.data_test.return_single_data(s, augmentation=torch.nn.Identity()).cuda() for s in slist]

                        else:
                            data_oute = [trainer.datamodule.data_test.return_single_data(s, augmentation=torch.nn.Identity()).cuda().unsqueeze(0) for s in slist]

                        cat_data = torch.cat(data_oute, 0)

                        if trainer.model.external is not None:
                            cat_data = trainer.model.external(cat_data)
                        if len(cat_data.shape) == 4:
                            cat_data = cat_data.squeeze(1)

                        gfeatures = trainer.model.forward_to_global_feature_vec(cat_data)

                        # if type(gfeatures)==dict:
                        #    gfeatures_d=[g['feature_vec'] for g in gfeatures ]
                        gfeature_stack.append(gfeatures)

                gfeature_stack = torch.cat(gfeature_stack, 0)

                refshape = gfeature_stack[0].unsqueeze(0)

                n_correct_seed_dict = {}
                bce_overall_seed_dict = {}
                import numpy as np

                with torch.no_grad():
                    for s in tqdm.tqdm(gf_seeds):
                        if len(refshape.shape) == 3:
                            gv_test = global_features[s].view_as(refshape).expand(N_REFERENCE_SEEDS, -1, -1)

                        elif len(refshape.shape) == 2 and type(global_features[s]) == torch.Tensor:
                            gv_test = global_features[s].view_as(refshape).expand(N_REFERENCE_SEEDS, -1)

                        elif len(refshape.shape) == 2 and type(global_features[s]) == np.ndarray:
                            gv_test = torch.from_numpy(global_features[s]).view_as(refshape).expand(N_REFERENCE_SEEDS, -1)

                        # use some sort of datloader here maybe

                        gvt = torch.utils.data.TensorDataset(gv_test, gfeature_stack)

                        dl = torch.utils.data.DataLoader(gvt, batch_size=8, shuffle=False, drop_last=False)

                        fps_list = []
                        for batch in iter(dl):
                            batch_gv_test = batch[0]
                            batch_gfeature_stack = batch[1]
                            pred1 = trainer.model.forward_from_cat_global_vectors(batch_gv_test.cuda(), batch_gfeature_stack.cuda(), with_softmax=False)
                            pred2 = trainer.model.forward_from_cat_global_vectors(batch_gfeature_stack.cuda(), batch_gv_test.cuda(), with_softmax=False)

                            first_pred = torch.nn.functional.softmax(pred1, -1)
                            second_pred = torch.nn.functional.softmax(pred2, -1)

                            second_pred = second_pred.flip(-1)
                            fps = (first_pred + second_pred) / 2

                            if len(fps.shape) == 3:
                                fps = fps.mean(1)

                            fps_list.append(fps)
                        fps = torch.vstack(fps_list)

                        # TEST_OFFSET=self.config.data.dset_dict.margin_for_offset_testing

                        # get the yaml
                        save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

                        cfg_fn = os.path.join(save_dir, ".hydra", "config.yaml")
                        import omegaconf

                        cfg = omegaconf.OmegaConf.load(cfg_fn)

                        TEST_OFFSET = cfg.data.dset_dict.margin_for_offset_testing

                        # try setting margin might be more interesting....say we want like 0.9pc win. or 0.8 for starters...
                        # mult by 2 to capture the best ones of them eall..............
                        n_correct_test = torch.sum(fps[:, 0] > fps[:, 1] + TEST_OFFSET * 2).detach().cpu().numpy()

                        # if n_correct_test>0:
                        n_correct_seed_dict[s] = n_correct_test

                        print(f"n_correct test, w offset {TEST_OFFSET}: {n_correct_test}\tof {gfeature_stack.shape[0]}")

                # now, we will try to save the highiest / lowest ranked activation maps

                new_out_dir = os.path.join(save_dir, "trans_acts")

                os.makedirs(new_out_dir, exist_ok=True)

                # margin_for_testing

                pairs_df = pd.DataFrame.from_dict(n_correct_seed_dict, orient="index").reset_index(drop=False)

                pairs_df.columns = ["seed", "n_wins"]

                pairs_df["margin"] = TEST_OFFSET
                wandb.log(
                    {
                        "test/pairs/vals_histogram": wandb.Histogram(pairs_df.n_wins),
                        "epoch": pl_module.current_epoch,
                    }
                )  # , on_epoch=True, on_step=False)
                wandb.log(
                    {
                        "test/pairs/vals": n_correct_seed_dict,
                        "epoch": pl_module.current_epoch,
                    }
                )  # , on_epoch=True, on_step=False)
                wandb.log(
                    {
                        f"test/dset_{dset_version}/pairs_df/": wandb.Table(dataframe=pairs_df),
                        "epoch": pl_module.current_epoch,
                    }
                )

                # as rwd_models
                if trainer.model.type == aw98_transformer:
                    # get activation maps of top 10 wins.
                    winning_seeds = []
                    if pairs_df[pairs_df.n_wins > 0].shape[0] > 0:
                        winning_seeds = pairs_df[pairs_df.n_wins > 0].sort_values(by="n_wins", ascending=False).head(4)["seed"]

                    for s in winning_seeds:
                        with torch.no_grad():
                            if len(refshape.shape) == 3:
                                gv_test = global_features[s].view_as(refshape).expand(N_REFERENCE_SEEDS, -1, -1)

                            elif len(refshape.shape) == 2:
                                gv_test = global_features[s].view_as(refshape).expand(N_REFERENCE_SEEDS, -1)

                            gvt = torch.utils.data.TensorDataset(gv_test, gfeature_stack)

                            dl = torch.utils.data.DataLoader(gvt, batch_size=8, shuffle=False, drop_last=False)

                            fps_list = []
                            for batch in iter(dl):
                                batch_gv_test = batch[0]
                                batch_gfeature_stack = batch[1]

                                pred1 = trainer.model.forward_from_cat_global_vectors(batch_gv_test, batch_gfeature_stack, with_softmax=False).detach().cpu()
                                pred2 = trainer.model.forward_from_cat_global_vectors(batch_gfeature_stack, batch_gv_test, with_softmax=False).detach().cpu()

                                first_pred = torch.nn.functional.softmax(pred1, -1)
                                second_pred = torch.nn.functional.softmax(pred2, -1)

                                second_pred = second_pred.flip(-1)
                                fps = (first_pred + second_pred) / 2

                                if len(fps.shape) == 3:
                                    fps = fps.mean(1)

                                fps_list.append(fps)

                            fps = torch.vstack(fps_list)

                            # get the yaml
                            save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

                            cfg_fn = os.path.join(save_dir, ".hydra", "config.yaml")
                            import omegaconf

                            cfg = omegaconf.OmegaConf.load(cfg_fn)

                            TEST_OFFSET = cfg.data.dset_dict.margin_for_offset_testing

                            # try setting margin might be more interesting....say we want like 0.9pc win. or 0.8 for starters...
                            n_correct_test = torch.sum(fps[:, 0] > fps[:, 1] + TEST_OFFSET * 2).detach().cpu().numpy()

                            # n_correct_seed_dict[s]=n_correct_test

                            # print(f'n_correct test, w offset {TEST_OFFSET}: {n_correct_test}\tof {gfeature_stack.shape[0]}')

                            # plotting...
                            PLOT_ACTIVATIONS = True
                            if PLOT_ACTIVATIONS:
                                if n_correct_test > 0:
                                    win_idx = torch.where(fps[:, 0] > fps[:, 1] + TEST_OFFSET * 2)[0].detach().cpu().numpy()
                                    lose_idx = torch.where(fps[:, 0] <= fps[:, 1] + TEST_OFFSET * 2)[0].detach().cpu().numpy()

                                    for swaporder in [False]:  # ,True]:
                                        activations_win_mean = activation_maps_across_heads(trainer=trainer, comparison_meshes_stack=gfeature_stack, comparison_meshes_idx=win_idx, winning_seed_mesh=global_features[s], swaporder=swaporder)
                                        activations_lose_mean = activation_maps_across_heads(trainer=trainer, comparison_meshes_stack=gfeature_stack, comparison_meshes_idx=lose_idx, winning_seed_mesh=global_features[s], swaporder=swaporder)

                                        # new_out_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

                                        if activations_win_mean.shape[-1] == 1024:
                                            activations_win_minus_lose = activations_win_mean - activations_lose_mean

                                            for plot_v1 in [False]:  # ,True]:
                                                win_minus_lose_heatmap = visualise_heatmaps_for_nn_patches(activations_win_minus_lose.squeeze(0).cpu().numpy(), heads=8, nn=32, plot_v1=plot_v1)
                                                out_fn_win_minus_lose = os.path.join(new_out_dir, f"s_{s}_correct_{n_correct_test}_win_minus_lose_act.jpg")
                                                Image.fromarray(win_minus_lose_heatmap).save(out_fn_win_minus_lose)
                                                plt.close("all")

                                                win_heatmap = visualise_heatmaps_for_nn_patches(activations_win_mean.squeeze(0).cpu().numpy(), heads=8, nn=32, plot_v1=plot_v1)
                                                out_fn_win = os.path.join(new_out_dir, f"s_{s}_correct_{n_correct_test}_win_mean_act.jpg")
                                                Image.fromarray(win_heatmap).save(out_fn_win)
                                                plt.close("all")

                                                lose_heatmap = visualise_heatmaps_for_nn_patches(activations_lose_mean.squeeze(0).cpu().numpy(), heads=8, nn=32, plot_v1=plot_v1)
                                                out_fn_lose = os.path.join(new_out_dir, f"s_{s}_correct_{n_correct_test}_lose_mean_act.jpg")
                                                # mps=plot_attention_maps_seq1_seq2(input_data=None,attn_maps=activations_lose_mean_list)
                                                Image.fromarray(lose_heatmap).save(out_fn_lose)
                                                plt.close("all")

                                                all_ims_fns = [out_fn_win, out_fn_lose, out_fn_win_minus_lose]
                                                combined_im = stack_ims_vertically_with_labels(all_ims_fns, labels=["win", "lose", "win_minus_lose"])
                                                out_fn_combined_fn = os.path.join(new_out_dir, f"s_{s}_correct_{n_correct_test}_combined_swaporder_{swaporder}_plotv1_{plot_v1}.jpg")
                                                combined_im.save(out_fn_combined_fn)

                                                for fn in all_ims_fns:
                                                    os.remove(fn)

                                        else:
                                            # subtract lose from win.............

                                            activations_win_minus_lose = [activations_win_mean - activations_lose_mean]

                                            out_fn_win_minus_lose = os.path.join(new_out_dir, f"s_{s}_correct_{n_correct_test}_win_minus_lose_act.jpg")
                                            mps = plot_attention_maps_seq1_seq2(input_data=None, attn_maps=activations_win_minus_lose)
                                            Image.fromarray(mps).save(out_fn_win_minus_lose)
                                            plt.close("all")

                                            out_fn_win = os.path.join(new_out_dir, f"s_{s}_correct_{n_correct_test}_win_mean_act.jpg")
                                            mps = plot_attention_maps_seq1_seq2(input_data=None, attn_maps=[activations_win_mean])
                                            Image.fromarray(mps).save(out_fn_win)
                                            plt.close("all")

                                            out_fn_lose = os.path.join(new_out_dir, f"s_{s}_correct_{n_correct_test}_lose_mean_act.jpg")
                                            mps = plot_attention_maps_seq1_seq2(input_data=None, attn_maps=[activations_lose_mean])
                                            Image.fromarray(mps).save(out_fn_lose)
                                            plt.close("all")

                                            all_ims_fns = [out_fn_win, out_fn_lose, out_fn_lose, out_fn_win_minus_lose]
                                            combined_im = stack_ims_vertically_with_labels(all_ims_fns, labels=["win", "lose", "win_minus_lose"])
                                            out_fn_combined_fn = os.path.join(new_out_dir, f"s_{s}_correct_{n_correct_test}_combined.jpg")
                                            combined_im.save(out_fn_combined_fn)

                                            for fn in all_ims_fns:
                                                os.remove(fn)

                # and then just log it

                save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

                ddir_func = trainer.datamodule.data_val.ddir_func
                seed_func = trainer.datamodule.data_val.seed_func

                thumbnail_size = trainer.thumbnail_size

                pairs_df = pairs_df.rename(columns={"n_wins": "rwd_val"})
                thumbnail_size = trainer.thumbnail_size
                log_best_worst_meshes(df=pairs_df, ddir_func=ddir_func, seed_func=seed_func, epoch=pl_module.current_epoch, save_dir=save_dir, n_meshes=10, remove_good=True, using_wandb=True, log_thumbnail_only=True, exp="test", comparison_type="pairs", dset_version=dset_version, thumbnail_size=thumbnail_size)

        # -----------------------------------------------------------------------

        if pl_module.hparams.loss.lambda_BT != 0.0:
            # BRADLEY-TERRY LOSSES

            newdict_keys = pl_module.test_step_preds_all[-1].keys()

            BT_pairwise = []
            for ll in pl_module.test_step_preds_all:
                BT_pairwise.append(torch.hstack(ll["BT_comparison_values"]).flatten())

            all_BT_comparison = torch.hstack(BT_pairwise).flatten()
            pc_correct_test_BT = torch.mean((all_BT_comparison > 0.5).float()).detach().cpu().numpy()
            wandb.log(
                {
                    "test/BT/pc_correct": pc_correct_test_BT,
                    "epoch": pl_module.current_epoch,
                }
            )  # , commit=False)#, on_epoch=True, on_step=False)

            BT_vals = {}

            for ll in pl_module.test_step_preds_all:
                BT_vals.update(ll["rwds_dist_dict_BT"])

            BT_values = {int(k): v.cpu().numpy() for k, v in BT_vals.items()}

            BT_df = pd.DataFrame.from_dict(BT_values, orient="index").reset_index(drop=False)

            BT_df.columns = ["seed", "rwd_val"]
            BT_df["rwd_val"] = pd.Series([x for x in BT_df["rwd_val"]])
            # BT_values=torch.hstack([v for v in BT_vals.values()]).detach().cpu().numpy()
            # OrderedSeeds=torch.hstack([v for v in BT_vals.values()]).detach().cpu().numpy()
            BT_df = BT_df.groupby("seed").mean().reset_index(drop=False)

            wandb.log(
                {
                    "test/BT/vals_histogram": wandb.Histogram(BT_df.rwd_val),
                    "epoch": pl_module.current_epoch,
                }
            )  # , on_epoch=True, on_step=False)
            wandb.log(
                {
                    "test/BT/vals": BT_values,
                    "epoch": pl_module.current_epoch,
                }
            )  # , on_epoch=True, on_step=False)

            save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

            ddir_func = trainer.datamodule.data_val.ddir_func
            seed_func = trainer.datamodule.data_val.seed_func

            thumbnail_size = trainer.thumbnail_size

            log_best_worst_meshes(df=BT_df, ddir_func=ddir_func, seed_func=seed_func, epoch=pl_module.current_epoch, save_dir=save_dir, n_meshes=10, remove_good=True, using_wandb=True, log_thumbnail_only=True, exp="test", comparison_type="BT", dset_version=dset_version, thumbnail_size=thumbnail_size)

            print("BT rwd correct")
            print(pc_correct_test_BT)

            current_pred = {"type": "BT", "epoch": pl_module.current_epoch, "pc_correct": pc_correct_test_BT, "data": "test", "model_weights": pl_module.weights_type, "dset_version": dset_version}
            pl_module.test_correct_list.append(current_pred)

            BT_df["epoch"] = pl_module.current_epoch

            wandb.log(
                {
                    f"test/dset_{dset_version}/BT_rwd_df/": wandb.Table(dataframe=BT_df),
                    "epoch": pl_module.current_epoch,
                }
            )

        # -----------------------------------------------------------------------

        # SCALAR LOSSES

        if pl_module.hparams.loss.lambda_scalar_rwd != 0.0:
            newdict_keys = pl_module.test_step_preds_all[-1].keys()

            scalar_rwd_pairwise = []
            for ll in pl_module.test_step_preds_all:
                scalar_rwd_pairwise.append(torch.hstack(ll["sigmoid_comparison_values"]).flatten())

            all_scalar_rwd_comparison = torch.hstack(scalar_rwd_pairwise).flatten()
            pc_correct_test_scalar = torch.mean((all_scalar_rwd_comparison > 0.0).float()).detach().cpu().numpy()

            scalar_rwd_vals = {}

            for ll in pl_module.test_step_preds_all:
                scalar_rwd_vals.update(ll["scalar_rwds_dist_dict_scalar"])

            scalar_rwd_values = {int(k): v.cpu().numpy() for k, v in scalar_rwd_vals.items()}

            scalar_rwd_df = pd.DataFrame.from_dict(scalar_rwd_values, orient="index").reset_index(drop=False)

            scalar_rwd_df.columns = ["seed", "rwd_val"]
            # Assuming df is your DataFrame and 'tensor_column' is the name of the column with the tensors
            scalar_rwd_df["rwd_val"] = pd.Series([x for x in scalar_rwd_df["rwd_val"]])

            scalar_rwd_df = scalar_rwd_df.groupby("seed").mean().reset_index(drop=False)

            save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

            ddir_func = trainer.datamodule.data_val.ddir_func
            seed_func = trainer.datamodule.data_val.seed_func
            thumbnail_size = trainer.thumbnail_size

            log_best_worst_meshes(df=scalar_rwd_df, ddir_func=ddir_func, seed_func=seed_func, epoch=pl_module.current_epoch, save_dir=save_dir, n_meshes=10, remove_good=True, using_wandb=True, log_thumbnail_only=True, exp="test", comparison_type="scalar_rwd", dset_version=dset_version, thumbnail_size=thumbnail_size)

            print("scalar rwd correct")
            print(pc_correct_test_scalar)

            scalar_rwd_df["epoch"] = pl_module.current_epoch

            # current_pred={"type":"scalar","epoch":pl_module.current_epoch,"pc_correct":pc_correct_test_scalar,"type":"test"}

            current_pred = {"type": "scalar", "epoch": pl_module.current_epoch, "pc_correct": pc_correct_test_scalar, "data": "test", "model_weights": pl_module.weights_type, "dset_version": dset_version}
            # current_pred={"type":"scalar","epoch":pl_module.current_epoch,"pc_correct":pc_correct_test_scalar,"data":"test","model_weights":pl_module.weights_type,"dset_version": dset_version }

            # pl_module.epo_correct_list.append(current_pred)

            pl_module.test_correct_list.append(current_pred)

            wandb.log(
                {
                    "test/scalar_rwd/vals_histogram": wandb.Histogram(scalar_rwd_df.rwd_val),
                    "epoch": pl_module.current_epoch,
                }
            )  # , on_epoch=True, on_step=False)
            wandb.log(
                {
                    "test/scalar_rwd/vals": scalar_rwd_values,
                    "epoch": pl_module.current_epoch,
                }
            )  # , on_epoch=True, on_step=False)
            wandb.log(
                {
                    "test/scalar_rwd/pc_correct": pc_correct_test_scalar,
                    "epoch": pl_module.current_epoch,
                }
            )  # , commit=False)#, on_epoch=True, on_step=False)
            wandb.log(
                {
                    f"test/dset_{dset_version}/scalar_rwd_df/": wandb.Table(dataframe=scalar_rwd_df),
                    "epoch": pl_module.current_epoch,
                }
            )

        # logging for reconstruction accuracies.....
        if "reconstruction_errors_dict_unet3d" in pl_module.test_step_preds_all[0].keys() and len(pl_module.test_step_preds_all[0]["reconstruction_errors_dict_unet3d"]) > 0:
            reconstruction_errors_dict_unet3d = {}

            for ll in pl_module.test_step_preds_all:
                reconstruction_errors_dict_unet3d.update(ll["reconstruction_errors_dict_unet3d"])

            reconstruction_errors_dict_unet3d = {int(k): v.cpu().numpy() for k, v in reconstruction_errors_dict_unet3d.items()}
            reconstruction_errors_dict_unet3d = pd.DataFrame.from_dict(reconstruction_errors_dict_unet3d, orient="index").reset_index(drop=False)
            reconstruction_errors_dict_unet3d.columns = ["seed", "recon_loss"]
            reconstruction_errors_dict_unet3d = reconstruction_errors_dict_unet3d.groupby("seed").mean().reset_index(drop=False)
            log.info(f"mean recon error over all example in val set: {reconstruction_errors_dict_unet3d.recon_loss.mean()}")
            reconstruction_errors_dict_unet3d["epoch"] = pl_module.current_epoch
            wandb.log(
                {
                    f"test/dset_{dset_version}/reconstruction_errors_dict_unet3d/": wandb.Table(dataframe=reconstruction_errors_dict_unet3d),
                    "epoch": pl_module.current_epoch,
                }
            )

            print("pausing here")

            # get min and max seeds....
            reconstruction_errors_dict_unet3d = reconstruction_errors_dict_unet3d.sort_values(by="recon_loss", ascending=False)

            reconstruction_errors_dict_unet3d = reconstruction_errors_dict_unet3d[reconstruction_errors_dict_unet3d.seed > 0]

            best_seed = reconstruction_errors_dict_unet3d.seed.head(1).item()
            best_example = trainer.datamodule.data_test.return_single_data(best_seed, augmentation=torch.nn.Identity()).cuda().unsqueeze(0)

            worst_seed = reconstruction_errors_dict_unet3d.seed.tail(1).item()
            worst_example = trainer.datamodule.data_test.return_single_data(worst_seed, augmentation=torch.nn.Identity()).cuda().unsqueeze(0)

            trainer.model.return_global_only = False

            ld = trainer.log_dir
            os.makedirs(ld, exist_ok=True)

            bb = trainer.model.forward_to_global_feature_vec(best_example)

            orig_x = bb["orig_x"][0].to(torch.float16)
            recon_x = bb["recon_x"][0].to(torch.float16)

            if trainer.model.hparams.loss.normalise_recon == True:
                orig_max = orig_x.max()
                orig_min = orig_x.min()
                rkx = recon_x  # [k]
                rkx = (rkx - rkx.min()) / (rkx.max() - rkx.min()) * (orig_max - orig_min) + orig_min

                recon_x = rkx

            orig_x_best_fn = os.path.join(ld, f"test_best_reconstructed_example_ORIGINAL_epoch_{trainer.current_epoch}_run_{wandb.run.id}.pt")
            recon_x_best_fn = os.path.join(ld, f"test_best_reconstructed_example_RECONSTRUCTED_epoch_{trainer.current_epoch}_run_{wandb.run.id}.pt")

            torch.save(obj=orig_x, f=orig_x_best_fn)
            torch.save(obj=recon_x, f=recon_x_best_fn)

            import mrcfile

            with mrcfile.new_mmap(orig_x_best_fn.replace(".pt", ".mrc"), overwrite=True, shape=orig_x.squeeze(0, 1).shape, mrc_mode=2) as mrc:
                mrc.data[:] = orig_x.squeeze(0, 1).cpu().numpy()

            with mrcfile.new_mmap(recon_x_best_fn.replace(".pt", ".mrc"), overwrite=True, shape=recon_x.squeeze(0, 1).shape, mrc_mode=2) as mrc:
                mrc.data[:] = recon_x.squeeze(0, 1).cpu().numpy()

            ww = trainer.model.forward_to_global_feature_vec(worst_example)

            orig_x = ww["orig_x"][0].to(torch.float16)
            recon_x = ww["recon_x"][0].to(torch.float16)

            if trainer.model.hparams.loss.normalise_recon == True:
                orig_max = orig_x.max()
                orig_min = orig_x.min()
                rkx = recon_x  # [k]
                rkx = (rkx - rkx.min()) / (rkx.max() - rkx.min()) * (orig_max - orig_min) + orig_min

                recon_x = rkx

            # output with rwd_model_id.......multirun not working!!
            orig_x_best_fn = os.path.join(ld, f"test_worst_reconstructed_example_ORIGINAL_epoch_{trainer.current_epoch}_run_{wandb.run.id}.pt")
            recon_x_best_fn = os.path.join(ld, f"test_worst_reconstructed_example_RECONSTRUCTED_epoch_{trainer.current_epoch}_run_{wandb.run.id}.pt")

            torch.save(obj=orig_x, f=orig_x_best_fn)
            torch.save(obj=recon_x, f=recon_x_best_fn)

            with mrcfile.new_mmap(orig_x_best_fn.replace(".pt", ".mrc"), overwrite=True, shape=orig_x.squeeze(0, 1).shape, mrc_mode=2) as mrc:
                mrc.data[:] = orig_x.squeeze(0, 1).cpu().numpy()

            with mrcfile.new_mmap(recon_x_best_fn.replace(".pt", ".mrc"), overwrite=True, shape=recon_x.squeeze(0, 1).shape, mrc_mode=2) as mrc:
                mrc.data[:] = recon_x.squeeze(0, 1).cpu().numpy()


def backup_files(source_dir, backup_dir, file_extension, exclude=[]):
    source_path = Path(source_dir).resolve()
    backup_path = Path(backup_dir).resolve().joinpath("code_archive")
    exclude_dirs = [source_path.joinpath(e) for e in exclude]

    for e in exclude_dirs:
        print(e)
    # Ensure the backup directory is not a subdirectory of the source directory
    if backup_path in (source_path / subdir for subdir in source_path.parts):
        raise ValueError("Backup directory cannot be a subdirectory of the source directory.")

    # Create backup directory if it doesn't exist
    backup_path.mkdir(parents=True, exist_ok=True)

    files_to_backup = []
    if type(file_extension) == list:
        for fff in file_extension:
            files_to_backup += list(source_path.rglob(f"*.{fff}"))
    else:
        # Search for files with the specified extension
        files_to_backup = list(source_path.rglob(f"*.{file_extension}"))

    log = logging.getLogger("manifest")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(pathlib.Path(backup_path).joinpath("manifest.log"))
    fh.setLevel(logging.INFO)
    # Create a formatter and set it for the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    log.addHandler(fh)
    log.propagate = False

    for file in files_to_backup:
        # print()
        # print([d.relative_to(source_path).parts for d in exclude_dirs])
        # return
        # Check if the directory of the file is in the exclude list
        if any([file.relative_to(source_path).parts[: len(d.relative_to(source_path).parts)] == d.relative_to(source_path).parts for d in exclude_dirs]):
            print(f"Skipped: {file}")
            continue

        # Construct the destination path
        relative_path = file.relative_to(source_path)
        destination = backup_path / relative_path

        # Create parent directories in the backup path if they don't exist
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Copy the file
        shutil.copy(file, destination)
        log.info(f"Backed up: {file} to {destination}")


@rank_zero_only
def back_up_run_code(run_folder):
    src_dir = pathlib.Path(os.environ["PROJECT_ROOT"]) / "reward_model_training" / "reward_model_framework" / "core_modules"
    EXT_LIST = ["yaml", "py"]
    EXCLUDE = ["wandb", "outputs", "multirun", "archived", "logs", "RWD_MODELS_FOR_TUNING", "notebooks"]

    backup_files(src_dir, run_folder, EXT_LIST, exclude=EXCLUDE)


# BK_DIR='/path/to/eg3d-rlhf-geometry/reward_model_training/reward_model_framework/core_modules/bka'
