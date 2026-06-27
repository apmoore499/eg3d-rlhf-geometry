import copy
import itertools
import os
import random
import sys
import types
from pathlib import Path
from typing import Any, Dict

import hydra
import logging
import monai
import numpy as np
import omegaconf
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule
from PIL import Image

from core_modules.models.utils_base import (
    CosineWarmupScheduler,
    bce_w_logits_loss,
    ce_loss,
    disable_running_stats,
    enable_running_stats,
    get_ncomb2,
    get_rand_reordered_pair,
    reduce_x,
    reorder_pair_by_idx,
    build_optimizer_and_scheduler,
)
from core_modules.utils import pylogger_c

import shutil

log = pylogger_c.RankedLogger(__name__, rank_zero_only=True)

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
if PROJECT_ROOT is None:
    PROJECT_ROOT = str(Path(__file__).resolve().parents[5])
MODEL_EXAMPLE_DIR = Path(PROJECT_ROOT) / "reward_model_training" / "reward_model_framework" / "core_modules" / "current_rwd_model_x_forward_input"


class UniversalRWDModel(LightningModule):
    # def __init__(self, global_feat = True, feature_transform = False,agg_type='std',global_feature_size=1024,spatial_transform_3d=True,mlp_global={},**kwargs):
    def __init__(self, **kwargs):
        super().__init__()

        # model example dir will store an example of the current reward model's forward batch
        # this will be bundled with the reward model saved weights
        self.model_example_dir = str(MODEL_EXAMPLE_DIR)
        MODEL_EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        # now clean the example dir if not empty
        shutil.rmtree(MODEL_EXAMPLE_DIR)
        MODEL_EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

        self.save_hyperparameters(logger=False)
        if not hasattr(self, "hparams") or not hasattr(self.hparams, "optimizer"):
            # Fallback for environments where Lightning's save_hyperparameters stub does not populate hparams.
            self.hparams = types.SimpleNamespace(**kwargs)

        self.return_global_embedding = kwargs["return_global_embedding"]
        self.global_feature_size = kwargs["mlp_global"]["output_size"]
        self.external_init_dict = kwargs["external"]  # should be omegaconf...

        self.augmentations = None

        self.setup_results_lists()

        if self.external_init_dict is not None:
            self.set_external()

        self.set_MLPs(kwargs)
        self.setup_opt_sched_init()

        self.run_embeddings_individually = False

    def setup_results_lists(self):
        self.training_step_losses_all = []
        self.training_step_preds_all = []

        self.validation_step_losses_all = []
        self.validation_step_preds_all = []

        self.test_step_losses_all = []
        self.test_step_preds_all = []

        self.epo_correct_list = []  # used to store metrics during trianing selecting best model etc. but is not the data frame itself!!
        self.test_correct_list = []  # used to store metrics during TEST selecting best model etc. but is not the data frame itself!!

        df_results_init = pd.DataFrame([-1, "INIT", "INIT", -1.0]).transpose()
        df_results_init.columns = ["epoch", "data", "type", "pc_correct"]
        self.joined_results = df_results_init.set_index(["epoch", "data", "type"])

        df_results_init = pd.DataFrame([-1, "INIT", "INIT", -1.0, "INIT", "INIT"]).transpose()
        df_results_init.columns = ["epoch", "data", "type", "pc_correct", "model_weights", "dset_version"]
        self.test_joined_results = df_results_init.set_index(["model_weights", "data", "type", "dset_version"])

        return self

    def set_external(self):
        self.external = hydra.utils.instantiate(self.external_init_dict, convert="all")

        self.external.eval()

        return self

    def remove_external(self):
        self.external = None
        return self

    def save_model_example_input(self, x):
        if len(os.listdir(self.model_example_dir)) == 0:
            ex_output_fn = os.path.join(self.model_example_dir, "model_example_input.pt")
            torch.save(obj=x, f=ex_output_fn)
            log.info(f"saved model input example to :{ex_output_fn}")

        return

    def set_MLPs(self, kwargs):
        self.MLP = hydra.utils.instantiate(kwargs["mlp_global"])
        self.scalar_rwd_head_BT = hydra.utils.instantiate(kwargs["mlp_BT"])
        self.scalar_rwd_head = hydra.utils.instantiate(kwargs["mlp_scalar_rwd"])
        self.scalar_rwd_head_pairs = hydra.utils.instantiate(kwargs["mlp_pairs"])

        return self

    def setup_opt_sched_init(self):
        self.swa_model = None
        self.automatic_optimization = True
        self.using_sam_optimizer = False
        self.using_scheduler = False

        self.hparams.optimizer = hydra.utils.instantiate(self.hparams.optimizer)  # ,partial=True)

        if hasattr(self.hparams, "scheduler") and self.hparams.scheduler is not None:
            self.hparams.scheduler = hydra.utils.instantiate(self.hparams.scheduler)  # ,partial=True)
        else:
            self.hparams.scheduler = None

        return self

    def on_train_epoch_end(self):
        b0 = self.training_step_losses_all[0]
        b1 = self.training_step_preds_all[0]

        batch_sizes = [preds["sum_in_batch"] for preds in self.training_step_preds_all]
        total_size = sum(batch_sizes)

        loss_keys = [k for k in b0.keys()]

        log.info("\n\n--------------------------------------------------------")

        for lk in loss_keys:
            entire_losses_scaled = torch.hstack([q[lk].flatten() * b for q, b in zip(self.training_step_losses_all, batch_sizes)])
            entire_losses_mean = torch.sum(entire_losses_scaled) / total_size

            self.log(f"train_epoch/{lk}", entire_losses_mean, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)

            log.info(f"current_epoch: {self.current_epoch}\ttrain_epoch/{lk:>30}:\t{entire_losses_mean:.5f}")

        log.info("--------------------------------------------------------\n\n")

        self.training_step_losses_all.clear()

        if self.hparams.swa:
            self.compute_swa_weights()

    def compute_swa_weights(self):
        epoch = self.trainer.current_epoch

        if epoch == self.hparams.swa_start:
            self.swa_n = 0

        # Update SWA weights after SWA start
        if epoch >= self.hparams.swa_start:
            self.swa_n += 1
            log.info(f"Computing SWA weights for swa_n: {self.swa_n}")

            with torch.no_grad():
                if self.swa_model is None:
                    # Initialize SWA model weights with the current model weights
                    self.swa_model = {k: v.cpu() for k, v in self.trainer.model.state_dict().items()}
                else:
                    # Average the SWA model weights with the current model weights
                    for swa_param, model_param in zip(self.swa_model.values(), self.trainer.model.state_dict().values()):
                        swa_param = swa_param * (self.swa_n - 1) / self.swa_n
                        swa_param = swa_param + (model_param.cpu() / self.swa_n)

            log.info("Computed SWA weights")

        return self

    def return_swa_weights(self):
        if self.swa_model is None:
            log.info("No swa model used, returning None")
            return None
        else:
            swa_sd = copy.deepcopy(self.swa_model)
            external_keys = [k for k in swa_sd.keys() if "external." in k]

            for key in external_keys:
                if key in swa_sd:
                    del swa_sd[key]

            return swa_sd

    def on_validation_epoch_end(self):
        b0 = self.validation_step_losses_all[0]

        batch_sizes = [preds["sum_in_batch"] for preds in self.validation_step_preds_all]
        total_size = sum(batch_sizes)

        loss_keys = [k for k in b0.keys()]

        log.info("\n\n--------------------------------------------------------")

        for lk in loss_keys:
            entire_losses_scaled = torch.hstack([q[lk].flatten() * b for q, b in zip(self.validation_step_losses_all, batch_sizes)])
            entire_losses_mean = torch.sum(entire_losses_scaled) / total_size

            log_prog = False
            if lk == "total_loss" or lk == "reco_loss":
                log_prog = True

            self.log(f"val_epoch/{lk}", entire_losses_mean, on_step=False, on_epoch=True, prog_bar=log_prog, batch_size=1)

            log.info(f"current_epoch: {self.current_epoch}\tval_epoch/{lk:>30}:\t{entire_losses_mean:.5f}")

        # output heatmaps.....
        for p in self.validation_step_preds_all:
            hmp = p["heatmap_images"]
            seeds = p["seeds"].flatten()

            outdir = self.trainer.default_root_dir

            vstep = self.trainer.current_epoch
            out_dir = os.path.join(outdir, f"val_epo_{vstep}")

            os.makedirs(out_dir, exist_ok=True)

            for hmp_image, seed in zip(hmp, seeds):
                out_fn = os.path.join(out_dir, f"s_{seed.item()}.jpg")

                hmp_image.save(out_fn)

        self.validation_step_losses_all.clear()
        self.validation_step_preds_all.clear()  # free memory

        if len(self.epo_correct_list) > 0:
            kk = pd.concat([pd.DataFrame.from_dict(p, orient="index").transpose() for p in self.epo_correct_list], axis=0, ignore_index=True)
            kk = kk[(kk.epoch == self.trainer.current_epoch) & (kk.data == "val")]
            kk_pairs = kk[kk.type == "pairs"]

            if not kk_pairs.empty:
                selected_for_log = kk_pairs.pc_correct.item()
                self.log("val_epoch/PAIR_acc/", selected_for_log, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)

                lk = "PAIR_acc"
                log.info(f"current_epoch: {self.current_epoch}\tval_epoch/{lk:>30}:\t{selected_for_log:.5f}")

            kkk = kk.set_index(["epoch", "data", "type"])
            joined_results = pd.concat([self.joined_results, kkk], axis=0, ignore_index=False)

            dup_idx = joined_results.index.duplicated()
            assert np.all(dup_idx == False), "error you have duplicated rows in result table"
            self.joined_results = joined_results
        else:
            log.info(f"current_epoch: {self.current_epoch}\tval_epoch/{'PAIR_acc':>30}:\tskipped (no pair-accuracy table)")

        log.info("--------------------------------------------------------\n\n")

        return self

    def on_test_epoch_end(self):
        b0 = self.test_step_losses_all[0]

        batch_sizes = [preds["sum_in_batch"] for preds in self.test_step_preds_all]
        total_size = sum(batch_sizes)

        loss_keys = [k for k in b0.keys()]

        for lk in loss_keys:
            entire_losses_scaled = torch.hstack([q[lk].flatten() * b for q, b in zip(self.test_step_losses_all, batch_sizes)])
            entire_losses_mean = torch.sum(entire_losses_scaled) / total_size

            log_prog = False
            if lk == "total_loss" or lk == "reco_loss":
                log_prog = True

            weights_type = self.weights_type

            self.log(f"test_epoch_{weights_type}/{lk}", entire_losses_mean, on_step=False, on_epoch=True, prog_bar=log_prog, batch_size=1)

            log.info(f"test_epoch_{weights_type}/{lk}: {entire_losses_mean:.4f}")

        self.test_step_losses_all.clear()
        self.test_step_preds_all.clear()  # free memory

        weights_type = self.weights_type

        #
        kk = pd.concat([pd.DataFrame.from_dict(p, orient="index").transpose() for p in self.test_correct_list], axis=0, ignore_index=True)
        kk = kk[(kk.epoch == self.trainer.current_epoch) & (kk.data == "test") & (kk.model_weights == weights_type) & (kk.dset_version == self.trainer.datamodule.dset_version)]

        kkk = kk.set_index(["model_weights", "data", "type", "dset_version"])
        joined_results = pd.concat([self.test_joined_results, kkk], axis=0, ignore_index=False)

        dup_idx = joined_results.index.duplicated()
        assert np.all(dup_idx == False), "error you have duplicated rows in result table"
        self.test_joined_results = joined_results

        return self

    def training_step(self, batch, batch_idx):
        # for debugging purposess:
        #        loss_to_return=torch.tensor((0.0),device=torch.device('cuda'))
        #        loss_to_return.requires_grad_()
        #        return loss_to_return
        #
        # https://github.com/davda54/sam
        if self.using_sam_optimizer == True and not self.automatic_optimization:
            optimizer = self.optimizers()

            def closure():
                losses_one, preds = self.run_forward_pass(batch, return_global_vector=False, return_preds=False)

                loss = losses_one["total_loss"]
                loss.backward()
                return loss

            losses_one, preds = self.run_forward_pass(batch, return_global_vector=False, return_preds=False)
            loss = losses_one["total_loss"]
            loss.backward()
            optimizer.step(closure)
            optimizer.zero_grad()

            seeds_batch = preds["batch_seedse"]
            seed_logger = logging.getLogger("train_seeds")
            seed_logger.propagate = False
            for sss in seeds_batch:
                seed_logger.info(",".join(map(str, sss.tolist())))

            for k in losses_one.keys():
                log_prog = False
                if k == "total_loss" or k == "reco_loss":
                    log_prog = True
                self.log(
                    f"train_step/{k}",
                    losses_one[k],
                    on_step=True,
                    on_epoch=False,
                    prog_bar=log_prog,
                    batch_size=preds["sum_in_batch"],
                )

            self.training_step_losses_all.append(losses_one)
            self.training_step_preds_all.append(preds)

            return loss

        if self.automatic_optimization:
            losses, preds = self.run_forward_pass(batch, return_global_vector=False)

            batch_size = preds["sum_in_batch"]

            seeds_batch = preds["batch_seedse"]
            seed_logger = logging.getLogger("train_seeds")
            seed_logger.propagate = False
            for sss in seeds_batch:
                seed_logger.info(",".join(map(str, sss.tolist())))

            for k in losses.keys():
                log_prog = False
                if k == "total_loss" or k == "reco_loss":
                    log_prog = True
                self.log(
                    f"train_step/{k}",
                    losses[k],
                    on_step=True,
                    on_epoch=False,
                    prog_bar=log_prog,
                    batch_size=batch_size,
                )

            losses_total = losses.pop("total_loss")

            losses["total_loss"] = losses_total.detach()

            self.training_step_losses_all.append(losses)
            self.training_step_preds_all.append(preds)

            return losses_total

    def step_cos_sched_step(self):
        if self.using_scheduler and not self.automatic_optimization:
            scheduler = self.lr_schedulers()
            if scheduler is not None:
                scheduler.step()

                self.log("cosine_LR_scheduler", scheduler.get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=True)

        return self

    def validation_step(self, batch, batch_idx):
        losses, preds = self.run_forward_pass(batch, generate_heatmaps=True, return_global_vector=True, return_preds=True)

        seeds_batch = preds["batch_seedse"]
        seed_logger = logging.getLogger("val_seeds")
        seed_logger.propagate = False
        for sss in seeds_batch:
            seed_logger.info(",".join(map(str, sss.tolist())))

        self.validation_step_losses_all.append(losses)
        self.validation_step_preds_all.append(preds)
        batch_size = preds["sum_in_batch"]

        for k in losses.keys():
            log_prog = False
            if k == "total_loss" or k == "reco_loss":
                log_prog = True

            self.log(
                f"val_step/{k}",
                losses[k],
                on_step=True,
                on_epoch=False,
                prog_bar=log_prog,
                batch_size=batch_size,
            )  # batch: Tuple[torch.Tensor, torch.Tensor]

    def test_step(self, batch, batch_idx):
        losses, preds = self.run_forward_pass(batch, return_global_vector=True, return_preds=True)
        self.test_step_losses_all.append(losses)
        self.test_step_preds_all.append(preds)
        batch_size = preds["sum_in_batch"]

        seeds_batch = preds["batch_seedse"]
        seed_logger = logging.getLogger("test_seeds")
        seed_logger.propagate = False
        for sss in seeds_batch:
            seed_logger.info(",".join(map(str, sss.tolist())))

        for k in losses.keys():
            log_prog = False
            if k == "total_loss" or k == "reco_loss":
                log_prog = True

            self.log(
                f"test_step/{k}",
                losses[k],
                on_step=True,
                on_epoch=False,
                prog_bar=log_prog,
                batch_size=batch_size,
            )

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            if self.external is not None:
                self.external = torch.compile(self.external)
            self.MLP = torch.compile(self.MLP)
            self.scalar_rwd_head = torch.compile(self.scalar_rwd_head)
            self.scalar_rwd_head_BT = torch.compile(self.scalar_rwd_head_BT)
            self.scalar_rwd_head_pairs = torch.compile(self.scalar_rwd_head_pairs)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """

        if type(self.hparams.optimizer) == omegaconf.dictconfig.DictConfig:
            self.hparams.optimizer = hydra.utils.instantiate(self.hparams.optimizer)

        if self.hparams.scheduler is not None and type(self.hparams.scheduler) == omegaconf.dictconfig.DictConfig:
            self.hparams.scheduler = hydra.utils.instantiate(self.hparams.scheduler)

        #
        # Delegate optimizer/scheduler construction to shared helper.
        optimizer, scheduler, using_sam = build_optimizer_and_scheduler(self.hparams, self.parameters(), log)
        self.using_sam_optimizer = using_sam
        self.automatic_optimization = not using_sam
        self.using_scheduler = scheduler is not None

        if scheduler is not None:
            log.info("RWD Model init: returning optimizer with scheduler")
            log.info(f"Optimizer type: {type(optimizer)}")
            log.info(f"Scheduler type: {type(scheduler)}")
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/total_loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }

        log.info("RWD Model init: returning optimizer no scheduler")
        log.info(f"Optimizer type: {type(optimizer)}")
        return {"optimizer": optimizer}

    def forward_to_global_feature_vec(self, x):
        """TO IMPLEMENT."""

        feature_vec = x

        return feature_vec

    def forward(self, x):
        feature_vec = self.forward_to_global_feature_vec(x)
        scalar_rwd = self.forward_to_scalar_reward_from_single_global(feature_vec)
        # calar_rwd=self.scalar_rwd_head(feature_vec)
        return scalar_rwd

    def forward_to_scalar_reward_from_single_global(self, x):
        # scalar_rwd=torch.nn.functional.softplus(self.scalar_rwd_head(x))+1e-3, maybe too unstable
        scalar_rwd = self.scalar_rwd_head(x)
        return scalar_rwd

        # forward_to_scalar_reward_from_single_global

    def forward_to_BT_lambda_from_single_global(self, x, mult=1.0):
        # scalar_rwd=torch.nn.functional.softplus(self.scalar_rwd_head(x))+1e-3, maybe too unstable
        scalar_rwd = torch.exp(self.scalar_rwd_head_BT(x) * mult)
        return scalar_rwd

    def forward_from_cat_global_vectors(self, v1, v2, with_softmax=False):
        gvec = torch.cat((v1, v2), dim=1)

        pred_logit = self.scalar_rwd_head_pairs(gvec)

        if with_softmax:
            x = torch.nn.functional.softmax(pred_logit)
            return x

        else:
            return pred_logit

    def return_heatmap_activations(self, data):
        out_ims = []

        with torch.no_grad():
            for pcd in data:
                #
                if self.agg_type == "max":
                    retval = self.forward_and_return_reduction_argvals(pcd.unsqueeze(0))

                    # rwd_val=rwd_model.forward(pcd)

                    vals_idx = [[r, c] for r in range(128) for c in range(128)]
                    vals_to_highlight = [vals_idx[i] for i in retval["idx"].flatten()]

                    imraw = torch.zeros(128, 128).float().detach().cpu().numpy()
                    imraw = (imraw * 0.0).astype(np.uint8)

                    for vv in vals_to_highlight:
                        k = imraw[vv[0], vv[1]]
                        imraw[vv[0], vv[1]] = 210

                    out_im = Image.fromarray(imraw)
                elif self.agg_type == "mean":
                    #
                    raw_out = self.forward_to_pre_aggregation(pcd.unsqueeze(0))

                    variances = raw_out.var(-1).squeeze(0)

                    maxvar = torch.argsort(variances, descending=True)

                    # take top variance
                    ro = (raw_out[0, maxvar[0], ...].reshape(1, 128, 128) + 1.0) / 2.0

                    this_image = (ro.squeeze(0) * 255.0).to(torch.uint8)

                    out_im = Image.fromarray(this_image.detach().cpu().numpy())

                out_ims.append(out_im)
        return out_ims

    def log_ids(self, batch):
        # Get the IDs from the batch (example: assuming batch is a tuple with (data, ids))
        ids = batch[1]
        self.id_logger.info(",".join(map(str, ids.tolist())))

    def run_forward_pass(self, batch, generate_heatmaps=False, return_global_vector=False, return_preds=True):
        device = batch.file_batch.device
        X_dmap = batch.file_batch.to(device, non_blocking=False).squeeze(2)
        Lengths = batch.lens_batch.to(device, non_blocking=False)
        seeds = batch.ordered_seeds.to(device, non_blocking=False)

        batch_seedse = seeds.cpu().numpy()

        heatmap_images = []
        if self.external is not None:
            Lengths_idx = [torch.arange(L) for L in Lengths]
            self.save_model_example_input(X_dmap[0][Lengths_idx[0]])  # save out the example

            feature_embeddings = [self.external.forward(xd[L]) for xd, L in zip(X_dmap, Lengths_idx)]  # single pass per each batch of ordered meshes

            feature_embeddings = [self.forward_to_global_feature_vec(x) for x in feature_embeddings]
            feature_emb_chunk = []
            reconstructed_x = []
            orig_x_list = []
            cpreds = feature_embeddings
        else:
            Lengths_idx = [torch.arange(L) for L in Lengths]
            permutation_of_lengths = [torch.arange(L) for L in Lengths]  # get rid of permutation, no need
            inverse_permutation_of_lengths = [torch.argsort(p) for p in permutation_of_lengths]
            Lengths_reordered = [L[p] for L, p in zip(Lengths_idx, permutation_of_lengths)]
            embds_for_in = [xd[L] for xd, L in zip(X_dmap, Lengths_reordered)]
            embds_for_in = torch.vstack(embds_for_in)
            embds_for_in = embds_for_in.chunk(max(1, int(embds_for_in.shape[0] / 8)))

            feature_emb_chunk = []
            reconstructed_x = []
            orig_x_list = []

            self.save_model_example_input(embds_for_in[0][0].unsqueeze(0))  # save out the example

            if self.run_embeddings_individually:
                for e in embds_for_in:
                    for indiv in e:
                        fec = self.forward_to_global_feature_vec(indiv.unsqueeze(0))

                        if type(fec) == dict:
                            recon_x = fec["recon_x"]
                            orig_x = fec["orig_x"]
                            reconstructed_x.append(recon_x)
                            orig_x_list.append(orig_x)
                            fec = fec["feature_vec"]
                        feature_emb_chunk.append(fec)

            else:
                feature_emb_chunk = [self.forward_to_global_feature_vec(x) for x in embds_for_in]  # embds_for_in[0].shape
            feature_emb_chunk = torch.vstack(feature_emb_chunk)
            feature_embeddings = torch.split(feature_emb_chunk, Lengths.tolist(), dim=0)

            # Only PointnetOneModNonLinearHeads implements return_heatmap_activations.
            if generate_heatmaps and self.__class__.__name__ == "PointnetOneModNonLinearHeads" and hasattr(self, "return_heatmap_activations"):
                hmp_i = [self.return_heatmap_activations(x) for x in embds_for_in]
                for h in hmp_i:
                    heatmap_images += h

            cpreds = [cp[i] for cp, i in zip(feature_embeddings, inverse_permutation_of_lengths)]

        cseeds = [s[:L].reshape(-1, 1) for s, L in zip(seeds, Lengths)]
        global_feature_preds = cpreds  # [cp.clone().detach().cpu() for cp in cpreds]

        dict_of_global_feature = {"-10000": torch.tensor(1)}

        CPU_SEEDS = torch.vstack(cseeds)  # .cpu().flatten().flatten().numpy()

        if return_global_vector and return_preds:
            global_embeddings = torch.vstack(global_feature_preds).view(CPU_SEEDS.shape[0], -1)

            if type(global_embeddings) == monai.data.meta_tensor.MetaTensor:
                global_embeddings = global_embeddings.as_tensor()

            dict_of_global_feature = {s: v[None, ...].detach() for s, v in zip(CPU_SEEDS, global_embeddings)}

        ordered_combos = [list(itertools.combinations(range(L), 2)) for L in Lengths]
        binary_idx_order_rand = [[get_rand_reordered_pair(p=1) for t in oc] for oc in ordered_combos]
        ordered_combos_rand = [[reorder_pair_by_idx(o, p) for o, p in zip(oc, pc)] for oc, pc in zip(ordered_combos, binary_idx_order_rand)]

        Lengths_np = Lengths.detach().cpu().numpy()

        sum_in_batch = np.sum([get_ncomb2(l) for l in Lengths_np])

        batches = [torch.cat([torch.cat((cp[o[0]].unsqueeze(0), cp[o[1]].unsqueeze(0)), 1) for o in oc]) for oc, cp in zip(ordered_combos_rand, cpreds)]

        seeds_batches = [torch.cat([torch.cat((cp[o[0]].unsqueeze(0), cp[o[1]].unsqueeze(0)), 1) for o in oc]) for oc, cp in zip(ordered_combos_rand, cseeds)]
        seeds_batches_rev = [torch.cat([torch.cat((cp[o[1]].unsqueeze(0), cp[o[0]].unsqueeze(0)), 1) for o in oc]) for oc, cp in zip(ordered_combos_rand, cseeds)]

        targets = [torch.tensor([0 for w in bi], dtype=torch.uint8, device=device) for bi in binary_idx_order_rand]
        targets_rev = [torch.tensor([1 for w in bi], dtype=torch.uint8, device=device) for bi in binary_idx_order_rand]

        cat_targ = torch.cat(targets, 0)
        cat_targ_rev = torch.cat(targets_rev, 0)

        all_targ = torch.cat((cat_targ, cat_targ_rev), dim=0).view(2, -1).t().reshape(-1, 1).split(2)

        cat_seeds = torch.cat(seeds_batches, 0)
        cat_seeds_rev = torch.cat(seeds_batches_rev, 0)

        return_for_seeds = torch.cat((cat_seeds, cat_seeds_rev), dim=0)
        return_for_logits = torch.ones_like(return_for_seeds)

        paired_loss = torch.tensor(0.0, device=device)

        if self.hparams.loss["lambda_pairs"] != 0.0:
            batches_pred = [
                self.forward_from_cat_global_vectors(
                    b[:, : self.global_feature_size],
                    b[:, self.global_feature_size :],
                    with_softmax=False,
                )
                for b in batches
            ]
            batches_rev_pred = [
                self.forward_from_cat_global_vectors(
                    b[:, self.global_feature_size :],
                    b[:, : self.global_feature_size],
                    with_softmax=False,
                )
                for b in batches
            ]

            cat_batches = torch.cat(batches_pred, 0)
            cat_batches_rev = torch.cat(batches_rev_pred, 0)

            all_batches = torch.stack((cat_batches, cat_batches_rev), dim=1).view(-1, 2).split(2)

            weights = torch.tensor([1.0, 1.0]).to(all_batches[0].device)

            sel_idx = torch.multinomial(weights, len(all_targ), replacement=True)  # select one or other of the seeds................

            blens = [b.shape[0] for b in batches]

            if self.trainer.current_epoch >= 3:  # use 3 burn in epochs...
                paired_loss = torch.hstack([ce_loss(a[i].unsqueeze(0), t.flatten()[i].unsqueeze(0)) for a, t, i in zip(all_batches, all_targ, sel_idx)]).mean() * self.hparams.loss["lambda_pairs"]

            else:
                paired_loss = torch.hstack([ce_loss(a, t.flatten()) for a, t in zip(all_batches, all_targ)]).mean() * self.hparams.loss["lambda_pairs"]

            batches_rev = torch.cat(batches_rev_pred, 0)
            batches = torch.cat(batches_pred, 0)
            return_for_logits = torch.cat((batches_rev[:, 1:], batches_rev[:, :1]), 1)  # already reversed!!!
            return_for_logits = torch.cat((batches, return_for_logits), 0).detach()

        # ----------------------------------------------------
        # ----------------------------------------------------
        # ----------------------------------------------------
        # scalar reward
        scalar_rwd_loss = torch.tensor(0.0, device=device)
        sigmoid_comparison_values = [-1]
        scalar_rwds_dist_dict_scalar = {}
        rwd_vals_for_ret_scalar = torch.tensor(-1.0, device=device)
        if self.hparams.loss["lambda_scalar_rwd"] != 0.0:
            globals_for_scalar = cpreds  # should be unscrambled.......
            Lens_for_global = [len(g) for g in globals_for_scalar]
            cpreds_scalar_rwd = [self.forward_to_scalar_reward_from_single_global(x) for x in globals_for_scalar]
            intermediate_losses = [[-torch.log(torch.sigmoid(cp[o[0]] - cp[o[1]])) for o in oc] for oc, cp in zip(ordered_combos, cpreds_scalar_rwd)]
            il = [torch.mean(torch.cat(im)).unsqueeze(0) for im in intermediate_losses]
            scalar_rwd_loss = torch.cat(il).mean() * self.hparams.loss["lambda_scalar_rwd"]

            rwd_vals_for_ret_scalar = torch.vstack(cpreds_scalar_rwd).detach()

            scalar_rwds_dist_dict_scalar = {s: v for s, v in zip(CPU_SEEDS, rwd_vals_for_ret_scalar.flatten())}

            cpreds_scalar_rwd = [c.detach() for c in cpreds_scalar_rwd]

            sigmoid_comparison_values = [torch.hstack([cp[o[0]] - cp[o[1]] for o in oc]) for oc, cp in zip(ordered_combos, cpreds_scalar_rwd)]

        # ----------------------------------------------------
        # ----------------------------------------------------
        # ----------------------------------------------------

        BT_rwd_loss = torch.tensor(0.0, device=device)
        BT_comparison_values = [-1]
        rwd_vals_for_ret_BT = torch.tensor(-1.0, device=device)
        rwds_dist_dict_BT = {}

        # bradley terry vals
        if self.hparams.loss["lambda_BT"] != 0.0:
            globals_for_BT = cpreds  # should be unscrambled.......
            Lens_for_global = [len(g) for g in globals_for_BT]
            BT_vals = [self.forward_to_BT_lambda_from_single_global(x) for x in globals_for_BT]  # minval is set in the classifier class itself forward method, usuallllly take 1e-3 based on very brief experiment
            intermediate_losses = [[-torch.log(cp[o[0]] / (cp[o[0]] + cp[o[1]])) for o in oc] for oc, cp in zip(ordered_combos, BT_vals)]
            il = [torch.mean(torch.cat(im)).unsqueeze(0) for im in intermediate_losses]
            BT_rwd_loss = torch.cat(il).mean()

            rwd_vals_for_ret_BT = torch.vstack(BT_vals).flatten().detach()

            BT_rwd_loss = BT_rwd_loss * self.hparams.loss["lambda_BT"]
            rwds_dist_dict_BT = {s: v for s, v in zip(CPU_SEEDS, rwd_vals_for_ret_BT.flatten().detach())}

            BT_vals = [b.detach() for b in BT_vals]

            BT_comparison_values = [torch.hstack([(cp[o[0]] / (cp[o[0]] + cp[o[1]])) for o in oc]) for oc, cp in zip(ordered_combos, BT_vals)]

        # ----------------------------------------------------
        # ----------------------------------------------------
        # ----------------------------------------------------

        l2_reg_lambda = torch.tensor(0.0, device=device)
        # add in l2 penalty.....
        if self.hparams.loss["lambda_BT"] != 0.0 and self.hparams.loss["lambda_reg_rwd_vals"] != 0.0:
            l2_reg_lambda = torch.mean(torch.norm(rwd_vals_for_ret_BT.flatten())) * self.hparams.loss["lambda_reg_rwd_vals"]  # for param in lambda_params) / sum(param.numel() for param in lambda_params)

        abs_val_loss = torch.tensor(0.0, device=device)
        l2_reg_agg_features = torch.tensor(0.0, device=device)

        if self.hparams.loss["lambda_agg_features_l2"] != 0.0 and self.training:  # requires_grad to check for traiing or eval mode
            l2_reg_agg_features = torch.mean(torch.norm(agg_features_together, dim=0)) * lambda_agg_features_l2

        reco_loss = torch.tensor(0.0, device=device)

        unet3d_preds = dict()
        reconstruction_errors_dict_unet3d = dict()
        if len(orig_x_list) > 0 and orig_x_list[0] is not None:
            contains_tensor = type(orig_x_list[0][0]) == torch.Tensor

            if not contains_tensor:
                orig_x = [xci[0].as_tensor() for xci in orig_x_list]
                reco_x = [xci[0].as_tensor() for xci in reconstructed_x]

            else:
                orig_x = [xci[0] for xci in orig_x_list]
                reco_x = [xci[0] for xci in reconstructed_x]

            if self.hparams.loss.recon_type == "mse":
                lfunc = torch.nn.MSELoss()

            elif self.hparams.loss.recon_type == "l1":
                lfunc = torch.nn.L1Loss()

            else:
                assert False, f"reconstruction loss type not mse or l1: {self.hparams.loss.recon_type}\t (specified in losses for run cfg)"

            if self.hparams.loss.normalise_recon == True:
                for k in range(len(orig_x)):
                    orig_max = orig_x[k].max()
                    orig_min = orig_x[k].min()
                    rkx = reco_x[k]
                    rkx = (rkx - rkx.min()) / (rkx.max() - rkx.min()) * (orig_max - orig_min) + orig_min

                    reco_x[k] = rkx

            reco_x_stack = torch.vstack([r.unsqueeze(0) for r in reco_x])
            orig_x_stack = torch.vstack([o for o in orig_x])

            reco_loss = lfunc(reco_x_stack, orig_x_stack) * self.hparams.loss.lambda_recon  # if this doesn't work, try scaling reco_x_stack to be within range 0,1.

            if return_preds:
                rxs = reco_x_stack.detach()
                oxs = orig_x_stack  # .detach()

                reconstruction_errors_dict_unet3d = {s: lfunc(rx, ox).cpu() for s, rx, ox in zip(CPU_SEEDS, rxs, oxs)}

                # preds.update(reconstruction_errors_dict_unet3d=reconstruction_errors_dict_unet3d)

        if return_preds:
            preds = dict(
                dict_of_global_feature=dict_of_global_feature,
                # logits (paired comparison)
                pred_logits=return_for_logits,
                seeds=return_for_seeds,
                batch_seedse=batch_seedse,
                sum_in_batch=sum_in_batch,  #
                # scalar reward
                rwd_vals_scalar=rwd_vals_for_ret_scalar,
                sigmoid_comparison_values=sigmoid_comparison_values,
                scalar_rwds_dist_dict_scalar=scalar_rwds_dist_dict_scalar,
                # Bradley-Terry comparison
                BT_comparison_values=BT_comparison_values,
                rwd_vals_BT=rwd_vals_for_ret_BT,
                rwds_dist_dict_BT=rwds_dist_dict_BT,
                heatmap_images=heatmap_images,
                # Reconstruction losses from unet3d models,
                reconstruction_errors_dict_unet3d=reconstruction_errors_dict_unet3d,
            )
        else:
            preds = dict(
                seeds=return_for_seeds,
                batch_seedse=batch_seedse,
            )

        losses = dict(
            paired_loss=paired_loss.detach(),
            BT_rwd_loss=BT_rwd_loss.detach(),
            abs_val_loss=abs_val_loss.detach(),
            scalar_rwd_loss=scalar_rwd_loss.detach(),
            l2_reg_lambda=l2_reg_lambda.detach(),
            l2_reg_agg_features=l2_reg_agg_features.detach(),
            reco_loss=reco_loss.detach(),
            total_loss=paired_loss + BT_rwd_loss + abs_val_loss + scalar_rwd_loss + l2_reg_lambda + l2_reg_agg_features + reco_loss,
        )

        return (losses, preds)


__all__ = [
    "UniversalRWDModel",
    "log",
]
