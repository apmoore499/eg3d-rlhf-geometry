# ruff: noqa: E402  # sys.path bootstrap below must run before package imports
import importlib.metadata as importlib_metadata
import logging
import math
import os
import sys
import warnings
from pathlib import Path
from typing import List

# Ensure `reward_model_framework/` is on sys.path so `import core_modules`
# resolves when this file is launched as a script (e.g.
# `python core_modules/train_rwd_model.py`). The PyPI `autoroot` package
# below only adds the repo root, which is two levels above `core_modules`.
_RMF_DIR = str(Path(__file__).resolve().parent.parent)
if _RMF_DIR not in sys.path:
    sys.path.insert(0, _RMF_DIR)

import autoroot  # noqa: F401
import hydra
import lightning as L
import matplotlib
import pprint
import shutil
import lightning
import subprocess

matplotlib.use("agg")
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import wandb
from hydra.core.hydra_config import HydraConfig  # isort:skip # to get paths / global cfg settings / log file etc
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.utilities.model_summary import summarize as summarize_pt_lightning  # isort:skip

import core_modules
import core_modules.data.collate_data as dc
from core_modules.data import lightning_dloader

from core_modules.data import ranking_datasets as rld
from core_modules.utils.instantiators import instantiate_callbacks, instantiate_loggers
from core_modules.utils.rich_utils import print_config_tree

log = core_modules.utils.pylogger_c.RankedLogger(__name__, rank_zero_only=True)

# Backward-compat: add packages_distributions if missing (py39)
if not hasattr(importlib_metadata, "packages_distributions"):
    try:
        import importlib_metadata as importlib_metadata_backport

        importlib_metadata.packages_distributions = importlib_metadata_backport.packages_distributions
    except Exception:
        importlib_metadata.packages_distributions = lambda: {}

# Expected large images; avoid PIL decompression bomb warnings
warnings.simplefilter("ignore", Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None


def setup_seeds_logger(name: str):
    """Create a dedicated log file for a given seed log name."""
    oeld = Path(HydraConfig.get().job_logging.handlers.file.filename).parent
    fh = logging.FileHandler(oeld.joinpath(f"{name}.log"))
    lgr = core_modules.utils.pylogger_c.RankedLogger(name, rank_zero_only=True)
    lgr.logger.addHandler(fh)
    lgr.propagate = False
    return lgr


try:
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
except Exception:
    pass

OmegaConf.register_new_resolver("multiply_to_int", lambda x, y: int(x * y), replace=True)
OmegaConf.register_new_resolver("divide_ceil", lambda x, y: int(math.ceil(x / y)), replace=True)
OmegaConf.register_new_resolver("divide_floor", lambda x, y: int(math.floor(x / y)), replace=True)


ce_loss = nn.CrossEntropyLoss()
bce_w_logits_loss = nn.BCEWithLogitsLoss()
mse_loss = torch.nn.MSELoss()

REPO_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
SAM_PATH = REPO_ROOT / "eg3d" / "external_modules" / "sam"
if SAM_PATH.exists():
    sys.path.append(str(SAM_PATH))


def trainer_query_swa(trainer):
    cbs = trainer.callbacks

    for c in cbs:
        if type(c) == lightning.pytorch.callbacks.stochastic_weight_avg.StochasticWeightAveraging:
            sdict = c._average_model.state_dict()
            return sdict
    return None


def set_debug_apis(state: bool = False):
    torch.autograd.profiler.profile(enabled=state)
    torch.autograd.profiler.emit_nvtx(enabled=state)
    torch.autograd.set_detect_anomaly(mode=state)


# Then in training code before the train loop
set_debug_apis(state=False)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _dump_loader_batch_seeds_csv(loader, out_path):
    """Write rows=batches, columns=flattened ordered seeds, iterating the actual
    DataLoader so the CSV reflects what the model sees -- including any goodseed
    that ``dset_single_stream_ordered_minimal.__getitem__`` prepends to each
    sample under ``self.goodseed_pred_prob``.
    """
    batch_rows = []
    for batch in loader:
        seeds = batch.ordered_seeds.detach().cpu().numpy().reshape(-1).tolist()
        batch_rows.append(seeds)
    df = pd.DataFrame(batch_rows)
    df.index.name = "batch_idx"
    df.to_csv(out_path)


def _keep_rows_without_goodseeds(dataset, goodseeds_set):
    """Indices of ranking rows in `dataset` whose seeds are all outside `goodseeds_set`.

    Each row of `dataset.all_combined_rankings_ordered` is a 1-D seed array padded
    with -1; the row is dropped if any of its real (non-padding) seeds is a goodseed.
    """
    rows = dataset.all_combined_rankings_ordered
    keep = []
    for i, row in enumerate(rows):
        seeds = [int(s) for s in row if int(s) != -1]
        if not any(s in goodseeds_set for s in seeds):
            keep.append(i)
    return keep


def _save_final_metrics(*, model, trainer, datamodule, best_weights, collate_fn, run_dir):
    """Evaluate the best (non-SWA) checkpoint on val/test, with and without goodseeds.

    Writes `final_metrics.csv` into `run_dir`. Accuracies aren't surfaced via
    `trainer.callback_metrics`; they're appended to `model.epo_correct_list` /
    `model.test_correct_list` by the logging callbacks, so we snapshot list
    lengths around each eval call and pick the new "pairs" row.
    """
    from core_modules.utils import rwd_model_utils as rmu

    goodseeds_set = set(rmu.GOODSEEDS)
    val_ds = datamodule.data_val
    test_ds = datamodule.data_test
    train_ds = datamodule.data_train

    # Train accuracy is optional: if data_train is wrapped (e.g. by the
    # commented-out dset_for_cont_integration path) it may not expose the
    # ranking-row attributes the filter relies on.
    include_train = hasattr(train_ds, "all_combined_rankings_ordered") and hasattr(train_ds, "goodseed_pred_prob")
    if not include_train:
        log.info("final-metrics: data_train lacks ranking/goodseed attrs; skipping train accuracy")

    keep_val = _keep_rows_without_goodseeds(val_ds, goodseeds_set)
    keep_test = _keep_rows_without_goodseeds(test_ds, goodseeds_set)
    keep_train = _keep_rows_without_goodseeds(train_ds, goodseeds_set) if include_train else []
    n_val, n_test = len(val_ds), len(test_ds)
    n_train = len(train_ds) if include_train else 0
    log.info(f"final-metrics: val rows kept {len(keep_val)}/{n_val} ({n_val - len(keep_val)} contain goodseeds)")
    log.info(f"final-metrics: test rows kept {len(keep_test)}/{n_test} ({n_test - len(keep_test)} contain goodseeds)")
    if include_train:
        log.info(f"final-metrics: train rows kept {len(keep_train)}/{n_train} ({n_train - len(keep_train)} contain goodseeds)")

    sd = best_weights["state_dict"] if isinstance(best_weights, dict) and "state_dict" in best_weights else best_weights
    model.load_state_dict(sd, strict=False)
    model.eval()
    model.weights_type = "not_SWA"

    p = datamodule.hparams

    def _make_loader(ds, indices, batch_size):
        subset = ds if indices is None else torch.utils.data.Subset(ds, indices)
        return torch.utils.data.DataLoader(
            dataset=subset,
            num_workers=p.num_workers,
            pin_memory=p.pin_memory,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn,
            batch_size=batch_size,
        )

    val_loader_full = _make_loader(val_ds, None, p.batch_size_val)
    val_loader_filt = _make_loader(val_ds, keep_val, p.batch_size_val)
    test_loader_full = _make_loader(test_ds, None, p.batch_size_test)
    test_loader_filt = _make_loader(test_ds, keep_test, p.batch_size_test)
    if include_train:
        # shuffle=False so the seed CSV is deterministic; goes through
        # validation_step (eval mode, no aug/dropout, no_grad).
        train_loader_full = _make_loader(train_ds, None, p.batch_size_train)
        train_loader_filt = _make_loader(train_ds, keep_train, p.batch_size_train)

    run_dir_s = str(run_dir)

    def _new_pairs_acc(buf, split):
        for row in buf:
            if row.get("type") == "pairs" and row.get("data") == split:
                return float(np.array(row["pc_correct"]).flatten().item())
        return float("nan")

    # The model's per-epoch hooks (base.py on_validation_epoch_end /
    # on_test_epoch_end) assume exactly one matching row per (epoch, split)
    # in epo_correct_list / test_correct_list and assert against duplicate
    # index entries in joined_results / test_joined_results. Calling
    # trainer.validate / trainer.test after fit (current_epoch is frozen)
    # would violate both. Isolate each extra eval with a snapshot/restore.

    def _run_validate(loader):
        saved_list = model.epo_correct_list
        saved_jr = model.joined_results
        model.epo_correct_list = []
        model.joined_results = saved_jr.iloc[0:0]
        try:
            trainer.validate(model=model, dataloaders=loader)
            return _new_pairs_acc(list(model.epo_correct_list), "val")
        finally:
            model.epo_correct_list = saved_list
            model.joined_results = saved_jr

    def _run_test(loader):
        saved_list = model.test_correct_list
        saved_jr = model.test_joined_results
        model.test_correct_list = []
        model.test_joined_results = saved_jr.iloc[0:0]
        try:
            trainer.test(model=model, dataloaders=loader)
            return _new_pairs_acc(list(model.test_correct_list), "test")
        finally:
            model.test_correct_list = saved_list
            model.test_joined_results = saved_jr

    # The "no_goodseeds" pass needs BOTH: drop ranking rows that contain a
    # goodseed (currently a no-op for val/test) AND disable the dynamic
    # goodseed injection from dset_single_stream_ordered_minimal.__getitem__,
    # which prepends a random seed from self.list_of_good_seeds with
    # probability self.goodseed_pred_prob. We toggle both datasets together
    # and restore on exit.
    saved_val_prob = val_ds.goodseed_pred_prob
    saved_test_prob = test_ds.goodseed_pred_prob

    saved_train_prob = train_ds.goodseed_pred_prob if include_train else None

    # all_seeds pass: keep injection as-is (matches current training behavior).
    val_full = _run_validate(val_loader_full)
    test_full = _run_test(test_loader_full)
    _dump_loader_batch_seeds_csv(val_loader_full, os.path.join(run_dir_s, "val_all_seeds_batches.csv"))
    _dump_loader_batch_seeds_csv(test_loader_full, os.path.join(run_dir_s, "test_all_seeds_batches.csv"))
    if include_train:
        train_full = _run_validate(train_loader_full)
        _dump_loader_batch_seeds_csv(train_loader_full, os.path.join(run_dir_s, "train_all_seeds_batches.csv"))
    else:
        train_full = float("nan")

    # no_goodseeds pass: disable injection, then eval + dump.
    val_ds.goodseed_pred_prob = 0.0
    test_ds.goodseed_pred_prob = 0.0
    if include_train:
        train_ds.goodseed_pred_prob = 0.0
    try:
        val_filt = _run_validate(val_loader_filt)
        test_filt = _run_test(test_loader_filt)
        _dump_loader_batch_seeds_csv(val_loader_filt, os.path.join(run_dir_s, "val_no_goodseeds_batches.csv"))
        _dump_loader_batch_seeds_csv(test_loader_filt, os.path.join(run_dir_s, "test_no_goodseeds_batches.csv"))
        if include_train:
            train_filt = _run_validate(train_loader_filt)
            _dump_loader_batch_seeds_csv(train_loader_filt, os.path.join(run_dir_s, "train_no_goodseeds_batches.csv"))
        else:
            train_filt = float("nan")
    finally:
        val_ds.goodseed_pred_prob = saved_val_prob
        test_ds.goodseed_pred_prob = saved_test_prob
        if include_train:
            train_ds.goodseed_pred_prob = saved_train_prob
    log.info(f"final-metrics: wrote per-batch seed CSVs to {run_dir_s}")

    rows = [
        {"split": "val",  "filter": "all_seeds",    "n_examples": n_val,         "pc_correct": val_full},
        {"split": "val",  "filter": "no_goodseeds", "n_examples": len(keep_val), "pc_correct": val_filt},
        {"split": "test", "filter": "all_seeds",    "n_examples": n_test,        "pc_correct": test_full},
        {"split": "test", "filter": "no_goodseeds", "n_examples": len(keep_test),"pc_correct": test_filt},
    ]
    if include_train:
        rows.extend([
            {"split": "train", "filter": "all_seeds",    "n_examples": n_train,           "pc_correct": train_full},
            {"split": "train", "filter": "no_goodseeds", "n_examples": len(keep_train),   "pc_correct": train_filt},
        ])
    df = pd.DataFrame(rows)
    out_fn = os.path.join(str(run_dir), "final_metrics.csv")
    df.to_csv(out_fn, index=False)
    log.info(f"final-metrics: wrote {out_fn}\n{df.to_markdown(index=False)}")


@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    log.info(f"Set precision <{cfg.torch_matmul_precision}>")
    torch.set_float32_matmul_precision(cfg.torch_matmul_precision)
    torch.backends.cudnn.benchmark = True

    if not cfg.using_wandb and cfg.get("callbacks"):
        with open_dict(cfg.callbacks):
            if "wandb_images_logger" in cfg.callbacks:
                log.info("using_wandb=false -> dropping wandb_images_logger callback")
                del cfg.callbacks["wandb_images_logger"]

            model_checkpoint = cfg.callbacks.get("model_checkpoint")
            if model_checkpoint and getattr(model_checkpoint, "dirpath", None):
                dirpath = str(model_checkpoint.dirpath)
                if "/wandb/" in dirpath or dirpath.endswith("/wandb/checkpoints"):
                    cfg.callbacks.model_checkpoint.dirpath = dirpath.replace("/wandb/", "/").replace(
                        "wandb/checkpoints", "checkpoints"
                    )

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
        torch.manual_seed(cfg.seed)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    cfg.model.loss = cfg.loss
    model: LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)
    log.info(f"\n\n{summarize_pt_lightning(model, max_depth=1)}\n\n")

    count_parameters(model)
    #'hello')

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating profiler...")

    if "profiler" not in cfg.trainer.keys():
        profiler = None

    elif cfg.trainer.profiler == None:
        profiler = None

    elif type(cfg.trainer.profiler) == str:
        profiler = cfg.trainer.profiler

    elif "_target_" in cfg.trainer.profiler.keys():
        profiler = hydra.utils.instantiate(cfg.trainer.profiler)

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, profiler=profiler)  # ,barebones=True)

    if cfg.dloader.num_workers == 0:
        collate_fn = dc.CollateVariableShapeBatch
        cfg.dloader.pin_memory = False

    else:
        # 5. Set Up Collate Fn
        collate_fn = dc.CollateVariableShapeBatch
        cfg.dloader.pin_memory = True

        if cfg.dloader.num_workers > 0 and cfg.dloader.get("prefetch_factor"):
            cfg.dloader.prefetch_factor = cfg.dloader.prefetch_factor  # None

    # Only datamodule_third is ever used (see the train_on guard below). Set its
    # dset version so the ranking dataset takes the "three" code path; the
    # partition splits already come from data_defaults (0.70/0.15/0.15).
    cfg.data.dset_dict.dset_version = "three"

    third_data = rld.set_up_dataloaders(cfg, collate_fn=collate_fn)["data"]

    train = third_data.finished_dset["train"].dsets[0]
    test = third_data.finished_dset["test"].dsets[0]
    val = third_data.finished_dset["val"].dsets[0]

    datamodule_third: LightningDataModule = lightning_dloader.UniversalDataModule(
        train=train,
        val=val,
        test=test,
        collate_fn=collate_fn,
        dset_version="third",
        worker_init_fn=None,
        **cfg.dloader,
    )

    datamodule_third.setup()

    my_dict = OmegaConf.to_container(cfg)

    if not cfg.get("optuna_tuning"):
        log.info("printing cfg....")

        print_config_tree(DictConfig(my_dict), resolve=True, save_to_file=True)

    # from rich import print as rprint

    cfg_container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    str_dict = pprint.pformat(cfg_container)

    # l##og.info(f'\n\n{str_dict}\n\n')

    yml_fn = os.path.join(cfg.paths.output_dir, "yaml_cfg_pre_train.yaml")

    with open(yml_fn, "w") as f:
        f.write(str_dict)

    if cfg.using_wandb:
        mode = "offline" if cfg.logger.wandb.offline else "online"

        # Use a thread-based start method to avoid occasional hangs during shutdown/artifact handoff
        wandb.init(
            project=cfg.logger.wandb.project,
            config=cfg_container,
            name=cfg.logger.wandb.name,
            mode=mode,
            group=cfg.logger.wandb.group,
            tags=cfg.logger.wandb.tags,
            job_type=cfg.logger.wandb.job_type,
            settings=wandb.Settings(start_method="thread"),
            # dir=trainer.default_root_dir,
            # offline=cfg.logger.wandb.offline,
            # name=f"{c.run_dir}".split("/")[-1],
        )

        # wandb.config.update(cfg)
        # if

        log.info(f"run id is : {wandb.run.id}")

        cfg.logger.wandb.id = wandb.run.id

        wandb.define_metric("val/pairs/pc_correct", summary="max")
        wandb.define_metric("val/total_loss", summary="min")

        #
        rundir_fn = os.path.join(wandb.run.dir, "run_config_before_training.yaml")

        with open(rundir_fn, "w") as f:
            OmegaConf.save(cfg, f.name)

    scratch_root = Path(cfg.trainer.default_root_dir)
    dm_dir = scratch_root / "tmp_datamodules"

    dm_dir.mkdir(parents=True, exist_ok=True)

    # dm1_fn=os.path.join(dm_dir,'datamodule_first.pt')
    # torch.save(obj=datamodule_first,f=dm1_fn)

    dm3_fn = dm_dir / "datamodule_third.pt"
    torch.save(obj=datamodule_third, f=dm3_fn)

    # current run folder for saving the rwd model forward input example
    model_example_dir = Path(__file__).resolve().parent.joinpath("current_rwd_model_x_forward_input")

    # also save it in the run dir
    if cfg.using_wandb and wandb.run is not None:
        tuning_root = Path(__file__).resolve().parent / "RWD_MODELS_FOR_TUNING"
        RWD_MODELS_FOR_TUNING_DIR = tuning_root.joinpath(wandb.run.id)

        RWD_MODELS_FOR_TUNING_DIR.mkdir(parents=True, exist_ok=True)

        torch.save(obj=datamodule_third, f=RWD_MODELS_FOR_TUNING_DIR.joinpath("datamodule_third.pt"))

        # back up the code used for run

        if cfg.backupcode:
            log.info("backing up code...")
            core_modules.utils.logging_utils.back_up_run_code(RWD_MODELS_FOR_TUNING_DIR)
            log.info(f"code backed up to: {RWD_MODELS_FOR_TUNING_DIR}")

    # ensure model example dir is refreshed for future runs
    if model_example_dir.exists():
        shutil.rmtree(model_example_dir)
    model_example_dir.mkdir(parents=True, exist_ok=True)

    if cfg.log_net_debug and cfg.using_wandb and wandb.run is not None:
        wandb.watch(model, log="all", log_freq=50)

    # hack the ckpt path method..........
    if cfg.get("ckpt_path"):  # just load state4 dict instead................
        # load it....
        ckpt_sd = torch.load(cfg.ckpt_path)  # e.g. /.../reward_model_training/reward_model_framework/core_modules/RWD_MODELS_FOR_TUNING/<run_id>/best_model.pt
        model.load_state_dict(ckpt_sd, strict=False)

    if cfg.get("train"):
        log.info("Starting training!")

        if cfg.train_on != "datamodule_third":
            raise NotImplementedError("Only training on datamodule_third is supported in this simplified script.")

        datamodule_third = torch.load(dm3_fn)
        trainer.fit(
            model=model,
            datamodule=datamodule_third,  # ckpt_path=cfg.get("ckpt_path")
        )  # ,checkpoint_callback=False)

        # copy example input data used for the current reward model training run
        # into the persistent run folder. Only when wandb tracking is on:
        # RWD_MODELS_FOR_TUNING_DIR is keyed by the wandb run id and is only set
        # in the `if cfg.using_wandb` block above (so non-wandb runs, e.g. the
        # backbone smoke test, skip this cleanly instead of crashing).
        if cfg.using_wandb and wandb.run is not None:
            src_model_example_pt_fn = model_example_dir.joinpath("model_example_input.pt")
            dst_model_example_pt_fn = RWD_MODELS_FOR_TUNING_DIR.joinpath("model_example_input.pt")
            shutil.copyfile(src=src_model_example_pt_fn, dst=dst_model_example_pt_fn)
            # then delete the given example so that it doesn't pollute future runs
            os.remove(src_model_example_pt_fn)

    current_weights = model.state_dict()

    if cfg.get("optuna_tuning"):
        results = model.joined_results.reset_index(drop=False)
        metric_result = results[(results["data"] == "val") & (results["type"] == "pairs")]
        max_acc = metric_result["pc_correct"].max()

        tloss = trainer.callback_metrics["val_epoch/total_loss"]
        wandb.finish()

        return (max_acc, tloss)

    ckpt_path = trainer.checkpoint_callback.best_model_path
    if cfg.get("test"):
        log.info("Starting testing!")
        log.info(f"Best ckpt path: {ckpt_path}")

    if ckpt_path == "":
        log.warning("Best ckpt not found! Using current weights for testing...")
        ckpt_path = None
        best_weights = current_weights
    else:
        best_weights = torch.load(ckpt_path)

    trainer.thumbnail_size = cfg.thumbnail_export_size

    checkpoints_for_val = {"ckpt_no_swa": best_weights}
    swa_state_dict = trainer.model.return_swa_weights()

    if cfg.model.get("swa") and cfg.model.swa == True and swa_state_dict is not None:
        checkpoints_for_val.update({"ckpt_swa": swa_state_dict})

    if cfg.get("test"):
        datamodules_for_eval = []  # [datamodule_first,datamodule_third]

        if cfg.get("test_on_third") and cfg.test_on_third:
            datamodule_third = torch.load(dm3_fn)
            datamodules_for_eval.append(datamodule_third)

        for k in checkpoints_for_val.keys():
            sd = checkpoints_for_val[k]

            model.load_state_dict(sd, strict=False)
            model.eval()

            for d in datamodules_for_eval:
                if k == "ckpt_swa":
                    d.dset_version += "_ckpt_swa"

                    model.weights_type = "SWA"

                else:
                    d.dset_version = d.dset_version.replace("_ckpt_swa", "")

                    model.weights_type = "not_SWA"
                # if type(d.data_train) == core_modules.dset_loaders.dset_for_cont_integration:
                # trainer.ddir_func = d.data_train.dsets[0].ddir_func
                # trainer.seed_func = d.data_train.dsets[0].seed_func
                # else:
                trainer.ddir_func = d.data_train.ddir_func
                trainer.seed_func = d.data_train.seed_func

                #

                trainer.test(model=model, datamodule=d)

    # Save final val/test accuracies (with and without goodseeds) into the run dir.
    if cfg.using_wandb and wandb.run is not None:
        try:
            _save_final_metrics(
                model=model,
                trainer=trainer,
                datamodule=datamodule_third,
                best_weights=best_weights,
                collate_fn=collate_fn,
                run_dir=RWD_MODELS_FOR_TUNING_DIR,
            )
        except Exception as e:
            log.exception(f"final-metrics: failed to save final metrics: {e}")

    if cfg.using_wandb and cfg.logger.get("wandb") and cfg.logger.wandb.offline == True and cfg.sync == True and wandb.run is not None:
        current_run_dir = wandb.run.dir
        current_run_id = wandb.run.id

        if not current_run_dir.endswith(current_run_id):
            current_run_dir = os.path.dirname(current_run_dir)

        if "state_dict" in best_weights.keys():
            best_weights = best_weights["state_dict"]

        # save best weights
        best_model_weights_fn = os.path.join(current_run_dir, "best_model.pt")
        torch.save(obj=best_weights, f=best_model_weights_fn)

        # Function to remove hooks
        def remove_hooks(model):
            for module in model.modules():
                if hasattr(module, "_backward_hooks"):
                    module._backward_hooks = OrderedDict()
                if hasattr(module, "_forward_hooks"):
                    module._forward_hooks = OrderedDict()
                if hasattr(module, "_forward_pre_hooks"):
                    module._forward_pre_hooks = OrderedDict()

        dm3_fn = os.path.join(current_run_dir, "datamodule_third.pt")

        # detach from the trainer instance if it has one

        if hasattr(datamodule_third, "trainer"):
            datamodule_third.trainer = None

        torch.save(obj=datamodule_third, f=dm3_fn)

        log.info(f"saved best model weights to: {best_model_weights_fn}")

        if cfg.model.get("swa") and cfg.model.swa == True:
            if "state_dict" in swa_state_dict.keys():
                swa_state_dict = swa_state_dict["state_dict"]

            # save SWA weights
            swa_weights_fn = os.path.join(current_run_dir, "swa_weights.pt")
            torch.save(obj=swa_state_dict, f=swa_weights_fn)

            log.info(f"saved swa best model weights to: {swa_weights_fn}")

        if cfg.sync_model_weights:
            wandb.save(best_model_weights_fn)

        if cfg.model.get("swa") and cfg.model.swa == True and cfg.sync_model_weights:
            wandb.save(swa_weights_fn)

        # now save the run config

        run_cfg_fn = os.path.join(current_run_dir, "run_config.yaml")

        # Save config as a YAML file
        with open(run_cfg_fn, "w") as f:
            OmegaConf.save(config=cfg, f=run_cfg_fn)

        # Log the config file
        wandb.save(run_cfg_fn)

        # Promote the model into the RWD_MODELS_FOR_TUNING archive (keyed by the
        # wandb run id). RWD_MODELS_FOR_TUNING_DIR is only set in the
        # `if cfg.using_wandb` block above, so skip cleanly when wandb tracking
        # is off (e.g. smoke runs with using_wandb=false).
        if cfg.using_wandb:
            yaml_save_fn = os.path.join(RWD_MODELS_FOR_TUNING_DIR, "run_config.yaml")

            shutil.copyfile(src=run_cfg_fn, dst=yaml_save_fn)

            # save best weights
            best_model_weights_fn = os.path.join(RWD_MODELS_FOR_TUNING_DIR, "best_model.pt")
            torch.save(obj=best_weights, f=best_model_weights_fn)

            # copy modeel example input!!
            model_example_input_fn = os.path.join(model_example_dir, "model_example_input.pt")
            model_example_input_fn_to_copy = os.path.join(RWD_MODELS_FOR_TUNING_DIR, "model_example_input.pt")
            shutil.copyfile(src=model_example_input_fn, dst=model_example_input_fn_to_copy)

            if cfg.model.get("swa") and cfg.model.swa == True:
                # save SWA weights
                swa_weights_fn = os.path.join(RWD_MODELS_FOR_TUNING_DIR, "swa_weights.pt")
                torch.save(obj=swa_state_dict, f=swa_weights_fn)

        jr = model.joined_results.reset_index(drop=False)
        jr["pc_correct"] = [np.array(v).flatten().item() for v in jr["pc_correct"]]

        cursed_idx = jr[(jr.epoch == -1) & (jr.pc_correct == -1.0)].index  # row to remove init row

        jr = jr.drop(cursed_idx)

        wandb.log(
            {
                "run_collected_results": wandb.Table(dataframe=jr),
                "epoch": model.trainer.current_epoch,
            }
        )

        jr = model.test_joined_results.reset_index(drop=False)
        jr["pc_correct"] = [np.array(v).flatten().item() for v in jr["pc_correct"]]

        cursed_idx = jr[(jr.epoch == -1) & (jr.pc_correct == -1.0)].index  # row to remove init row

        jr = jr.drop(cursed_idx)
        jr.dset_version = jr.dset_version.apply(lambda f: f.replace("_ckpt_swa", ""))

        jr = jr.sort_values(by="pc_correct", ascending=False)

        try:
            log.info(f"\n-----------------------------------------\n\n{jr.to_markdown()}\n\n--------------------------------------------")
            csv_results_fn = os.path.join(cfg.paths.output_dir, "test_results.csv")
            jr.to_csv(csv_results_fn)

            wandb.log(
                {
                    "run_test_results": wandb.Table(dataframe=jr),
                    "epoch": model.trainer.current_epoch,
                }
            )
        except Exception as e:
            log.exception(f"exception while logging test results: {e}")

        # Close the wandb run before attempting an offline sync to avoid hangs
        wandb.finish(quiet=True)

        # Construct the wandb sync command and enforce a timeout to avoid getting stuck.
        # You can disable automatic sync (e.g., if it hangs) by setting WANDB_SKIP_AUTO_SYNC=1.
        if os.environ.get("WANDB_SKIP_AUTO_SYNC", "1") == "1":
            log.info(f"Skipping automatic wandb sync for run at {current_run_dir}; set WANDB_SKIP_AUTO_SYNC=0 to enable.")
        else:
            command = f"wandb sync {current_run_dir}"
            sync_timeout_s = int(os.environ.get("WANDB_SYNC_TIMEOUT", "120"))
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                stdout, stderr = process.communicate(timeout=sync_timeout_s)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                log.warning(f"wandb sync timed out after {sync_timeout_s}s; stdout: {stdout.decode().strip()}, stderr: {stderr.decode().strip()}")
            else:
                stdout_str = stdout.decode().strip()
                stderr_str = stderr.decode().strip()
                if stdout_str:
                    log.info(stdout_str)
                if stderr_str:
                    log.error(f"wandb sync stderr: {stderr_str}")

    if cfg.using_wandb and wandb.run is not None:
        # Ensure wandb always shuts down cleanly to avoid hanging during artifact handoff
        wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
