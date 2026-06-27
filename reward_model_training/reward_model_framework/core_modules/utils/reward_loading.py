import os
import pathlib
from typing import Optional

import hydra
import pandas as pd
import torch
from omegaconf import OmegaConf


def _rewrite_legacy_string(value: str) -> str:
    replacements = {
        "src_rlhf.": "core_modules.",
        "/reward_model_framework/src/": "/reward_model_framework/core_modules/",
        "/RLHF_Codebase/src_rlhf/": "/reward_model_framework/core_modules/",
        "/RLHF_Codebase/": "/reward_model_framework/",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def _load_legacy_compatible_config(path: pathlib.Path):
    text = _require_path(path, "RWD_MODELS_DIR", "paths.rwd_model_dir").read_text()
    return OmegaConf.create(_rewrite_legacy_string(text))


def _require_path(path: pathlib.Path, env_var: str, cfg_key: str) -> pathlib.Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; override via {env_var} or Hydra {cfg_key}")
    return path


def _rwd_models_dir() -> pathlib.Path:
    env_val = os.environ.get("RWD_MODELS_DIR")
    if env_val:
        return pathlib.Path(env_val)
    try:
        from core_modules.utils.config_store import ConfigStore

        cfg_store = ConfigStore.instance()
        if cfg_store and getattr(cfg_store, "cfg", None):
            paths = getattr(cfg_store.cfg, "paths", None)
            if paths and getattr(paths, "rwd_model_dir", None):
                return pathlib.Path(paths.rwd_model_dir)
    except Exception:
        pass
    project_root = pathlib.Path(os.environ.get("PROJECT_ROOT", pathlib.Path(__file__).resolve().parents[5]))
    return project_root / "reward_model_training" / "reward_model_framework" / "core_modules" / "RWD_MODELS_FOR_TUNING"


def _runs_summary_path() -> pathlib.Path:
    env_val = os.environ.get("RUNS_SUMMARY_CSV")
    if env_val:
        return pathlib.Path(env_val)
    try:
        from core_modules.utils.config_store import ConfigStore

        cfg_store = ConfigStore.instance()
        if cfg_store and getattr(cfg_store, "cfg", None):
            paths = getattr(cfg_store.cfg, "paths", None)
            if paths and getattr(paths, "run_summary_csv", None):
                return pathlib.Path(paths.run_summary_csv)
    except Exception:
        pass
    project_root = pathlib.Path(os.environ.get("PROJECT_ROOT", pathlib.Path(__file__).resolve().parents[5]))
    return project_root / "reward_model_training" / "reward_model_framework" / "core_modules" / "notebooks" / "runs_summary_for_tune.csv"


def get_datatype_from_model_id(current_id: Optional[int]):
    if current_id is None:
        return None
    rcfg = _rwd_models_dir() / f"{current_id}/run_config.yaml"
    run_config = _load_legacy_compatible_config(rcfg)
    rwd_model_data_type = run_config.data.dset_dict.selected_dtypes[0]

    return rwd_model_data_type


def load_tune_augmentation_from_cfg(current_id: Optional[int]):
    """Return the reward model's recorded `tune` augmentation spec, or None.

    This is the single source of truth for the tune-time transform: eg3d
    fine-tuning instantiates the SAME normalising transform the reward model was
    trained to expect, instead of a hand-copied redeclaration in the eg3d tune
    config that can silently desync (classic train/serve skew).

    The returned value is a detached, fully-resolved OmegaConf node ready for
    `hydra.utils.instantiate`. Interpolations in the saved spec (e.g.
    `${data.augmentations.upper_norm}`) are resolved against the full run config
    before the `tune` subtree is detached, so it instantiates standalone. Legacy
    `src_rlhf.*` targets are rewritten to `core_modules.*` by the loader.
    """
    if current_id is None:
        return None
    rcfg = _rwd_models_dir() / f"{current_id}/run_config.yaml"
    run_config = _load_legacy_compatible_config(rcfg)

    augs = OmegaConf.select(run_config, "data.augmentations")
    if augs is None or OmegaConf.select(run_config, "data.augmentations.tune") is None:
        return None

    # Resolve against the root config (interpolations are absolute, e.g.
    # ${data.augmentations.upper_norm}) THEN detach the tune subtree.
    tune_container = OmegaConf.to_container(run_config.data.augmentations.tune, resolve=True)
    return OmegaConf.create(tune_container)


def get_cfg_fn_from_id(current_id: int) -> str:
    runs_summary = _require_path(_runs_summary_path(), "RUNS_SUMMARY_CSV", "paths.run_summary_csv")
    new_runs_file = pd.read_csv(runs_summary, index_col="ID")
    if current_id not in new_runs_file.index:
        raise KeyError(f"run id {current_id} not found in {runs_summary}")

    rcfg = _rewrite_legacy_string(new_runs_file.loc[current_id].run_config_fn)

    return rcfg


def load_cfg_from_rm_id(current_id: int):
    cfg_fn = get_cfg_fn_from_id(current_id)

    run_config = _load_legacy_compatible_config(pathlib.Path(cfg_fn))

    return run_config


def load_rwd_model_from_cfg_id(current_id: int, strict: bool = True):
    cfg_fn = get_cfg_fn_from_id(current_id)

    runs_summary = _require_path(_runs_summary_path(), "RUNS_SUMMARY_CSV", "paths.run_summary_csv")
    new_runs_file = pd.read_csv(runs_summary, index_col="ID")
    if current_id not in new_runs_file.index:
        raise KeyError(f"run id {current_id} not found in {runs_summary}")

    run_config = _load_legacy_compatible_config(pathlib.Path(cfg_fn))

    if run_config.data.dset_dict.selected_dtypes == ["triple_dmap"] or run_config.data.dset_dict.selected_dtypes == ["single_dmap"]:
        external = run_config.model.external
        external_class = hydra.utils.instantiate(run_config.model.external)
        external_class.eval()

        external_embedding_size = external_class.embedding_size

        run_config.model.mlp_global.input_size = external_embedding_size * run_config.model.n_dmaps
        run_config.model.name = "dummy_name"

        run_config.model.external = None

    run_config.model.optimizer = None
    run_config.model["scheduler"] = None

    rwd_model = hydra.utils.instantiate(run_config.model, _recursive_=False)  # ,partial=False)

    sd_fn = _rewrite_legacy_string(new_runs_file.loc[current_id].best_model_pt_fn)

    state_dict = torch.load(_require_path(pathlib.Path(sd_fn), "RUNS_SUMMARY_CSV", "paths.run_summary_csv"))

    if "state_dict" in state_dict.keys():
        rwd_model.load_state_dict(state_dict["state_dict"], strict=strict)
    else:
        rwd_model.load_state_dict(state_dict, strict=strict)

    if run_config.data.dset_dict.selected_dtypes == ["triple_dmap"] or run_config.data.dset_dict.selected_dtypes == ["single_dmap"]:
        rwd_model.external = external_class

    rwd_model.eval()

    rwd_model = rwd_model.to(torch.device("cuda"))

    return rwd_model


def get_mdir_from_cfg_id(current_id: int):
    cfg_fn = get_cfg_fn_from_id(current_id)
    cm_dir = pathlib.Path(cfg_fn).parent

    return cm_dir


def load_datamodules_from_cfg_id(current_id: int):
    cfg_fn = get_cfg_fn_from_id(current_id)
    cm_dir = pathlib.Path(cfg_fn).parent
    if not cm_dir.exists():
        raise FileNotFoundError(f"{cm_dir} not found; check RUNS_SUMMARY_CSV or Hydra paths.run_summary_csv")

    dm_names = [d for d in cm_dir.glob("datamodule_*.pt")]

    ret_dict = {}
    for d in dm_names:
        rk = d.name.replace(d.suffix, "")
        ret_dict[rk] = torch.load(d)

    return ret_dict


def load_rwd_model_from_cfg(current_id: Optional[int], strict: bool = True):
    if current_id is None:
        dummy_class = torch.nn.Linear(1, 1)
        return dummy_class
    rcfg = _rwd_models_dir() / f"{current_id}/run_config.yaml"
    run_config = _load_legacy_compatible_config(rcfg)

    if run_config.data.dset_dict.selected_dtypes == ["triple_dmap"] or run_config.data.dset_dict.selected_dtypes == ["single_dmap"]:
        external = run_config.model.external
        external_class = hydra.utils.instantiate(run_config.model.external)
        external_class.eval()

        external_embedding_size = external_class.embedding_size

        run_config.model.mlp_global.input_size = external_embedding_size * run_config.model.n_dmaps
        run_config.model.name = "dummy_name"

        run_config.model.external = None

    run_config.model.optimizer = None
    run_config.model["scheduler"] = None

    rwd_model = hydra.utils.instantiate(run_config.model, _recursive_=False)  # ,partial=False)

    sd_fn = _rwd_models_dir() / f"{current_id}/best_model.pt"

    state_dict = torch.load(_require_path(sd_fn, "RWD_MODELS_DIR", "paths.rwd_model_dir"))

    if "state_dict" in state_dict.keys():
        rwd_model.load_state_dict(state_dict["state_dict"], strict=strict)
    else:
        rwd_model.load_state_dict(state_dict, strict=strict)

    if run_config.data.dset_dict.selected_dtypes == ["triple_dmap"]:
        rwd_model.external = external_class

    if run_config.data.dset_dict.selected_dtypes == ["single_dmap"]:
        rwd_model.external = external_class

    rwd_model.eval()

    rwd_model = rwd_model.to(torch.device("cuda"))

    rwd_model.return_global_only = True

    return rwd_model
