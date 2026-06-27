from importlib import import_module

_SUBMODULES = {
    "instantiators",
    "logging_utils",
    "misc_helpers",
    "pylogger_c",
    "rich_utils",
    "camera_utils",
    "depth_to_pcd",
    "meshing_utils",
    "ray_sampling_utils",
    "radiance_field_utils",
    "reward_loading",
    "rlhf_data_utils",
    "rwd_model_utils",
    "visual_tools",
    "finetuning_utils",
    "config_store",
}

__all__ = sorted(_SUBMODULES | {"ConfigStore"})


def __getattr__(name):
    if name == "ConfigStore":
        module = import_module(".config_store", __name__)
        value = module.ConfigStore
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
