import os
from pathlib import Path

import torch


def _resolve_framework_root() -> Path:
    project_root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[5]))
    if project_root.name == "reward_model_framework":
        return project_root
    candidate = project_root / "reward_model_training" / "reward_model_framework"
    if candidate.exists():
        return candidate
    return project_root


def get_static_configs_dir() -> Path:
    """Resolve static config directory from env/Hydra, falling back to legacy default."""
    env_val = os.environ.get("STATIC_CONFIGS_DIR")
    if env_val:
        return Path(env_val)
    try:
        from core_modules.utils.config_store import ConfigStore

        cfg_store = ConfigStore.instance()
        if cfg_store and getattr(cfg_store, "cfg", None):
            paths = getattr(cfg_store.cfg, "paths", None)
            if paths and getattr(paths, "static_configs_dir", None):
                return Path(paths.static_configs_dir)
    except Exception:
        pass
    framework_root = _resolve_framework_root()
    if framework_root.name == "reward_model_framework":
        return framework_root.parent / "static_configs"
    return framework_root / "reward_model_training" / "static_configs"


def triple_dmap_cameras_path() -> Path:
    return get_static_configs_dir() / "triple_dmap_cameras.pt"


def get_canonical_dmap_cams_for_rlhf():
    """Return canonical dmap camera matrices (single view)."""
    tdmap_cams = torch.load(triple_dmap_cameras_path(), map_location=torch.device("cpu"))
    canon_cam = tdmap_cams[1].unsqueeze(0)
    c = canon_cam
    cam2world_matrix = c[:, :16].view(-1, 4, 4)
    intrinsics = c[:, 16:25].view(-1, 3, 3)
    return dict(cam2world_matrix=cam2world_matrix, intrinsics=intrinsics, gen_c=c)


def get_triple_dmap_cams_for_rlhf():
    """Return all three dmap camera matrices."""
    tdmap_cams = torch.load(triple_dmap_cameras_path(), map_location=torch.device("cpu"))

    intrinsics_list = []
    gen_c_list = []
    c2w_mat_list = []

    for i in range(3):
        canon_cam = tdmap_cams[i].unsqueeze(0)
        c = canon_cam
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        intrinsics_list.append(intrinsics)
        gen_c_list.append(c)
        c2w_mat_list.append(c)

    return dict(cam2world_matrix=c2w_mat_list, intrinsics=intrinsics_list, gen_c=gen_c_list)


def get_triple_dmap_cameras():
    """Stacked gen_c template used in training loops."""
    gen_c_template = get_triple_dmap_cams_for_rlhf()["gen_c"]
    gen_c_template = torch.vstack(gen_c_template)
    return gen_c_template


def get_single_dmap_camera():
    """Canonical single dmap camera."""
    gen_c_template = get_canonical_dmap_cams_for_rlhf()["gen_c"]
    return gen_c_template
