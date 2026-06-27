from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EG3D_ROOT = REPO_ROOT / "eg3d"
RLHF_CORE_ROOT = (
    REPO_ROOT / "reward_model_training" / "reward_model_framework" / "core_modules"
)
EXTERNAL_PROJECTS_ROOT = Path(
    os.environ.get("EG3D_EXTERNAL_PROJECTS_ROOT", REPO_ROOT.parents[1])
).expanduser()

DEFAULT_REPORTED_RUN_NAME = "01446-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20"
DEFAULT_NOREWARD_RUN_NAME = "01447-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20"
DEFAULT_DATASET_NAME = "eg3d_w_mirrore.zip"


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser()


def training_runs_root() -> Path:
    return _env_path("EG3D_RLHF_TRAINING_RUNS_DIR") or (
        REPO_ROOT / "paper_artifacts" / "_external_training_runs"
    )


def reported_run_dir(
    run_name: str = DEFAULT_REPORTED_RUN_NAME,
    explicit_env: str = "EG3D_RLHF_REPORTED_RUN_DIR",
) -> Path:
    return _env_path(explicit_env) or (training_runs_root() / run_name)


def reward_embedding_analysis_dir(
    run_name: str = DEFAULT_REPORTED_RUN_NAME,
    explicit_env: str = "EG3D_RLHF_REPORTED_RUN_DIR",
) -> Path:
    return reported_run_dir(run_name=run_name, explicit_env=explicit_env) / (
        "reward_embedding_analysis"
    )


def generated_figure_dir() -> Path:
    return _env_path("EG3D_RLHF_PAPER_FIG_DIR") or (
        REPO_ROOT / "paper_artifacts" / "generated_figures"
    )


def dataset_zip_path(default_name: str = DEFAULT_DATASET_NAME) -> Path:
    return _env_path("EG3D_RLHF_DATASET_ZIP") or (
        REPO_ROOT / "paper_artifacts" / "_external_data" / default_name
    )


def external_repo_dir(project_name: str) -> Path:
    env_name = f"{project_name.upper()}_ROOT"
    return _env_path(env_name) or (EXTERNAL_PROJECTS_ROOT / project_name)


def panohead_root() -> Path:
    return external_repo_dir("PanoHead")


def hyplanehead_root() -> Path:
    return external_repo_dir("HyPlaneHead")


def spherehead_root() -> Path:
    return external_repo_dir("SphereHead")
