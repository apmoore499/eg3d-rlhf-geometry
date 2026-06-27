# Minimal project-root helper to avoid scattered sys.path tweaks.
import os
import sys
from pathlib import Path


def add_root(marker: str = ".project-root") -> Path:
    """Locate the repository root and register the moved reward-model codebase."""
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / marker).exists():
            root = parent
            break
    else:
        root = here.parent

    codebase = root / "reward_model_training" / "reward_model_framework"
    static_configs = root / "reward_model_training" / "static_configs"
    rwd_models_dir = codebase / "core_modules" / "RWD_MODELS_FOR_TUNING"
    runs_summary_csv = codebase / "core_modules" / "notebooks" / "runs_summary_for_tune.csv"

    for path in (root, codebase, root / "eg3d"):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)

    os.environ.setdefault("PROJECT_ROOT", str(root))
    os.environ.setdefault("STATIC_CONFIGS_DIR", str(static_configs))
    os.environ.setdefault("RWD_MODELS_DIR", str(rwd_models_dir))
    os.environ.setdefault("RUNS_SUMMARY_CSV", str(runs_summary_csv))

    return root


add_root()
