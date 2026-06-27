"""Compatibility shim for legacy `data.custom_transforms` Hydra targets."""

from pathlib import Path
import sys


_REWARD_FRAMEWORK_ROOT = Path(__file__).resolve().parents[2] / "reward_model_training" / "reward_model_framework"
if str(_REWARD_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_REWARD_FRAMEWORK_ROOT))

from core_modules.data.custom_transforms import *  # noqa: F401,F403
