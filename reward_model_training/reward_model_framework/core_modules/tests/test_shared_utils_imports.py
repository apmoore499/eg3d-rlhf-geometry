import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import autoroot  # noqa: F401


def test_shared_utils_imports():
    """Smoke test that shared helper modules import successfully."""
    mods = [
        "core_modules.utils.camera_utils",
        "core_modules.utils.depth_to_pcd",
        "core_modules.utils.meshing_utils",
        "core_modules.utils.ray_sampling_utils",
        "core_modules.utils.radiance_field_utils",
        "core_modules.utils.reward_loading",
    ]
    for m in mods:
        importlib.import_module(m)


test_shared_utils_imports()
