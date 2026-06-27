"""Apply the reward-model pads_vals_entire slab crop to full 256^3 sigma cubes
and save as MRC, so the exact volume fed to the reward model can be inspected.

Reproduces MeshUtilsDataClass.get_samples_coordinates_from_pads_vals_dict:
    sam_rs = np.flip(grid, 0)
    slab = sam_rs[rhs:R-lhs, bot:R-top, rear:R-front]
with pads from static_configs/pads_vals_entire.yaml and cpad() rounding.
"""
import sys
from pathlib import Path

import mrcfile
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_result_analyses.path_defaults import reward_embedding_analysis_dir  # noqa: E402

# pads_vals_entire.yaml (shape_res=256)
R = 256
PADS = dict(rhs=0.25, lhs=0.25, bot=0.25, top=0.2, rear=0.4, front=0.1)


def cpad(frac):
    cp = int(frac * R)
    return 1 if cp == 0 else cp


rhs, lhs = cpad(PADS["rhs"]), cpad(PADS["lhs"])
bot, top = cpad(PADS["bot"]), cpad(PADS["top"])
rear, front = cpad(PADS["rear"]), cpad(PADS["front"])

OUTDIR = reward_embedding_analysis_dir() / "mrc_exports"
OUTDIR.mkdir(parents=True, exist_ok=True)

for src_str in sys.argv[1:]:
    src = Path(src_str)
    cube = torch.load(src, map_location="cpu")
    if hasattr(cube, "numpy"):
        cube = cube.float().numpy()
    assert cube.shape == (R, R, R), f"expected {R}^3, got {cube.shape}"

    grid = np.flip(cube, 0)
    slab = grid[rhs:R - lhs, bot:R - top, rear:R - front]
    slab = np.ascontiguousarray(slab.astype(np.float32))

    # tag by parent GAN dir name so the three don't collide on seed name
    gan = src.parent.parent.parent.name  # .../<GAN>/<cubes>/<trunc>/file.pt
    dst = OUTDIR / f"{gan}_{src.stem}_rewardslab.mrc"
    print(f"[slab] {src.name}  full={cube.shape} -> slab={slab.shape}  "
          f"min={slab.min():.3f} max={slab.max():.3f}", flush=True)
    with mrcfile.new_mmap(str(dst), overwrite=True, shape=slab.shape,
                          mrc_mode=2) as mrc:
        mrc.data[:] = slab
    print(f"[slab] wrote {dst}  size_MB={dst.stat().st_size / 1e6:.1f}",
          flush=True)
