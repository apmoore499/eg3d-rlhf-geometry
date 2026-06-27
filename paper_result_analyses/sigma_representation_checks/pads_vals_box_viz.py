"""Visualise the pads_vals_entire crop box over a full 256^3 sigma field, to
confirm (L1169/1186/1196) that the slab fed to the reward model removes the
background and keeps the face region.

Draws orthogonal mid-slices of the FULL cube with the crop rectangle
[64:192, 64:205, 102:231] (rhs/lhs=64, bot=64/top=51, rear=102/front=25 at
shape_res=256) overlaid, plus the cropped slab beside each.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_result_analyses.path_defaults import spherehead_root  # noqa: E402

R = 256
# pads_vals_entire crop indices (see sigma_norm pipeline)
X0, X1 = 64, 192   # axis0 horizontal (rhs : R-lhs)
Y0, Y1 = 64, 205   # axis1 vertical   (bot : R-top)
Z0, Z1 = 102, 231  # axis2 depth      (rear: R-front)

OUT = REPO_ROOT / "paper_artifacts" / "mrc_exports" / "histograms"
OUT.mkdir(parents=True, exist_ok=True)

# Full 256^3 cube on hand (geometry of the box is model-independent; SphereHead
# full cube used purely to show what the crop keeps vs discards).
SRC = (
    spherehead_root()
    / "spherehead_sigma_cubes_for_reward"
    / "trunc0.70"
    / "sigma_seed_200001.pt"
)

cube = torch.load(SRC, map_location="cpu").float().numpy()
cube = np.flip(cube, 0)  # training orientation (flip axis0)
assert cube.shape == (R, R, R)

# clip for display contrast
disp = np.clip(cube, 0, np.percentile(cube, 99))

fig, axes = plt.subplots(2, 3, figsize=(13, 9))

# Row 0: full cube mid-slices with crop box
# axial (fix axis0 mid): plane spans (axis1=Y vertical, axis2=Z depth)
axes[0, 0].imshow(disp[R // 2].T, origin="lower", cmap="magma", aspect="auto")
axes[0, 0].add_patch(Rectangle((Y0, Z0), Y1 - Y0, Z1 - Z0, fill=False,
                               edgecolor="cyan", linewidth=2))
axes[0, 0].set_title("slice axis0=128  (x=vert, y=depth)\nbox = Y[64:205], Z[102:231]")

# fix axis1 mid: plane spans (axis0=X horiz, axis2=Z depth)
axes[0, 1].imshow(disp[:, R // 2].T, origin="lower", cmap="magma", aspect="auto")
axes[0, 1].add_patch(Rectangle((X0, Z0), X1 - X0, Z1 - Z0, fill=False,
                               edgecolor="cyan", linewidth=2))
axes[0, 1].set_title("slice axis1=128  (x=horiz, y=depth)\nbox = X[64:192], Z[102:231]")

# fix axis2 mid: plane spans (axis0=X horiz, axis1=Y vertical)
axes[0, 2].imshow(disp[:, :, R // 2].T, origin="lower", cmap="magma", aspect="auto")
axes[0, 2].add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fill=False,
                               edgecolor="cyan", linewidth=2))
axes[0, 2].set_title("slice axis2=128  (x=horiz, y=vert)\nbox = X[64:192], Y[64:205]")

# Row 1: cropped slab mid-slices (what the reward actually sees)
slab = cube[X0:X1, Y0:Y1, Z0:Z1]
sd = np.clip(slab, 0, np.percentile(slab, 99))
sx, sy, sz = slab.shape
axes[1, 0].imshow(sd[sx // 2].T, origin="lower", cmap="magma", aspect="auto")
axes[1, 0].set_title(f"CROPPED slab axis0={sx // 2}")
axes[1, 1].imshow(sd[:, sy // 2].T, origin="lower", cmap="magma", aspect="auto")
axes[1, 1].set_title(f"CROPPED slab axis1={sy // 2}")
axes[1, 2].imshow(sd[:, :, sz // 2].T, origin="lower", cmap="magma", aspect="auto")
axes[1, 2].set_title(f"CROPPED slab axis2={sz // 2}  shape={slab.shape}")

fig.suptitle("pads_vals_entire crop over full 256^3 sigma field "
             "(cyan = region kept; outside = discarded background)",
             fontsize=13)
fig.tight_layout()
out = OUT / "pads_vals_crop_box.png"
fig.savefig(out, dpi=120)
print(f"[viz] wrote {out}")
print(f"[viz] full cube {cube.shape} -> slab {slab.shape}  "
      f"kept {slab.size / cube.size * 100:.1f}% of voxels")
