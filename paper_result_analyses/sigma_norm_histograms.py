"""Post-normalise_sigma_self histogram comparison across GAN sigma slabs.

For each .mrc slab:
  * load voxels
  * apply normalise_sigma_self exactly:  (x-min)/(max-min)*100
  * also a percentile-clipped variant (clip to [p1,p99] then min-max*100)
    to test whether single-voxel outliers compress the bulk distribution.

Outputs a stats table (CSV) + a multi-panel figure under mrc_exports/histograms.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ROOT = REPO_ROOT / "paper_artifacts" / "mrc_exports"
OUT = ROOT / "histograms"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "EG3D-FFHQ (ref, s29320)": ROOT / "entire_sigma_field_256_s_29320.mrc",
    "SphereHead (s200001)": ROOT / "SphereHead_sigma_seed_200001_rewardslab.mrc",
    "PanoHead (s200001)": ROOT / "PanoHead_sigma_seed_200001_rewardslab.mrc",
    "HyPlaneHead (s200000)": ROOT / "HyPlaneHead_sigma_seed_200000_rewardslab.mrc",
}


def self_norm(x, lo=None, hi=None):
    if lo is None:
        lo, hi = x.min(), x.max()
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo) * 100.0


vols, stats = {}, []
for name, path in FILES.items():
    with mrcfile.open(str(path), permissive=True) as m:
        v = np.asarray(m.data, dtype=np.float64).ravel()
    vols[name] = v
    p1, p99 = np.percentile(v, [1, 99])
    stats.append({
        "model": name, "n_vox": v.size,
        "raw_min": v.min(), "raw_max": v.max(),
        "raw_mean": v.mean(), "raw_median": np.median(v),
        "p1": p1, "p99": p99,
        # after plain self-norm, what fraction of voxels land in [0,5]?
        # high value = bulk compressed into a thin low band by outliers.
        "frac_below5_selfnorm": float((self_norm(v) < 5).mean()),
    })

df = pd.DataFrame(stats)
df.to_csv(OUT / "sigma_norm_stats.csv", index=False)
pd.set_option("display.width", 200, "display.max_columns", 20)
print(df.to_string(index=False))

colors = ["k", "tab:blue", "tab:green", "tab:red"]
bins = np.linspace(0, 100, 101)

fig, axes = plt.subplots(2, 1, figsize=(10, 9))
# panel A: plain self-norm (each cube to its own [0,100])
for (name, v), c in zip(vols.items(), colors):
    axes[0].hist(self_norm(v), bins=bins, histtype="step", log=True,
                 density=True, label=name, color=c, linewidth=1.6)
axes[0].set_title("normalise_sigma_self  (per-cube min-max -> [0,100])  "
                  "— exactly what the reward model sees")
axes[0].set_xlabel("normalised sigma [0,100]")
axes[0].set_ylabel("density (log)")
axes[0].legend(fontsize=8)

# panel B: percentile-clipped [p1,p99] then min-max -> robust to outliers
for (name, v), c in zip(vols.items(), colors):
    p1, p99 = np.percentile(v, [1, 99])
    axes[1].hist(self_norm(v, p1, p99), bins=bins, histtype="step", log=True,
                 density=True, label=name, color=c, linewidth=1.6)
axes[1].set_title("robust variant: clip to [p1,p99] then min-max -> [0,100]  "
                  "(outlier-insensitive)")
axes[1].set_xlabel("normalised sigma [0,100]")
axes[1].set_ylabel("density (log)")
axes[1].legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT / "sigma_norm_histograms.png", dpi=130)
print(f"\n[hist] wrote {OUT / 'sigma_norm_histograms.png'}")
print(f"[hist] wrote {OUT / 'sigma_norm_stats.csv'}")
