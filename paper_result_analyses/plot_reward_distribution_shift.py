"""Plot the change in σ_XYZ reward distribution from EG3D-orig to EG3D-tuned
on the same 100 latent codes (exp3 bank). Reproduces the histogram used as
Figure fig:reward_hist in §4.3 of main.tex.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_result_analyses.path_defaults import (  # noqa: E402
    generated_figure_dir,
    reward_embedding_analysis_dir,
)

SAMPLES_CSV = (
    reward_embedding_analysis_dir() / "exp3_orig_vs_tuned" / "exp3_orig_vs_tuned_samples.csv"
)
OUT_PATH = generated_figure_dir() / "fig_reward_hist.png"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SAMPLES_CSV)
    orig = df.loc[df["model_label"] == "orig", "reward_score"].to_numpy(dtype=np.float64)
    tuned = df.loc[df["model_label"] == "tuned", "reward_score"].to_numpy(dtype=np.float64)
    delta = tuned - orig if len(orig) == len(tuned) else None

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    bins = np.linspace(min(orig.min(), tuned.min()) - 0.5, max(orig.max(), tuned.max()) + 0.5, 30)
    axes[0].hist(orig, bins=bins, alpha=0.7, color="#377eb8", label=f"orig (mean={orig.mean():+.2f})")
    axes[0].hist(tuned, bins=bins, alpha=0.7, color="#e41a1c", label=f"tuned (mean={tuned.mean():+.2f})")
    axes[0].axvline(orig.mean(), color="#377eb8", linewidth=1.5, linestyle="--")
    axes[0].axvline(tuned.mean(), color="#e41a1c", linewidth=1.5, linestyle="--")
    axes[0].set_xlabel(r"$\sigma_{XYZ}$ reward score")
    axes[0].set_ylabel("count of samples")
    axes[0].set_title(r"Reward distribution at $\psi=0.7$, $n=100$ paired latents")
    axes[0].legend(loc="upper left", frameon=True)
    axes[0].grid(True, alpha=0.3)

    if delta is not None:
        bins_d = np.linspace(delta.min() - 0.5, delta.max() + 0.5, 30)
        axes[1].hist(delta, bins=bins_d, color="#4daf4a", alpha=0.8)
        axes[1].axvline(0.0, color="black", linewidth=1.0)
        axes[1].axvline(delta.mean(), color="#4daf4a", linewidth=1.5, linestyle="--",
                        label=f"mean Δ = {delta.mean():+.2f}")
        axes[1].set_xlabel(r"Per-seed reward delta  $r_\theta(G_{r_\theta^*}(z)) - r_\theta(G(z))$")
        axes[1].set_ylabel("count of seeds")
        axes[1].set_title(rf"All $100/100$ deltas positive (median {np.median(delta):+.2f})")
        axes[1].legend(loc="upper left", frameon=True)
        axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=160)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
