"""Within-generator rank consistency: for each of 5 generators, compute
Spearman correlation between per-seed σ_XYZ reward at psi=0.7 and
psi=0.25 on the same 100 latent codes. High Spearman means the reward
produces a STABLE preference ordering of latent codes across truncation
regimes; low Spearman means rank depends on truncation choice.

Also computes the cross-generator Spearman matrix at psi=0.7:
do different generators AGREE on which seeds are best?

Inputs:
  EG3D-orig psi=0.7  : exp3 CSV (model_label=orig, reward_score)
  EG3D-orig psi=0.25 : eg3d_orig_reward_vs_truncation.json (per_seed)
  EG3D-tuned psi=0.7 : exp3 CSV (model_label=tuned)
  EG3D-tuned psi=0.25: eg3d_tuned_reward_vs_truncation.json
  PanoHead psi=0.7   : panohead_trunc0.70/per_seed_rewards.csv
  PanoHead psi=0.25  : panohead_trunc0.25/per_seed_rewards.csv
  HyPlaneHead psi=0.7: hyplanehead trunc0.70 per_seed_rewards.csv
  HyPlaneHead psi=0.25: hyplanehead trunc0.25 per_seed_rewards.csv
  SphereHead psi=0.7 : spherehead trunc0.70 per_seed_rewards.csv
  SphereHead psi=0.25: spherehead trunc0.25 per_seed_rewards.csv

Outputs:
  reward_embedding_analysis/within_generator_rank_consistency/
    summary.json
    spearman_within_generator.csv
    spearman_cross_generator_matrix.csv
    figs: within_generator_rank_spearman.png, cross_generator_spearman_matrix.png,
          per_seed_psi_scatter.png
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_result_analyses.path_defaults import reward_embedding_analysis_dir  # noqa: E402

ANALYSIS_ROOT = reward_embedding_analysis_dir()
OUT = ANALYSIS_ROOT / "within_generator_rank_consistency"


def _eg3d_per_seed(json_path: Path, psi: float) -> pd.DataFrame:
    """Extract per-seed rewards at a given psi from eg3d_*_reward_vs_truncation.json."""
    d = json.load(open(json_path))
    key = f"trunc_psi={psi:.2f}"
    rows = d[key]["per_seed"]
    return pd.DataFrame(rows).rename(columns={"reward": "reward"})


def _exp3_per_seed(model_label: str) -> pd.DataFrame:
    csv = ANALYSIS_ROOT / "exp3_orig_vs_tuned" / "exp3_orig_vs_tuned_samples.csv"
    df = pd.read_csv(csv)
    sub = df[df["model_label"] == model_label][["seed", "reward_score"]].rename(
        columns={"reward_score": "reward"})
    return sub.reset_index(drop=True)


def _csv_per_seed(csv_path: Path, reward_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if reward_col not in df.columns:
        raise SystemExit(f"{csv_path} missing column {reward_col}; has {df.columns.tolist()}")
    return df[["seed", reward_col]].rename(columns={reward_col: "reward"}).reset_index(drop=True)


def gather() -> Dict[str, Dict[float, pd.DataFrame]]:
    """Return nested dict: {generator -> {psi -> DataFrame[seed, reward]}}."""
    data: Dict[str, Dict[float, pd.DataFrame]] = {}
    # EG3D-orig
    data["EG3D-orig"] = {
        0.7:  _exp3_per_seed("orig"),
        0.25: _eg3d_per_seed(
            ANALYSIS_ROOT / "panohead_reward_transfer" / "eg3d_orig_reward_vs_truncation.json",
            0.25,
        ),
    }
    # EG3D-tuned
    data["EG3D-tuned"] = {
        0.7:  _exp3_per_seed("tuned"),
        0.25: _eg3d_per_seed(
            ANALYSIS_ROOT / "panohead_reward_transfer" / "eg3d_tuned_reward_vs_truncation.json",
            0.25,
        ),
    }
    # PanoHead / HyPlaneHead / SphereHead
    for label, root_name, reward_col in [
        ("PanoHead",    "panohead_reward_transfer",    "reward_panohead"),
        ("HyPlaneHead", "hyplanehead_reward_transfer", "reward_hyplanehead"),
        ("SphereHead",  "spherehead_reward_transfer",  "reward_spherehead"),
    ]:
        d_0_7 = _csv_per_seed(ANALYSIS_ROOT / root_name / ("panohead_trunc0.70/per_seed_rewards.csv"
                              if label == "PanoHead" else "trunc0.70/per_seed_rewards.csv"),
                              reward_col)
        d_0_25 = _csv_per_seed(ANALYSIS_ROOT / root_name / ("panohead_trunc0.25/per_seed_rewards.csv"
                               if label == "PanoHead" else "trunc0.25/per_seed_rewards.csv"),
                               reward_col)
        data[label] = {0.7: d_0_7, 0.25: d_0_25}
    return data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = gather()
    gens = list(data.keys())

    # 1) Within-generator: Spearman(reward@0.7, reward@0.25)
    within_rows: List[Dict] = []
    for gen in gens:
        a = data[gen][0.7]
        b = data[gen][0.25]
        merged = a.merge(b, on="seed", suffixes=("_07", "_025"))
        if len(merged) < 5:
            continue
        sp = float(merged[["reward_07", "reward_025"]].corr(method="spearman").iloc[0, 1])
        pr = float(merged[["reward_07", "reward_025"]].corr(method="pearson").iloc[0, 1])
        # Also: do the TOP-10 at psi=0.7 stay in top-half at psi=0.25?
        merged_sorted = merged.sort_values("reward_07", ascending=False).reset_index(drop=True)
        top10_07 = set(merged_sorted.head(10)["seed"])
        top50_025 = set(merged.sort_values("reward_025", ascending=False).head(50)["seed"])
        top10_overlap = len(top10_07 & top50_025) / 10.0
        bot10_07 = set(merged_sorted.tail(10)["seed"])
        bot50_025 = set(merged.sort_values("reward_025", ascending=True).head(50)["seed"])
        bot10_overlap = len(bot10_07 & bot50_025) / 10.0
        within_rows.append({
            "generator": gen,
            "n": int(len(merged)),
            "mean_psi0.7": float(merged["reward_07"].mean()),
            "mean_psi0.25": float(merged["reward_025"].mean()),
            "spearman_psi0.7_vs_0.25": sp,
            "pearson_psi0.7_vs_0.25": pr,
            "top10@0.7_in_top50@0.25": top10_overlap,
            "bot10@0.7_in_bot50@0.25": bot10_overlap,
        })
    within_df = pd.DataFrame(within_rows)
    within_df.to_csv(OUT / "spearman_within_generator.csv", index=False)

    # 2) Cross-generator at psi=0.7: Spearman matrix
    psi = 0.7
    df_wide = None
    for gen in gens:
        sub = data[gen][psi][["seed", "reward"]].rename(columns={"reward": gen})
        df_wide = sub if df_wide is None else df_wide.merge(sub, on="seed")
    mat = df_wide[gens].corr(method="spearman")
    mat.to_csv(OUT / "spearman_cross_generator_matrix.csv")

    # 3) Plots
    # Within-generator: bar chart of Spearman
    fig, ax = plt.subplots(figsize=(10, 4.5))
    xs = np.arange(len(within_df))
    bars = ax.bar(xs, within_df["spearman_psi0.7_vs_0.25"],
                  color=["#1f77b4"]*2 + ["#d62728", "#2ca02c", "#9467bd"])
    for b, v in zip(bars, within_df["spearman_psi0.7_vs_0.25"]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02 if v >= 0 else v - 0.04,
                f"{v:+.2f}", ha="center", fontsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels(within_df["generator"])
    ax.set_ylabel("Spearman rank correlation")
    ax.set_title("Within-generator rank consistency: reward@ψ=0.7 vs reward@ψ=0.25 (n=100 seeds)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(-0.2, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "within_generator_rank_spearman.png", dpi=160)
    plt.close(fig)

    # Cross-generator matrix
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(mat.values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(gens))); ax.set_yticks(range(len(gens)))
    ax.set_xticklabels(gens, rotation=45, ha="right")
    ax.set_yticklabels(gens)
    for i in range(len(gens)):
        for j in range(len(gens)):
            ax.text(j, i, f"{mat.values[i, j]:+.2f}",
                    ha="center", va="center", fontsize=11,
                    color="white" if abs(mat.values[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Spearman rank correlation")
    ax.set_title(f"Cross-generator Spearman matrix at ψ={psi}\n"
                 "do generators agree on which seeds are 'best'?")
    plt.tight_layout()
    plt.savefig(OUT / "cross_generator_spearman_matrix.png", dpi=160)
    plt.close(fig)

    # Per-seed psi=0.7 vs psi=0.25 scatter (small multiples)
    fig, axes = plt.subplots(1, len(gens), figsize=(4*len(gens), 4.5))
    for ax, gen in zip(axes, gens):
        a = data[gen][0.7]; b = data[gen][0.25]
        merged = a.merge(b, on="seed", suffixes=("_07", "_025"))
        ax.scatter(merged["reward_07"], merged["reward_025"], s=14, alpha=0.7, color="#1f77b4")
        sp = float(merged[["reward_07", "reward_025"]].corr(method="spearman").iloc[0, 1])
        ax.set_title(f"{gen}\nSpearman={sp:+.3f}")
        ax.set_xlabel("reward at ψ=0.7"); ax.set_ylabel("reward at ψ=0.25")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "per_seed_psi_scatter.png", dpi=160)
    plt.close(fig)

    summary = {
        "within": within_df.to_dict(orient="records"),
        "cross_matrix": {gens[i]: {gens[j]: float(mat.values[i, j])
                                   for j in range(len(gens))}
                         for i in range(len(gens))},
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Console summary
    print("\n=== Within-generator rank consistency (Spearman psi=0.7 vs psi=0.25) ===")
    print(within_df[["generator", "mean_psi0.7", "mean_psi0.25",
                     "spearman_psi0.7_vs_0.25", "pearson_psi0.7_vs_0.25",
                     "top10@0.7_in_top50@0.25", "bot10@0.7_in_bot50@0.25"]].to_string(index=False))
    print(f"\n=== Cross-generator Spearman matrix at psi=0.7 ===")
    print(mat.round(3).to_string())
    print(f"\n[rank-consistency] outputs in {OUT}")


if __name__ == "__main__":
    main()
