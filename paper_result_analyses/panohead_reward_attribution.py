"""Per-seed Integrated-Gradients attribution of the σ_XYZ reward on PanoHead
σ-cubes, decomposed by PanoHead-aligned WFLW-98 anatomical region.

Baseline is the zero-σ cube; integration target is each seed's PanoHead σ
cube (cropped to pads_vals_entire). Output:
  reward_embedding_analysis/panohead_reward_attribution/
    per_seed_ig_by_region.csv
    summary.json
    top10_vs_bot10_region_contribution.png

We then split seeds into top-10 / bottom-10 by σ_XYZ reward and compare
mean IG contribution per region — testing the user's hypothesis that
eye-orbit geometry attribution discriminates the reward tails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_result_analyses.path_defaults import (  # noqa: E402
    EG3D_ROOT,
    RLHF_CORE_ROOT,
    panohead_root,
    reward_embedding_analysis_dir,
)

RLHF_SRC_ROOT = RLHF_CORE_ROOT
for _p in (REPO_ROOT, EG3D_ROOT, RLHF_SRC_ROOT.parent):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

try:
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
except Exception:
    pass

import hydra  # noqa: E402

from core_modules.data.create_train_data import generation_utils as gen_utils  # noqa: E402
from core_modules.utils import finetuning_utils, reward_loading  # noqa: E402

PSI = 0.70
TRUNC_STR = f"{PSI:.2f}"
REWARD_ID = "7wnzkgie"
SHAPE_RES = 256
IG_STEPS = 8

PANO_SIGMA_ROOT = panohead_root() / "panohead_sigma_cubes_for_reward" / f"trunc{TRUNC_STR}"
MASKS_PT = reward_embedding_analysis_dir() / "panohead_aw98_template_masks" / "region_masks.pt"
REWARD_CSV = (
    reward_embedding_analysis_dir()
    / "panohead_reward_transfer"
    / f"panohead_trunc{TRUNC_STR}"
    / "per_seed_rewards.csv"
)
OUT_DIR = reward_embedding_analysis_dir() / "panohead_reward_attribution" / f"trunc{TRUNC_STR}"


def _crop_full_to_pads(full_cube_xyz: torch.Tensor, tri_idx, full_res: int) -> torch.Tensor:
    rhs = int(tri_idx.ax1horiz.right)
    lhs = int(full_res - tri_idx.ax1horiz.left)
    bot = int(tri_idx.ax2vert.bottom)
    top = int(full_res - tri_idx.ax2vert.top)
    rear = int(tri_idx.ax3depth.rear)
    front = int(full_res - tri_idx.ax3depth.front)
    return full_cube_xyz[rhs:full_res - lhs, bot:full_res - top, rear:full_res - front].clone()


def _reward_forward(reward_model, sigma_aug, vol_xyz: torch.Tensor):
    aug = sigma_aug(vol_xyz)
    inp = aug.permute(2, 1, 0).contiguous().unsqueeze(0)
    emb8192 = reward_model.Conv3DModule.forward_to_global_vec(inp, return_global_only=True)
    emb512 = reward_model.MLP(emb8192)
    return reward_model.forward_to_scalar_reward_from_single_global(emb512).reshape(())


def _integrated_gradients(reward_model, sigma_aug,
                          vol_xyz: torch.Tensor, device: torch.device,
                          steps: int = IG_STEPS) -> torch.Tensor:
    """IG from zero baseline to vol_xyz, returns same-shape attribution tensor."""
    x1 = vol_xyz.to(device)
    x0 = torch.zeros_like(x1)
    delta = x1 - x0
    total_grad = torch.zeros_like(x0)
    for alpha in torch.linspace(1.0 / steps, 1.0, steps, device=device):
        x = (x0 + alpha * delta).clone().detach().requires_grad_(True)
        r = _reward_forward(reward_model, sigma_aug, x)
        grad = torch.autograd.grad(r, x, retain_graph=False, create_graph=False)[0]
        total_grad += grad.detach()
    ig = (delta * total_grad / float(steps)).detach().cpu()
    return ig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    print(f"[attr] loading reward {REWARD_ID}")
    reward_model = reward_loading.load_rwd_model_from_cfg(REWARD_ID).to(device).eval()
    run_cfg = OmegaConf.load(
        RLHF_SRC_ROOT / "RWD_MODELS_FOR_TUNING" / REWARD_ID / "run_config.yaml"
    )
    sigma_aug = hydra.utils.instantiate(run_cfg.data.augmentations.sigma_norm).eval()
    if hasattr(sigma_aug, "to"):
        sigma_aug = sigma_aug.to(device)

    print(f"[attr] loading PanoHead masks from {MASKS_PT}")
    masks_payload = torch.load(MASKS_PT, map_location="cpu")
    masks: Dict[str, torch.Tensor] = masks_payload["masks"]
    region_priority: List[str] = list(masks_payload["region_priority"])
    cube_shape = tuple(int(x) for x in masks_payload["cube_shape"])

    # Load EG3D-orig just to get pads_vals_entire tri_idx for cropping the
    # raw PanoHead σ-cubes from 256³ → cube_shape.
    print("[attr] loading EG3D-orig to get pads_vals_entire crop indices")
    da = gen_utils.load_generator(
        model_path=REPO_ROOT / "pkl_pt" / "eg3d_1" / "ffhq512-128.pkl",
        truncation_psi=0.7, truncation_cutoff=14, shape_res=SHAPE_RES,
    )
    mudc = finetuning_utils.MeshUtilsDataClass()
    pads = OmegaConf.load(gen_utils.STATIC_CONFIGS_DIR / "pads_vals_entire.yaml")
    _samples, _shape, tri_idx = mudc.get_samples_coordinates_from_pads_vals_dict(
        pads_vals=pads, G=da.G, shape_res=SHAPE_RES,
    )

    masks_dev = {k: v.to(device) for k, v in masks.items()}

    rdf = pd.read_csv(REWARD_CSV)
    seeds = sorted(rdf["seed"].astype(int).tolist())

    rows: List[Dict] = []
    for seed in tqdm(seeds, desc="IG per seed"):
        full_pt = PANO_SIGMA_ROOT / f"sigma_seed_{seed}.pt"
        if not full_pt.exists():
            print(f"  missing {full_pt}, skipping")
            continue
        full = torch.load(full_pt, map_location="cpu").float()  # (256, 256, 256)
        vol = _crop_full_to_pads(full, tri_idx, SHAPE_RES)
        if tuple(vol.shape) != cube_shape:
            raise RuntimeError(
                f"cube shape mismatch: cropped {tuple(vol.shape)} vs masks {cube_shape}"
            )
        vol_dev = vol.to(device)
        ig = _integrated_gradients(reward_model, sigma_aug, vol_dev, device).to(device)
        with torch.no_grad():
            r = float(_reward_forward(reward_model, sigma_aug, vol_dev).cpu())
        row = {"seed": int(seed), "reward_panohead": r,
               "ig_total_signed": float(ig.sum().cpu()),
               "ig_total_abs": float(ig.abs().sum().cpu())}
        for region in region_priority:
            m = masks_dev[region]
            row[f"ig_signed_{region}"] = float(ig[m].sum().cpu())
            row[f"ig_abs_{region}"] = float(ig[m].abs().sum().cpu())
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_seed_ig_by_region.csv", index=False)
    print(f"[attr] saved per_seed_ig_by_region.csv ({len(df)} rows)")

    # Compare top-10 vs bottom-10 by reward
    df_sorted = df.sort_values("reward_panohead", ascending=False).reset_index(drop=True)
    top = df_sorted.head(10)
    bot = df_sorted.tail(10)

    # Per-region: mean signed IG and abs IG in each group
    rows2 = []
    for region in region_priority:
        col_s = f"ig_signed_{region}"
        col_a = f"ig_abs_{region}"
        rows2.append({
            "region": region,
            "top10_mean_signed": float(top[col_s].mean()),
            "bot10_mean_signed": float(bot[col_s].mean()),
            "delta_top_minus_bot_signed": float(top[col_s].mean() - bot[col_s].mean()),
            "top10_mean_abs": float(top[col_a].mean()),
            "bot10_mean_abs": float(bot[col_a].mean()),
            "delta_top_minus_bot_abs": float(top[col_a].mean() - bot[col_a].mean()),
            "all_mean_signed": float(df[col_s].mean()),
            "spearman_with_reward_signed": float(df[[col_s, "reward_panohead"]].corr(method="spearman").iloc[0, 1]),
        })
    region_df = pd.DataFrame(rows2).sort_values("delta_top_minus_bot_signed", ascending=False).reset_index(drop=True)
    region_df.to_csv(OUT_DIR / "region_top10_vs_bot10.csv", index=False)
    summary = {
        "n_seeds": int(len(df)),
        "top10_reward_range": [float(top["reward_panohead"].min()), float(top["reward_panohead"].max())],
        "bot10_reward_range": [float(bot["reward_panohead"].min()), float(bot["reward_panohead"].max())],
        "top10_mean_signed_ig_total": float(top["ig_total_signed"].mean()),
        "bot10_mean_signed_ig_total": float(bot["ig_total_signed"].mean()),
        "region_table": region_df.to_dict(orient="records"),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[attr] saved region_top10_vs_bot10.csv + summary.json")

    # Plot
    fig, ax = plt.subplots(figsize=(13, 5))
    regions_plot = [r for r in region_priority if r not in ("other",)]
    xs = np.arange(len(regions_plot))
    width = 0.35
    top_vals = [region_df.set_index("region").loc[r, "top10_mean_signed"] for r in regions_plot]
    bot_vals = [region_df.set_index("region").loc[r, "bot10_mean_signed"] for r in regions_plot]
    ax.bar(xs - width / 2, top_vals, width, label="top-10 reward", color="#3aaa3a")
    ax.bar(xs + width / 2, bot_vals, width, label="bot-10 reward", color="#cc3333")
    ax.set_xticks(xs)
    ax.set_xticklabels(regions_plot, rotation=45, ha="right")
    ax.set_ylabel("mean signed IG contribution")
    ax.set_title(f"PanoHead reward attribution by region — top-10 vs bottom-10 of {len(df)} seeds")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "top10_vs_bot10_region_contribution.png", dpi=150)
    plt.close(fig)
    print(f"[attr] saved top10_vs_bot10_region_contribution.png to {OUT_DIR}")


if __name__ == "__main__":
    main()
