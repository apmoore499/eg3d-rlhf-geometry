"""Score σ-cubes from any external EG3D-family generator (PanoHead,
HyPlaneHead, SphereHead, etc.) with the 7wnzkgie σ_XYZ reward.

Inputs are pre-extracted full 256³ σ tensors saved as:
  {SIGMA_ROOT}/sigma_seed_{seed}.pt

The script:
 * Loads EG3D-orig to derive the pads_vals_entire crop indices (these only
   depend on box_warp=1, shared with our PanoHead/HyPlaneHead/SphereHead
   extractions),
 * Loads the 7wnzkgie reward + normalise_sigma_self augmentation,
 * Crops each external σ-cube to the same slab convention,
 * Applies the reward forward pass per seed,
 * Writes per_seed_rewards.csv + summary.json under {OUT_DIR}.

Usage:
  python paper_result_analyses/score_external_sigma_cubes.py \\
      --sigma-root "${HYPLANEHEAD_ROOT}/hyplanehead_sigma_cubes_for_reward/trunc0.70" \\
      --out-dir reward_embedding_analysis/hyplanehead_reward_transfer/\\
trunc0.70 \\
      --label hyplanehead
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
EG3D_ROOT = REPO_ROOT / "eg3d"
RLHF_SRC_ROOT = REPO_ROOT / "reward_model_training" / "reward_model_framework" / "core_modules"
for _p in (REPO_ROOT, EG3D_ROOT, RLHF_SRC_ROOT.parent):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

try:
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
except Exception:
    pass

import hydra  # noqa: E402

import core_modules  # noqa: E402
from core_modules.data.create_train_data import generation_utils as gen_utils  # noqa: E402
from core_modules.utils import finetuning_utils, reward_loading  # noqa: E402

# The reward run_config.yaml references transforms under the training-time
# package name `src_rlhf`; on disk that package is `core_modules`. Alias it so
# hydra.instantiate can resolve `src_rlhf.data.custom_transforms.*`.
sys.modules.setdefault("src_rlhf", core_modules)

REWARD_ID = "7wnzkgie"
SHAPE_RES = 256
TRUNC_PSI = 0.7
TRUNC_CUT = 14
NOISE_MODE = "const"
MAX_BATCH = 1_000_000


def _crop_full_to_pads(full_cube: torch.Tensor, tri_idx, full_res: int) -> torch.Tensor:
    rhs = int(tri_idx.ax1horiz.right)
    lhs = int(full_res - tri_idx.ax1horiz.left)
    bot = int(tri_idx.ax2vert.bottom)
    top = int(full_res - tri_idx.ax2vert.top)
    rear = int(tri_idx.ax3depth.rear)
    front = int(full_res - tri_idx.ax3depth.front)
    return full_cube[rhs:full_res - lhs, bot:full_res - top, rear:full_res - front].clone()


def _reward_forward(reward_model, sigma_aug, vol_xyz: torch.Tensor) -> float:
    aug = sigma_aug(vol_xyz)
    inp = aug.permute(2, 1, 0).contiguous().unsqueeze(0)
    with torch.no_grad():
        emb8192 = reward_model.Conv3DModule.forward_to_global_vec(inp, return_global_only=True)
        emb512 = reward_model.MLP(emb8192)
        scalar = reward_model.forward_to_scalar_reward_from_single_global(emb512)
    return float(scalar.reshape(()).detach().cpu().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-root", required=True,
                    help="dir containing sigma_seed_{seed}.pt files (256^3 each)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="external",
                    help="column-name suffix for reward")
    ap.add_argument("--pre-cropped", action="store_true", default=False,
                    help="input σ tensors are already cropped to "
                         "pads_vals_entire region (128x141x129); skip "
                         "internal cropping and just convert dtype")
    ap.add_argument("--robust-norm", action="store_true", default=False,
                    help="clip each cropped σ cube to its [lo,hi] percentiles "
                         "BEFORE the normalise_sigma_self min-max, so a few "
                         "extreme voxels can't compress the bulk into [0,5]. "
                         "Reproduces the 'robust variant' histogram.")
    ap.add_argument("--clip-pct", default="1,99",
                    help="lower,upper percentiles for --robust-norm "
                         "(default 1,99)")
    ap.add_argument("--glob", default="sigma_seed_*.pt",
                    help="filename glob for input cubes (default "
                         "sigma_seed_*.pt; use 'entire_sigma_field_256_s_*.pt' "
                         "for the EG3D training cubes)")
    ap.add_argument("--max-n", type=int, default=0,
                    help="if >0, score only the first N matched files")
    ap.add_argument("--flip-axis0", action="store_true", default=False,
                    help="np.flip(axis 0) each cube before scoring. The reward "
                         "training cubes are saved flipped (post "
                         "get_samples_coordinates_from_pads_vals_dict); the "
                         "external-GAN path crops unflipped. Use this to put "
                         "the two in the SAME orientation when comparing.")
    args = ap.parse_args()
    clip_lo, clip_hi = (float(x) for x in args.clip_pct.split(","))

    sigma_root = Path(args.sigma_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    print(f"[score] loading reward {REWARD_ID}")
    reward_model = reward_loading.load_rwd_model_from_cfg(REWARD_ID).to(device).eval()
    run_cfg = OmegaConf.load(
        RLHF_SRC_ROOT / "RWD_MODELS_FOR_TUNING" / REWARD_ID / "run_config.yaml"
    )
    sigma_aug = hydra.utils.instantiate(run_cfg.data.augmentations.sigma_norm).eval()
    if hasattr(sigma_aug, "to"):
        sigma_aug = sigma_aug.to(device)

    print("[score] loading EG3D-orig for pads_vals_entire crop indices")
    da = gen_utils.load_generator(
        model_path=REPO_ROOT / "pkl_pt" / "eg3d_1" / "ffhq512-128.pkl",
        truncation_psi=TRUNC_PSI, truncation_cutoff=TRUNC_CUT, shape_res=SHAPE_RES,
    )
    mudc = finetuning_utils.MeshUtilsDataClass()
    pads = OmegaConf.load(gen_utils.STATIC_CONFIGS_DIR / "pads_vals_entire.yaml")
    _samples, shape, tri_idx = mudc.get_samples_coordinates_from_pads_vals_dict(
        pads_vals=pads, G=da.G, shape_res=SHAPE_RES,
    )
    cube_shape = tuple(int(x) for x in shape[1:4])
    print(f"[score] cropped cube shape = {cube_shape}")

    reward_col = f"reward_{args.label}"
    rows: List[Dict] = []
    pt_files = sorted(sigma_root.glob(args.glob))
    if not pt_files:
        raise SystemExit(f"no files matching {args.glob} under {sigma_root}")
    if args.max_n > 0:
        pt_files = pt_files[:args.max_n]
    for p in tqdm(pt_files, desc=f"scoring {args.label}"):
        m = re.search(r"(-?\d+)$", p.stem)  # trailing integer = seed
        seed = int(m.group(1)) if m else -1
        loaded = torch.load(p, map_location="cpu").float()
        if args.flip_axis0:
            loaded = torch.flip(loaded, dims=[0])
        if args.pre_cropped:
            cropped = loaded
        else:
            cropped = _crop_full_to_pads(loaded, tri_idx, SHAPE_RES)
        if tuple(cropped.shape) != cube_shape:
            raise RuntimeError(
                f"shape mismatch for {p}: cropped {tuple(cropped.shape)} vs "
                f"expected {cube_shape}")
        if args.robust_norm:
            lo, hi = torch.quantile(
                cropped.flatten(),
                torch.tensor([clip_lo / 100.0, clip_hi / 100.0]))
            cropped = cropped.clamp(float(lo), float(hi))
        r = _reward_forward(reward_model, sigma_aug, cropped.to(device))
        rows.append({"seed": seed, reward_col: r})

    df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    df.to_csv(out_dir / "per_seed_rewards.csv", index=False)
    arr = df[reward_col].to_numpy(dtype=np.float64)
    summary = {
        "label": args.label,
        "sigma_root": str(sigma_root),
        "robust_norm": bool(args.robust_norm),
        "clip_pct": args.clip_pct if args.robust_norm else None,
        "flip_axis0": bool(args.flip_axis0),
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[score] CSV → {out_dir / 'per_seed_rewards.csv'}")


if __name__ == "__main__":
    main()
