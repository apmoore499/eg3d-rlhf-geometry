"""Score PanoHead σ-cubes with the 7wnzkgie reward model and compare against
EG3D-orig and EG3D-tuned distributions on the same 100 latent seeds.

PanoHead σ-cubes are produced by PanoHead/extract_sigmas_for_reward_transfer.py
in PanoHead's own conda env. They live as full 256³ cubes at
voxel_origin=[0,0,0], cube_length=box_warp=1.0 — physically the same world-coord
grid that EG3D uses, so we can crop them with EG3D's pads_vals_entire tripleaxis
indices and feed the cropped tensor straight into the reward model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

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
    reported_run_dir,
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
from core_modules.utils import finetuning_utils  # noqa: E402
from core_modules.utils import reward_loading  # noqa: E402

REWARD_ID = "7wnzkgie"
ORIG_PKL = REPO_ROOT / "pkl_pt" / "eg3d_1" / "ffhq512-128.pkl"
TUNED_PKL = reported_run_dir() / "network-snapshot-002068_LAST.pkl"
PANOHEAD_SIGMA_ROOT = panohead_root() / "panohead_sigma_cubes_for_reward"
SHAPE_RES = 256
TRUNC_PSI = 0.7
TRUNC_CUT = 14
NOISE_MODE = "const"
MAX_BATCH = 1_000_000
OUT_DIR_ROOT = reward_embedding_analysis_dir() / "panohead_reward_transfer"
SEEDS = list(range(200000, 200100))


def _load_reward(device: torch.device):
    """Load 7wnzkgie reward + the σ-augmentation it was trained with."""
    model = reward_loading.load_rwd_model_from_cfg(REWARD_ID).to(device).eval()
    cfg_path = RLHF_SRC_ROOT / "RWD_MODELS_FOR_TUNING" / REWARD_ID / "run_config.yaml"
    run_config = OmegaConf.load(cfg_path)
    sigma_aug = hydra.utils.instantiate(run_config.data.augmentations.sigma_norm)
    sigma_aug.eval()
    if hasattr(sigma_aug, "to"):
        sigma_aug = sigma_aug.to(device)
    return model, sigma_aug


def _reward_forward(reward_model, sigma_aug, volume_xyz: torch.Tensor) -> float:
    """Same forward pass as reward_geometry_explainability._reward_forward_device."""
    aug = sigma_aug(volume_xyz)
    model_input = aug.permute(2, 1, 0).contiguous().unsqueeze(0)
    with torch.no_grad():
        emb8192 = reward_model.Conv3DModule.forward_to_global_vec(
            model_input, return_global_only=True,
        )
        emb512 = reward_model.MLP(emb8192)
        scalar = reward_model.forward_to_scalar_reward_from_single_global(emb512)
    return float(scalar.reshape(()).detach().cpu().item())


def _sample_sigma_eg3d(mudc, G, conditioning_params, samples, shape, seed, device):
    z = torch.from_numpy(
        np.random.RandomState(int(seed)).randn(1, 512).astype(np.float32),
    ).to(device)
    with torch.no_grad():
        sigmas = mudc.mesh_subset_of_points_from_samples_from_z_with_grad(
            G=G,
            z=z,
            conditioning_params=conditioning_params,
            samples=samples,
            truncation_psi=TRUNC_PSI,
            truncation_cutoff=TRUNC_CUT,
            noise_mode=NOISE_MODE,
            update_emas=False,
            max_batch=MAX_BATCH,
        )
    return sigmas.squeeze(0).squeeze(-1).reshape(shape[1:4]).detach().cpu().float()


def _crop_full_cube_to_eg3d_pads(full_cube_xyz: torch.Tensor, tri_idx, full_res: int) -> torch.Tensor:
    """Crop a (full_res, full_res, full_res) world-coord-aligned σ cube to the
    same axis-aligned subregion that pads_vals_entire defines. The cropping
    indices come from `tri_idx`; both PanoHead and EG3D sample at the same
    [-box_warp/2, +box_warp/2]^3 grid at shape_res=256, so applying the same
    index slab is physically the right comparison."""
    rhs = int(tri_idx.ax1horiz.right)
    lhs = int(full_res - tri_idx.ax1horiz.left)
    bot = int(tri_idx.ax2vert.bottom)
    top = int(full_res - tri_idx.ax2vert.top)
    rear = int(tri_idx.ax3depth.rear)
    front = int(full_res - tri_idx.ax3depth.front)
    return full_cube_xyz[rhs : full_res - lhs, bot : full_res - top, rear : full_res - front].clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panohead-trunc", type=float, default=0.7,
                    help="truncation_psi used when PanoHead σ cubes were extracted")
    args = ap.parse_args()

    panohead_dir = PANOHEAD_SIGMA_ROOT / f"trunc{args.panohead_trunc:.2f}"
    out_dir = OUT_DIR_ROOT / f"panohead_trunc{args.panohead_trunc:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[reward-transfer] panohead cubes from {panohead_dir}")
    print(f"[reward-transfer] output goes to {out_dir}")
    device = torch.device("cuda")
    print(f"[reward-transfer] loading reward {REWARD_ID}")
    reward_model, sigma_aug = _load_reward(device)

    print("[reward-transfer] loading EG3D orig + tuned (for direct same-seed baselines)")
    da_orig = gen_utils.load_generator(model_path=ORIG_PKL, truncation_psi=TRUNC_PSI,
                                       truncation_cutoff=TRUNC_CUT, shape_res=SHAPE_RES)
    da_tuned = gen_utils.load_generator(model_path=TUNED_PKL, truncation_psi=TRUNC_PSI,
                                        truncation_cutoff=TRUNC_CUT, shape_res=SHAPE_RES)
    cond = torch.load(gen_utils.STATIC_CONFIGS_DIR / "single_dmap_conditioning.pt",
                      map_location=device)
    # Use pads_vals_entire.yaml — the convention the reward model expects at
    # inference (exp3 reward deltas reproduce exactly here at +12.89 mean).
    # The HANDOVER mentions front_full_rear40 as the *training* crop, but the
    # entire convention is what the published §4.3.x numbers use and what the
    # reward forward pass is effectively calibrated against (verified by
    # reproducing exp3's published per-seed deltas).
    pads = OmegaConf.load(gen_utils.STATIC_CONFIGS_DIR / "pads_vals_entire.yaml")
    mudc = finetuning_utils.MeshUtilsDataClass()
    samples_o, shape_o, tri_idx = mudc.get_samples_coordinates_from_pads_vals_dict(
        pads_vals=pads, G=da_orig.G, shape_res=SHAPE_RES,
    )
    samples_t, shape_t, _ = mudc.get_samples_coordinates_from_pads_vals_dict(
        pads_vals=pads, G=da_tuned.G, shape_res=SHAPE_RES,
    )
    print(f"[reward-transfer] cube_shape (cropped) = {tuple(shape_o[1:4])}")

    rows: List[Dict] = []
    for seed in tqdm(SEEDS, desc="scoring orig/tuned/panohead"):
        # EG3D orig
        vol_orig = _sample_sigma_eg3d(mudc, da_orig.G, cond, samples_o, shape_o, seed, device)
        r_orig = _reward_forward(reward_model, sigma_aug, vol_orig.to(device))
        # EG3D tuned
        vol_tuned = _sample_sigma_eg3d(mudc, da_tuned.G, cond, samples_t, shape_t, seed, device)
        r_tuned = _reward_forward(reward_model, sigma_aug, vol_tuned.to(device))
        # PanoHead — full 256³ cube on disk, crop to same pads_vals_entire region.
        pano_pt = panohead_dir / f"sigma_seed_{seed}.pt"
        if not pano_pt.exists():
            rows.append({"seed": seed, "reward_orig": r_orig, "reward_tuned": r_tuned,
                         "reward_panohead": float("nan"),
                         "missing_panohead": True})
            continue
        full_cube = torch.load(pano_pt, map_location="cpu").float()  # (256, 256, 256)
        cropped = _crop_full_cube_to_eg3d_pads(full_cube, tri_idx, SHAPE_RES)
        if tuple(cropped.shape) != tuple(shape_o[1:4]):
            raise RuntimeError(
                f"cropped panohead cube shape {tuple(cropped.shape)} != "
                f"EG3D cube shape {tuple(shape_o[1:4])}; aborting."
            )
        r_pano = _reward_forward(reward_model, sigma_aug, cropped.to(device))
        rows.append({"seed": seed, "reward_orig": r_orig, "reward_tuned": r_tuned,
                     "reward_panohead": r_pano, "missing_panohead": False})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_seed_rewards.csv", index=False)

    def stat(arr):
        a = np.asarray(arr, dtype=np.float64)
        a = a[~np.isnan(a)]
        return {
            "n": int(len(a)),
            "mean": float(a.mean()) if len(a) else float("nan"),
            "median": float(np.median(a)) if len(a) else float("nan"),
            "std": float(a.std()) if len(a) else float("nan"),
            "min": float(a.min()) if len(a) else float("nan"),
            "max": float(a.max()) if len(a) else float("nan"),
        }
    summary = {
        "reward_id": REWARD_ID,
        "n_seeds": int(len(df)),
        "orig":     stat(df["reward_orig"]),
        "tuned":    stat(df["reward_tuned"]),
        "panohead": stat(df["reward_panohead"]),
        "panohead_vs_orig_mean_delta":  float((df["reward_panohead"] - df["reward_orig"]).mean()),
        "panohead_vs_tuned_mean_delta": float((df["reward_panohead"] - df["reward_tuned"]).mean()),
        "panohead_frac_higher_than_orig":  float((df["reward_panohead"] > df["reward_orig"]).mean()),
        "panohead_frac_higher_than_tuned": float((df["reward_panohead"] > df["reward_tuned"]).mean()),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
