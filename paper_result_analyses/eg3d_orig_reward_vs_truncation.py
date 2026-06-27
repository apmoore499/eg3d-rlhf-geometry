"""Score the EG3D-orig (untuned) generator with the 7wnzkgie reward at a
sweep of truncation psi values, to check whether the reward is genuinely
truncation-aware. Result for the 100 exp3-matched seeds (200000-200099):

  trunc_psi=0.00 → mean +17.93  (essentially identical to EG3D-tuned at psi=0.7)
  trunc_psi=0.25 → mean +14.70
  trunc_psi=0.50 → mean  +8.70
  trunc_psi=0.70 → mean  +5.76  (canonical exp3 baseline)
  trunc_psi=1.00 → mean  +2.78

The reward is monotonically decreasing in truncation, with a span of ~15
reward units between the mean face and the full-diversity samples. This
contextualises the §4.3.6 PanoHead transfer finding: PanoHead at trunc=0.25
gives only +0.52 mean reward improvement vs trunc=0.7 (vs +8.94 for EG3D),
so the reward cannot 'see' PanoHead's truncation knob through the
representation mismatch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_result_analyses.path_defaults import (  # noqa: E402
    EG3D_ROOT,
    RLHF_CORE_ROOT,
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

REWARD_ID = "7wnzkgie"
ORIG_PKL = REPO_ROOT / "pkl_pt" / "eg3d_1" / "ffhq512-128.pkl"
SEEDS = list(range(200000, 200100))
SHAPE_RES = 256
TRUNC_PSIS = (0.0, 0.25, 0.5, 0.7, 1.0)
OUT_PATH = (
    reward_embedding_analysis_dir()
    / "panohead_reward_transfer"
    / "eg3d_orig_reward_vs_truncation.json"
)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = reward_loading.load_rwd_model_from_cfg(REWARD_ID).to(device).eval()
    sigma_aug = hydra.utils.instantiate(
        OmegaConf.load(
            RLHF_SRC_ROOT / "RWD_MODELS_FOR_TUNING" / REWARD_ID / "run_config.yaml"
        ).data.augmentations.sigma_norm
    ).eval()
    if hasattr(sigma_aug, "to"):
        sigma_aug = sigma_aug.to(device)
    da = gen_utils.load_generator(
        model_path=ORIG_PKL, truncation_psi=1.0, truncation_cutoff=14, shape_res=SHAPE_RES,
    )
    cond = torch.load(
        gen_utils.STATIC_CONFIGS_DIR / "single_dmap_conditioning.pt",
        map_location=device,
    )
    pads = OmegaConf.load(gen_utils.STATIC_CONFIGS_DIR / "pads_vals_entire.yaml")
    mudc = finetuning_utils.MeshUtilsDataClass()
    samples, shape, _ = mudc.get_samples_coordinates_from_pads_vals_dict(
        pads_vals=pads, G=da.G, shape_res=SHAPE_RES,
    )

    def score_at_psi(psi: float) -> np.ndarray:
        rewards = []
        for seed in SEEDS:
            z = torch.from_numpy(
                np.random.RandomState(int(seed)).randn(1, 512).astype(np.float32)
            ).to(device)
            with torch.no_grad():
                sigmas = mudc.mesh_subset_of_points_from_samples_from_z_with_grad(
                    G=da.G, z=z, conditioning_params=cond, samples=samples,
                    truncation_psi=psi, truncation_cutoff=14, noise_mode="const",
                    update_emas=False, max_batch=1_000_000,
                )
            vol = sigmas.squeeze(0).squeeze(-1).reshape(shape[1:4]).detach().float().to(device)
            aug = sigma_aug(vol)
            x = aug.permute(2, 1, 0).contiguous().unsqueeze(0)
            with torch.no_grad():
                emb = model.Conv3DModule.forward_to_global_vec(x, return_global_only=True)
                emb = model.MLP(emb)
                r = model.forward_to_scalar_reward_from_single_global(emb)
            rewards.append(float(r.reshape(()).cpu()))
        return np.asarray(rewards, dtype=np.float64)

    payload = {}
    for psi in TRUNC_PSIS:
        arr = score_at_psi(psi)
        s = {
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "per_seed": [
                {"seed": int(SEEDS[i]), "reward": float(arr[i])} for i in range(len(arr))
            ],
        }
        payload[f"trunc_psi={psi:.2f}"] = s
        print(f"trunc_psi={psi:.2f}: mean={s['mean']:+.3f}, median={s['median']:+.3f}, "
              f"std={s['std']:.3f}, range=[{s['min']:+.3f}, {s['max']:+.3f}]")
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
