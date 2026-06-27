"""Synthesise 128^3 whole-scene sigma fields for reward-model training.

Sibling of synthesize_sigma_field_256_combined.py:
  - Uses MeshUtilsDataClass.get_samples_coordinates_entire_no_pads(shape_res=128)
    so the cube covers the full box_warp region (no pads-based cropping).
  - Saves as `entire_sigma_field_128_s_{seed}.pt` (float16) into save dir.

Same seed coverage as the 256 script:
  - HIQ seeds 100000-100999 at truncation_psi=0.25
  - Ranked seeds from rankedseedsall.csv at truncation_psi=1.0
"""

import argparse
import os
from pathlib import Path
import sys

import autoroot  # noqa: F401
import numpy as np
import torch
from tqdm import tqdm

from core_modules.utils import finetuning_utils
from core_modules.data.create_train_data import generation_utils as gen_utils

SAVE_DIR = gen_utils.DEFAULT_SAVE_DIR
STATIC_CONFIGS_DIR = gen_utils.STATIC_CONFIGS_DIR
NETWORK_PKL_PATH = gen_utils.DEFAULT_MODEL_PATH
DEFAULT_OUTPUT_DIR = SAVE_DIR / "entire_sigma_field_128"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT_DIR = Path(__file__).parent

DATA_PREFIX = "entire_sigma_field_128_s_"
SHAPE_RES = 128

# High quality generation parameters
HIQ_SEED_RANGE = (100000, 101000)
HIQ_TRUNCATION_PSI = 0.25

# Ranked seeds parameters
RANKED_TRUNCATION_PSI = 1.0
RANKED_SEEDS_FILE = "rankedseedsall.csv"


def load_model_and_config(truncation_psi):
    da = gen_utils.load_generator(NETWORK_PKL_PATH, truncation_psi=truncation_psi, shape_res=512)
    cond_pms_path = STATIC_CONFIGS_DIR / "single_dmap_conditioning.pt"
    cond_pms = torch.load(cond_pms_path, map_location=da.device)
    return da, cond_pms


def load_ranked_seeds(base_dir):
    return gen_utils.load_ranked_seeds(Path(base_dir) / RANKED_SEEDS_FILE)


def get_existing_seeds(save_dir, prefix):
    return gen_utils.get_existing_seeds(save_dir, prefix)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 128^3 whole-scene sigma fields for reward training.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "E3D_RLHF_SIGMA_DATA_DIR_128",
                str(DEFAULT_OUTPUT_DIR),
            )
        ),
    )
    parser.add_argument("--data-prefix", type=str, default=DATA_PREFIX)
    parser.add_argument("--shape-res", type=int, default=SHAPE_RES)
    return parser.parse_args()


def generate_sigma_fields_for_seeds(seeds, truncation_psi, description, output_dir, data_prefix, shape_res):
    device = torch.device("cuda")
    da, cond_pms = load_model_and_config(truncation_psi)
    mudc = finetuning_utils.MeshUtilsDataClass()
    coordinates, shape, _ = mudc.get_samples_coordinates_entire_no_pads(shape_res=shape_res, G=da.G)

    output_dir.mkdir(parents=True, exist_ok=True)

    for s in tqdm(seeds, desc=description):
        z = torch.from_numpy(np.random.RandomState(s).randn(1, 512)).to(device)

        with torch.no_grad():
            sigmas = mudc.mesh_subset_of_points_from_samples_from_z_with_grad(
                da.G,
                z=z.view(1, 512),
                conditioning_params=cond_pms,
                samples=coordinates,
                truncation_psi=da.truncation_psi,
                truncation_cutoff=da.truncation_cutoff,
                update_emas=False,
                noise_mode="const",
            )

        sigmas = sigmas.squeeze(-1, 0).reshape(shape[1:4]).half()

        output_path = output_dir / f"{data_prefix}{s}.pt"
        torch.save(sigmas, output_path)


def generate_sigma_fields(args):
    save_dir = Path(args.save_dir)
    data_prefix = args.data_prefix
    shape_res = int(args.shape_res)

    existing_seeds = get_existing_seeds(save_dir, data_prefix)
    print(f"Found {len(existing_seeds)} existing seeds, will skip these")
    print(f"Saving to: {save_dir}")
    print(f"Shape res: {shape_res}^3 (no pads, entire scene)")
    print(f"Data prefix: {data_prefix}")

    # 1. HIQ seeds (100000-100999, truncation_psi=0.25)
    print("\n=== Generating high-quality samples ===")
    print(f"Seed range: {HIQ_SEED_RANGE}, truncation PSI: {HIQ_TRUNCATION_PSI}")
    hiq_seeds = list(range(*HIQ_SEED_RANGE))
    hiq_remaining = [s for s in hiq_seeds if s not in existing_seeds]
    print(f"HIQ seeds remaining: {len(hiq_remaining)}/{len(hiq_seeds)}")
    if hiq_remaining:
        generate_sigma_fields_for_seeds(
            hiq_remaining,
            HIQ_TRUNCATION_PSI,
            "HIQ sigma fields",
            save_dir,
            data_prefix,
            shape_res,
        )

    # 2. Ranked seeds (truncation_psi=1.0)
    print("\n=== Generating ranked seeds ===")
    print(f"Loading seeds from: {SCRIPT_DIR / RANKED_SEEDS_FILE}, truncation PSI: {RANKED_TRUNCATION_PSI}")
    ranked_seeds = load_ranked_seeds(SCRIPT_DIR)
    ranked_remaining = [s for s in ranked_seeds if s not in existing_seeds]
    print(f"Ranked seeds remaining: {len(ranked_remaining)}/{len(ranked_seeds)}")
    if ranked_remaining:
        generate_sigma_fields_for_seeds(
            ranked_remaining,
            RANKED_TRUNCATION_PSI,
            "Ranked sigma fields",
            save_dir,
            data_prefix,
            shape_res,
        )

    print("\n=== Generation complete ===")


if __name__ == "__main__":
    generate_sigma_fields(parse_args())
