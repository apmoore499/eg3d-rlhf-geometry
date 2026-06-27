import argparse
import os
from pathlib import Path
import sys

import autoroot  # noqa: F401
import mrcfile
import numpy as np
import omegaconf
import torch
from tqdm import tqdm

from core_modules.utils import finetuning_utils
from core_modules.data.create_train_data import generation_utils as gen_utils

SAVE_DIR = gen_utils.DEFAULT_SAVE_DIR
DEFAULT_OUTPUT_DIR = SAVE_DIR / "frontslab_sigma_field_256_ffhq512-128_const_noise_t1"

# Configuration paths
STATIC_CONFIGS_DIR = gen_utils.STATIC_CONFIGS_DIR
NETWORK_PKL_PATH = gen_utils.DEFAULT_MODEL_PATH
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT_DIR = Path(__file__).parent

DATA_PREFIX = "entire_sigma_field_256_s_"

# High quality generation parameters
HIQ_SEED_RANGE = (100000, 101000)
HIQ_TRUNCATION_PSI = 0.25

# Ranked seeds parameters
RANKED_TRUNCATION_PSI = 1.0
RANKED_SEEDS_FILE = "rankedseedsall.csv"


def export_sample_mrc(sigmas, output_dir):
    """Export sigma field to MRC format for visualization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nfiles = len(list(output_dir.glob("*.mrc")))
    fn_export = output_dir / f"testing{nfiles + 1}.mrc"

    with mrcfile.new_mmap(fn_export, overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
        mrc.data[:] = sigmas.detach().cpu()


def load_model_and_config(truncation_psi):
    """Initialize model and configuration."""
    da = gen_utils.load_generator(NETWORK_PKL_PATH, truncation_psi=truncation_psi, shape_res=512)

    cond_pms_path = STATIC_CONFIGS_DIR / "single_dmap_conditioning.pt"
    cond_pms = torch.load(cond_pms_path, map_location=da.device)

    return da, cond_pms


def load_ranked_seeds(base_dir):
    """Load ranked seeds from the canonical combined CSV."""
    return gen_utils.load_ranked_seeds(Path(base_dir) / RANKED_SEEDS_FILE)


def get_existing_seeds(save_dir, prefix):
    """Get list of seeds that have already been processed."""
    return gen_utils.get_existing_seeds(save_dir, prefix)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate sigma_field_256 reward-model volumes.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "E3D_RLHF_SIGMA_DATA_DIR",
                str(DEFAULT_OUTPUT_DIR),
            )
        ),
    )
    parser.add_argument(
        "--pads-config",
        type=Path,
        default=Path(
            os.environ.get(
                "E3D_RLHF_SIGMA_PADS_CONFIG",
                str(STATIC_CONFIGS_DIR / "pads_vals_front_full_rear40.yaml"),
            )
        ),
    )
    parser.add_argument("--data-prefix", type=str, default=DATA_PREFIX)
    return parser.parse_args()


def generate_sigma_fields_for_seeds(seeds, truncation_psi, description, output_dir, pads_vals_path, data_prefix):
    """Generate sigma fields for a list of seeds with specified truncation parameters."""
    device = torch.device("cuda")

    # Load model and configuration
    da, cond_pms = load_model_and_config(truncation_psi)

    # Load pads configuration
    pads_vals = omegaconf.OmegaConf.load(pads_vals_path)

    # Initialize mesh utilities
    mudc = finetuning_utils.MeshUtilsDataClass()
    coordinates, shape, coords = mudc.get_samples_coordinates_from_pads_vals_dict(pads_vals=pads_vals, G=da.G, shape_res=pads_vals.shape_res)

    # Generate sigma fields for each seed
    output_dir.mkdir(parents=True, exist_ok=True)

    for s in tqdm(seeds, desc=description):
        z = torch.from_numpy(np.random.RandomState(s).randn(1, 512)).to(device)

        with torch.no_grad():
            sigmas = mudc.mesh_subset_of_points_from_samples_from_z_with_grad(da.G, z=z.view(1, 512), conditioning_params=cond_pms, samples=coordinates, truncation_psi=da.truncation_psi, truncation_cutoff=da.truncation_cutoff, update_emas=False, noise_mode="const")

        sigmas = sigmas.squeeze(-1, 0).reshape(shape[1:4]).half()

        output_path = output_dir / f"{data_prefix}{s}.pt"
        torch.save(sigmas, output_path)


def generate_sigma_fields(args):
    """Generate sigma fields for both high-quality samples and ranked seeds."""
    save_dir = Path(args.save_dir)
    pads_vals_path = Path(args.pads_config)
    data_prefix = args.data_prefix

    # Get existing seeds to skip
    existing_seeds = get_existing_seeds(save_dir, data_prefix)
    print(f"Found {len(existing_seeds)} existing seeds, will skip these")
    print(f"Saving to: {save_dir}")
    print(f"Using pads config: {pads_vals_path}")

    # 1. Generate high-quality samples (seeds 100000-101000, truncation_psi=0.25)
    print(f"\n=== Generating high-quality samples ===")
    print(f"Seed range: {HIQ_SEED_RANGE}")
    print(f"Truncation PSI: {HIQ_TRUNCATION_PSI}")

    hiq_seeds = list(range(*HIQ_SEED_RANGE))
    hiq_remaining = [s for s in hiq_seeds if s not in existing_seeds]
    print(f"High-quality seeds to process: {len(hiq_remaining)}/{len(hiq_seeds)}")

    if hiq_remaining:
        generate_sigma_fields_for_seeds(hiq_remaining, HIQ_TRUNCATION_PSI, "Generating high-quality sigma fields", save_dir, pads_vals_path, data_prefix)

    # 2. Generate ranked seeds from CSV files (truncation_psi=1.0)
    print(f"\n=== Generating ranked seeds ===")
    print(f"Loading seeds from: {SCRIPT_DIR / RANKED_SEEDS_FILE}")
    print(f"Truncation PSI: {RANKED_TRUNCATION_PSI}")

    ranked_seeds = load_ranked_seeds(SCRIPT_DIR)
    ranked_remaining = [s for s in ranked_seeds if s not in existing_seeds]
    print(f"Ranked seeds to process: {len(ranked_remaining)}/{len(ranked_seeds)}")

    if ranked_remaining:
        generate_sigma_fields_for_seeds(ranked_remaining, RANKED_TRUNCATION_PSI, "Generating ranked sigma fields", save_dir, pads_vals_path, data_prefix)

    print("\n=== Generation complete ===")


if __name__ == "__main__":
    generate_sigma_fields(parse_args())
