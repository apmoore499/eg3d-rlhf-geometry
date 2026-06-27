"""Script to synthesize triple depth map data from EG3D generator."""

from pathlib import Path

import autoroot  # noqa: F401
import numpy as np
import torch
from tqdm import tqdm

from core_modules.data.create_train_data import generation_utils as gen_utils


# Configuration
DEFAULT_MODEL_PATH = gen_utils.DEFAULT_MODEL_PATH
SAVE_DIR = gen_utils.DEFAULT_SAVE_DIR
DATA_PREFIX = "triple_dmap_s_"
NEURAL_RESOLUTION = 128

# High quality generation parameters
HIQ_SEED_RANGE = (100000, 101000)
HIQ_TRUNCATION_PSI = 0.25

# Ranked seeds parameters
RANKED_SEEDS_FILE = "rankedseedsall.csv"
RANKED_TRUNCATION_PSI = 1.0


def load_generator(model_path=DEFAULT_MODEL_PATH, truncation_psi=1.0):
    """Load the EG3D generator model with specified truncation."""
    return gen_utils.load_generator(model_path, truncation_psi=truncation_psi)


def synthesize_triple_dmap(da, seed):
    """Synthesize a triple depth map for a given seed."""
    G = da.G
    device = torch.device("cuda")

    # Get camera parameters
    tdca = gen_utils.get_triple_dmap_cams(da)

    # Generate latent code
    z = torch.from_numpy(np.random.RandomState(seed).randn(3, G.z_dim)).to(device)
    z.requires_grad = False

    # Generate image
    ws = G.mapping(z, tdca.conditioning_params, truncation_psi=da.truncation_psi, truncation_cutoff=da.truncation_cutoff)
    img = G.synthesis(ws, tdca.camera_params, neural_rendering_resolution=NEURAL_RESOLUTION, noise_mode="const")

    # breakpoint()

    return img["image_depth"]


def load_ranked_seeds(base_dir):
    """Load ranked seeds from rankedseedsall.csv."""
    return gen_utils.load_ranked_seeds(Path(base_dir) / RANKED_SEEDS_FILE)


def get_existing_seeds(save_dir, prefix):
    """Get list of seeds that have already been processed."""
    return gen_utils.get_existing_seeds(save_dir, prefix)


def generate_triple_dmaps_for_seeds(seeds, truncation_psi, description):
    """Generate triple depth maps for a list of seeds with specified truncation parameters."""
    # Load generator with specified truncation
    print(f"Loading generator with truncation_psi={truncation_psi}...")
    da = load_generator(truncation_psi=truncation_psi)

    # Generate triple depth maps for each seed
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for seed in tqdm(seeds, desc=description):
        triple_dmap = synthesize_triple_dmap(da, seed)
        output_file = SAVE_DIR / f"{DATA_PREFIX}{seed}.pt"
        torch.save(triple_dmap, output_file)


def main():
    """Main execution function - generates both high-quality and ranked seeds."""
    # Setup
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).parent

    # Get existing seeds to skip
    existing_seeds = get_existing_seeds(SAVE_DIR, DATA_PREFIX)
    print(f"Found {len(existing_seeds)} existing seeds, will skip these")

    # 1. Generate high-quality samples (seeds 100000-101000, truncation_psi=0.25)
    print(f"\n=== Generating high-quality samples ===")
    print(f"Seed range: {HIQ_SEED_RANGE}")
    print(f"Truncation PSI: {HIQ_TRUNCATION_PSI}")

    hiq_seeds = list(range(*HIQ_SEED_RANGE))
    hiq_remaining = [s for s in hiq_seeds if s not in existing_seeds]
    print(f"High-quality seeds to process: {len(hiq_remaining)}/{len(hiq_seeds)}")

    if hiq_remaining:
        generate_triple_dmaps_for_seeds(hiq_remaining, HIQ_TRUNCATION_PSI, "Generating high-quality triple dmaps")

    # 2. Generate ranked seeds from CSV file (truncation_psi=1.0)
    print(f"\n=== Generating ranked seeds ===")
    print(f"Loading seeds from: {script_dir / RANKED_SEEDS_FILE}")
    print(f"Truncation PSI: {RANKED_TRUNCATION_PSI}")

    ranked_seeds = load_ranked_seeds(script_dir)
    ranked_remaining = [s for s in ranked_seeds if s not in existing_seeds]
    print(f"Ranked seeds to process: {len(ranked_remaining)}/{len(ranked_seeds)}")

    if ranked_remaining:
        generate_triple_dmaps_for_seeds(ranked_remaining, RANKED_TRUNCATION_PSI, "Generating ranked triple dmaps")

    print("\n=== Generation complete ===")


if __name__ == "__main__":
    main()
