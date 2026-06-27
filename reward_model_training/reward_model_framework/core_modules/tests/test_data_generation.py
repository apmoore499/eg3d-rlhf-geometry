"""
Regenerate a small set of seeds for core data types (triple_rgb, triple_dmap, sigma_field_256)
and produce mesh visualisations. Intended as a quick end-to-end sanity run.
"""

import argparse
from pathlib import Path
from typing import List
import sys

# # Ensure repo root (contains autoroot.py) is on path for local imports
# REPO_ROOT = Path(__file__).resolve().parents[2]
# if str(REPO_ROOT) not in sys.path:
#     sys.path.append(str(REPO_ROOT))

# import autoroot  # noqa: F401

import torch

from core_modules.data.create_train_data import generation_utils as gen_utils
from core_modules.data.create_train_data import synthesize_triple_rgb
from core_modules.data.create_train_data import synthesize_triple_dmap
from core_modules.data.create_train_data import synthesize_sigma_field_256_combined as synth_sigma
from core_modules.data.create_train_data import visualise_mesh
from core_modules.data.create_train_data import synthesize_landmarks as synth_lmks
from core_modules.utils import finetuning_utils


def _parse_seeds(spec: str) -> List[int]:
    seeds: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(list(range(int(a), int(b) + 1)))
        elif part:
            seeds.append(int(part))
    return seeds


def regenerate_triple_rgb(seeds, truncation_psi: float):
    da = gen_utils.load_generator(truncation_psi=truncation_psi)
    save_dir = Path(gen_utils.DEFAULT_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        print(f"[triple_rgb] generating seed {seed}")
        triple_rgb = synthesize_triple_rgb.synthesize_triple_rgb(da, seed)
        images = synthesize_triple_rgb.convert_stylegan_to_rgb_images(triple_rgb)
        for idx, img in enumerate(images):
            img_file = save_dir / f"triple_rgb_s_{seed}_{idx}.jpg"
            img.save(img_file, quality=95)
            print(f"  saved {img_file}")
        torch.save(triple_rgb, save_dir / f"triple_rgb_s_{seed}_tensor.pt")
        print(f"  saved {save_dir / f'triple_rgb_s_{seed}_tensor.pt'}")


def regenerate_triple_dmap(seeds, truncation_psi: float):
    da = gen_utils.load_generator(truncation_psi=truncation_psi)
    save_dir = Path(gen_utils.DEFAULT_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        print(f"[triple_dmap] generating seed {seed}")
        triple_dmap = synthesize_triple_dmap.synthesize_triple_dmap(da, seed)
        out_fn = save_dir / f"triple_dmap_s_{seed}.pt"
        torch.save(triple_dmap, out_fn)
        print(f"  saved {out_fn}")


def regenerate_sigma_fields(seeds, truncation_psi: float):
    da, cond_pms = synth_sigma.load_model_and_config(truncation_psi=truncation_psi)
    pads_vals_path = gen_utils.STATIC_CONFIGS_DIR / "pads_vals_entire.yaml"
    pads_vals = synth_sigma.omegaconf.OmegaConf.load(pads_vals_path)
    mudc = finetuning_utils.MeshUtilsDataClass()
    coordinates, shape, _ = mudc.get_samples_coordinates_from_pads_vals_dict(pads_vals=pads_vals, G=da.G, shape_res=pads_vals.shape_res)
    out_dir = Path(gen_utils.DEFAULT_SAVE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    for s in seeds:
        print(f"[sigma_field_256] generating seed {s}")
        z = torch.from_numpy(synth_sigma.np.random.RandomState(s).randn(1, 512)).to(device)
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
        out_fn = out_dir / f"entire_sigma_field_256_s_{s}.pt"
        torch.save(sigmas, out_fn)
        print(f"  saved {out_fn}")


def regenerate_landmarks(seeds, views=None, overwrite: bool = False):
    views = views or [0, 1, 2]
    out_dir = Path(gen_utils.DEFAULT_SAVE_DIR)
    for seed in seeds:
        print(f"[landmarks] generating seed {seed}")
        synth_lmks.process_seed(seed, views, rgb_dir=out_dir, out_dir=out_dir)
        for v in views:
            for base in ["triple_rgb_lmks_98", "triple_rgb_lmks_98_3d"]:
                fn = out_dir / f"{base}_s_{seed}_{v}.pt"
                print(f"  expected {fn}")


def regenerate_visualisations(seeds, overwrite: bool = True):
    visualise_mesh.generate_mesh_visuals(
        sigma_dir=Path(gen_utils.DEFAULT_SAVE_DIR),
        save_dir=Path(gen_utils.DEFAULT_SAVE_DIR) / "visualisations",
        seeds=seeds,
        overwrite=overwrite,
        window_size=1200,
    )


def main():
    parser = argparse.ArgumentParser(description="Regenerate a small set of seeds for core datatypes.")
    parser.add_argument("--seeds", type=str, default="0-9", help="Seed spec, e.g., '0-9' or '100000-100009'")
    parser.add_argument("--truncation_psi", type=float, default=1.0, help="Truncation psi for generator")
    parser.add_argument("--overwrite_landmarks", action="store_true", help="(Ignored; landmarks always regenerated)")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    print(f"[RUN] seeds={seeds}")

    regenerate_triple_rgb(seeds, truncation_psi=args.truncation_psi)
    regenerate_triple_dmap(seeds, truncation_psi=args.truncation_psi)
    regenerate_sigma_fields(seeds, truncation_psi=args.truncation_psi)
    regenerate_landmarks(seeds, overwrite=args.overwrite_landmarks)
    regenerate_visualisations(seeds, overwrite=True)
    print("[DONE] regeneration complete")


if __name__ == "__main__":
    main()
