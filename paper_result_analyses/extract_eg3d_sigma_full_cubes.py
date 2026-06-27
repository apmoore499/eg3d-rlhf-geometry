"""Extract FULL 256³ σ-cubes (uncropped) from EG3D-orig and EG3D-tuned,
matching the convention used by the 360° generators
(PanoHead/HyPlaneHead/SphereHead). Saves one sigma_seed_{seed}.pt per
seed under aw98_template_workdir/eg3d_{orig,tuned}_sigma_full256/.

Used downstream by visualise_mesh_tails_generic.py to render mesh tails.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_result_analyses.path_defaults import (  # noqa: E402
    EG3D_ROOT,
    RLHF_CORE_ROOT,
    reported_run_dir,
    reward_embedding_analysis_dir,
)

for _p in (REPO_ROOT, EG3D_ROOT, RLHF_CORE_ROOT.parent):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)
try:
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
except Exception:
    pass
from core_modules.data.create_train_data import generation_utils as gen_utils  # noqa: E402

PSI = 0.7
PSI_CUT = 14
SHAPE_RES = 256
SEEDS = list(range(200000, 200100))
PADS_CROP = (slice(64, 192), slice(64, 205), slice(102, 231))


def create_samples(N: int, voxel_origin=(0, 0, 0), cube_length: float = 1.0):
    voxel_origin = np.array(voxel_origin) - cube_length / 2
    voxel_size = cube_length / (N - 1)
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    return samples.unsqueeze(0), voxel_origin, voxel_size


ORIG_PKL = REPO_ROOT / "pkl_pt" / "eg3d_1" / "ffhq512-128.pkl"
TUNED_PKL = reported_run_dir() / "network-snapshot-002068_LAST.pkl"
WORKDIR = reward_embedding_analysis_dir() / "aw98_template_workdir"


def extract(pkl: Path, out_dir: Path, seeds, crop_f16: bool,
            max_batch: int = 1_000_000):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eg3d-sigma] loading {pkl}")
    da = gen_utils.load_generator(
        model_path=pkl, truncation_psi=PSI, truncation_cutoff=PSI_CUT,
        shape_res=SHAPE_RES,
    )
    G = da.G
    device = next(G.parameters()).device
    tdca = gen_utils.get_single_dmap_cam(da)
    box_warp = float(G.rendering_kwargs.get("box_warp", 1.0))
    samples, _, _ = create_samples(N=SHAPE_RES, voxel_origin=[0, 0, 0],
                                    cube_length=box_warp * 1.0)
    samples = samples.to(device)
    n_voxels = samples.shape[1]
    rays = torch.zeros((samples.shape[0], max_batch, 3), device=device)
    rays[..., -1] = -1.0
    for seed in tqdm(seeds, desc=f"σ extract {out_dir.name}"):
        out_pt = out_dir / f"sigma_seed_{seed}.pt"
        if out_pt.exists():
            continue
        z = torch.from_numpy(
            np.random.RandomState(int(seed)).randn(1, G.z_dim).astype(np.float32)
        ).to(device)
        sigmas = torch.zeros((1, n_voxels, 1), device=device, dtype=torch.float32)
        head = 0
        torch.manual_seed(0)
        while head < n_voxels:
            end = min(head + max_batch, n_voxels)
            out = G.sample(
                samples[:, head:end], rays[:, : end - head],
                z, tdca.conditioning_params,
                truncation_psi=PSI, truncation_cutoff=PSI_CUT,
                noise_mode="const",
            )["sigma"]
            sigmas[:, head:end] = out
            head = end
        cube = sigmas.reshape(SHAPE_RES, SHAPE_RES, SHAPE_RES).cpu()
        if crop_f16:
            cube = cube[PADS_CROP[0], PADS_CROP[1], PADS_CROP[2]].to(torch.float16)
        torch.save(cube, out_pt)


def parse_seeds(spec: str):
    out = []
    for chunk in spec.split(","):
        if "-" in chunk:
            a, b = chunk.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["orig", "tuned", "both"], default="both")
    ap.add_argument("--seeds", default=None,
                    help="comma/range seed spec (e.g. 200000-200999); "
                         "default uses the hard-coded 100-seed bank")
    ap.add_argument("--crop-f16", action="store_true", default=False,
                    help="save σ pre-cropped to pads_vals_entire (128x141x129) "
                         "as float16 — matches the comparison-model pipeline")
    ap.add_argument("--outdir-suffix", default=None,
                    help="override out dir name (default eg3d_*_sigma_full256)")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds) if args.seeds else SEEDS
    suffix = args.outdir_suffix or "sigma_full256"
    if args.which in ("orig", "both"):
        extract(ORIG_PKL, WORKDIR / f"eg3d_orig_{suffix}", seeds,
                args.crop_f16)
    if args.which in ("tuned", "both"):
        extract(TUNED_PKL, WORKDIR / f"eg3d_tuned_{suffix}", seeds,
                args.crop_f16)


if __name__ == "__main__":
    main()
