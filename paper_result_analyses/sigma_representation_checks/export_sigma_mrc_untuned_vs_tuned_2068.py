"""Export 512^3 sigma fields as MRC files for untuned (baseline) and tuned (01446/002068) EG3D generators.

For each requested seed, writes two .mrc files (~537 MB each at float32, mrc_mode=2):
  seed{SEED}_untuned.mrc        (orig ffhq pkl)
  seed{SEED}_tuned_2068.mrc     (01446 final tuned ckpt)

Default seeds: [200050]. Pass --seeds 200050,200057,... for more.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_result_analyses.path_defaults import reported_run_dir  # noqa: E402

EG3D_ROOT = REPO_ROOT / "eg3d"
for _p in (THIS_DIR, REPO_ROOT, EG3D_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import mrcfile
import numpy as np
import torch

import dnnlib
from eg3d import legacy

ORIG_PKL = str(REPO_ROOT / "pkl_pt" / "eg3d_1" / "ffhq512-128.pkl")
RUN_DIR = reported_run_dir()
OUT_DIR = RUN_DIR / "reward_embedding_analysis" / "mrc_extracted_sigma_fields_untuned_vs_tuned_2068"
STATIC_CONFIGS_DIR = REPO_ROOT / "reward_model_training" / "static_configs"

DEVICE = torch.device("cuda")
SHAPE_RES = 512
TRUNCATION_PSI = 0.7
TRUNCATION_CUTOFF = 14
NOISE_MODE = "const"
MAX_BATCH = 262_144


def load_generator(pkl_path: str):
    orig_torch_load = torch.load
    def _load(*args, **kwargs):
        kwargs.setdefault("map_location", DEVICE)
        return orig_torch_load(*args, **kwargs)
    torch.load = _load
    try:
        with dnnlib.util.open_url(pkl_path) as f:
            G = legacy.load_network_pkl(f)["G_ema"].to(DEVICE).eval()
    finally:
        torch.load = orig_torch_load
    return G


def _create_samples(N: int, cube_length: float) -> torch.Tensor:
    voxel_origin = np.array([0, 0, 0]) - cube_length / 2
    voxel_size = cube_length / (N - 1)
    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 3)
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    return samples.unsqueeze(0)


def sample_sigma_cube(G, seed: int, conditioning_params: torch.Tensor) -> np.ndarray:
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, 512)).to(DEVICE)
    with torch.no_grad():
        ws = G.mapping(z, conditioning_params, truncation_psi=TRUNCATION_PSI,
                       truncation_cutoff=TRUNCATION_CUTOFF)
    cube_length = float(G.rendering_kwargs["box_warp"])
    samples = _create_samples(N=SHAPE_RES, cube_length=cube_length).to(DEVICE)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=DEVICE)
    dirs = torch.zeros((samples.shape[0], MAX_BATCH, 3), device=DEVICE)
    dirs[..., -1] = -1
    head = 0
    with torch.no_grad():
        while head < samples.shape[1]:
            batch = samples[:, head:head + MAX_BATCH]
            sig = G.sample_mixed(
                coordinates=batch,
                directions=dirs[:, :batch.shape[1]],
                ws=ws,
                truncation_psi=TRUNCATION_PSI,
                truncation_cutoff=TRUNCATION_CUTOFF,
                noise_mode=NOISE_MODE,
            )["sigma"]
            sigmas[:, head:head + batch.shape[1]] = sig
            head += batch.shape[1]
    cube = sigmas.reshape((SHAPE_RES, SHAPE_RES, SHAPE_RES)).detach().cpu().numpy()
    cube = np.flip(cube, 0)
    return np.ascontiguousarray(cube.astype(np.float32))


def write_mrc(path: Path, cube: np.ndarray):
    print(f"[mrc] writing {path}  shape={cube.shape}  dtype={cube.dtype}", flush=True)
    with mrcfile.new_mmap(str(path), overwrite=True, shape=cube.shape, mrc_mode=2) as mrc:
        mrc.data[:] = cube
    print(f"[mrc] done {path}  size_MB={path.stat().st_size / 1e6:.1f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="200050",
                        help="comma-separated seed ints (default 200050)")
    parser.add_argument("--tuned-kimgs", default="2068",
                        help="comma-separated kimg ints; each maps to "
                             "network-snapshot-{kimg:06d}.pkl in 01446 run dir")
    parser.add_argument("--skip-untuned", action="store_true",
                        help="don't sample/write the untuned MRC (use when "
                             "seed{SEED}_untuned.mrc already exists)")
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    kimgs = [int(k.strip()) for k in args.tuned_kimgs.split(",") if k.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cond = torch.load(STATIC_CONFIGS_DIR / "single_dmap_conditioning.pt",
                      map_location=DEVICE)

    if not args.skip_untuned:
        print(f"[mrc] loading untuned: {ORIG_PKL}", flush=True)
        G_untuned = load_generator(ORIG_PKL)
        for seed in seeds:
            out_u = OUT_DIR / f"seed{seed}_untuned.mrc"
            if out_u.exists():
                print(f"[mrc] skip {out_u.name} (already exists)", flush=True)
                continue
            print(f"[mrc] sampling untuned cube for seed {seed}", flush=True)
            cube_u = sample_sigma_cube(G_untuned, seed, cond)
            write_mrc(out_u, cube_u)
            del cube_u
        del G_untuned
        torch.cuda.empty_cache()

    for kimg in kimgs:
        tuned_pkl = str(RUN_DIR / f"network-snapshot-{kimg:06d}.pkl")
        print(f"\n[mrc] loading tuned ckpt {kimg:06d}: {tuned_pkl}", flush=True)
        G_tuned = load_generator(tuned_pkl)
        for seed in seeds:
            out_t = OUT_DIR / f"seed{seed}_tuned_{kimg:04d}.mrc"
            if out_t.exists():
                print(f"[mrc] skip {out_t.name} (already exists)", flush=True)
                continue
            print(f"[mrc] sampling tuned cube ckpt={kimg:06d} seed {seed}",
                  flush=True)
            cube_t = sample_sigma_cube(G_tuned, seed, cond)
            write_mrc(out_t, cube_t)
            del cube_t
        del G_tuned
        torch.cuda.empty_cache()

    print(f"\n[mrc] all done. outputs in: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
