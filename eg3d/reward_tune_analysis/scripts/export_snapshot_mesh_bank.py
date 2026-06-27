import autoroot  # noqa: F401

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from core_modules.data.create_train_data import generation_utils
from core_modules.utils import finetuning_utils


@dataclass(frozen=True)
class MethodSpec:
    name: str
    level: float
    use_solidify: bool


METHODS_BY_NAME = {
    "legacy_sigma10": MethodSpec(name="legacy_sigma10", level=10.0, use_solidify=False),
    "cummax": MethodSpec(name="cummax", level=30.0, use_solidify=True),
}
DEFAULT_METHODS = ("legacy_sigma10",)


def parse_args():
    parser = argparse.ArgumentParser(description="Export before/after mesh image banks for EG3D snapshots.")
    parser.add_argument("--tuned-pkl", required=True, help="Final tuned network snapshot (.pkl).")
    parser.add_argument("--baseline-pkl", required=True, help="Untuned baseline network snapshot (.pkl).")
    parser.add_argument("--outdir", required=True, help="Output directory for the image bank.")
    parser.add_argument("--start-seed", type=int, default=9100000, help="First seed in the export bank.")
    parser.add_argument("--num-seeds", type=int, default=40, help="Number of consecutive seeds to export.")
    parser.add_argument("--shape-res", type=int, default=256, help="Sigma sampling resolution.")
    parser.add_argument("--truncation-psi", type=float, default=0.7)
    parser.add_argument("--truncation-cutoff", type=int, default=14)
    parser.add_argument("--win-size", type=int, default=4096)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHODS_BY_NAME.keys()),
        default=list(DEFAULT_METHODS),
        help=(
            "Surface extraction methods to export. The public paper-facing path uses "
            "legacy_sigma10 only; cummax is available for exploratory comparison."
        ),
    )
    return parser.parse_args()


def load_generator(network_pkl: str):
    da = generation_utils.set_defaults(generation_utils.DArgs())
    da.set_network_pkl(network_pkl)
    return generation_utils.load_pkl_G(da).cuda().eval()


def resolve_methods(method_names):
    return [METHODS_BY_NAME[name] for name in method_names]


def sample_mesh(mesh_utils, G, ws, method: MethodSpec, shape_res: int, truncation_cutoff: int, truncation_psi: float):
    kwargs = dict(
        G=G,
        z=ws,
        conditioning_params=mesh_utils.canonical_pose,
        truncation_cutoff=truncation_cutoff,
        truncation_psi=truncation_psi,
        bordermain=30,
        bordersides=60,
        borderback=80,
        level=method.level,
        shape_res=shape_res,
    )
    if method.use_solidify:
        mesh = mesh_utils.sample_sigmas_to_trimesh_from_ws_and_solidify(**kwargs)
    else:
        mesh = mesh_utils.sample_sigmas_to_trimesh_from_ws(**kwargs)
    mesh.fix_normals()
    return finetuning_utils.half_unit_scale_center_mesh_for_vis(mesh)


def render_mesh_image(mesh_utils, mesh, out_path: Path, win_size: int):
    temp_obj = out_path.with_suffix(".obj")
    try:
        mesh.export(temp_obj)
        try:
            finetuning_utils.clean_inverted_mesh(str(temp_obj), tverts=100000)
        except Exception as exc:
            print(f"mesh cleanup fallback for {out_path.name}: {exc}")
        cleaned_mesh = trimesh.load(temp_obj)
        vis = mesh_utils.visualise_mesh(
            cleaned_mesh,
            ply_fn=str(out_path.with_suffix(".ply")),
            save=False,
            azimuth_angle_initial=-40,
            azimuth_angle_interval=80,
            translate=[0.25, 0.0, -0.25],
            zoom=1.4,
            n_angles=3,
            win_size=win_size,
            opacity_cube=0.1,
            specular=0.35,
            bkgd="#090b0f",
            plotting_kwargs={
                "specular": 0.35,
                "smooth_shading": False,
                "split_sharp_edges": False,
            },
            offset_vis=130,
        )
        vis.save(out_path)
    finally:
        if temp_obj.exists():
            temp_obj.unlink()


def render_seed_image(mesh_utils, G, seed: int, method: MethodSpec, shape_res: int, truncation_cutoff: int, truncation_psi: float, win_size: int, out_path: Path):
    device = torch.device("cuda")
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, 512)).to(device)
    with torch.no_grad():
        ws = G.mapping(
            z=z,
            c=mesh_utils.canonical_pose,
            truncation_cutoff=truncation_cutoff,
            truncation_psi=truncation_psi,
        )
        mesh = sample_mesh(
            mesh_utils=mesh_utils,
            G=G,
            ws=ws,
            method=method,
            shape_res=shape_res,
            truncation_cutoff=truncation_cutoff,
            truncation_psi=truncation_psi,
        )
    render_mesh_image(mesh_utils, mesh, out_path, win_size=win_size)


def combine_before_after(before_path: Path, after_path: Path, out_path: Path):
    before = Image.open(before_path).convert("RGB")
    after = Image.open(after_path).convert("RGB")
    canvas = Image.new("RGB", (before.width + after.width, max(before.height, after.height)), (9, 11, 15))
    canvas.paste(before, (0, 0))
    canvas.paste(after, (before.width, 0))
    canvas.save(out_path, quality=95)


def make_contact_sheet(compare_paths, out_path: Path, columns: int = 4):
    if not compare_paths:
        return
    images = [Image.open(path).convert("RGB") for path in compare_paths]
    width, height = images[0].size
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), (9, 11, 15))
    for idx, image in enumerate(images):
        x = (idx % columns) * width
        y = (idx // columns) * height
        sheet.paste(image, (x, y))
    sheet.save(out_path, quality=92)


def export_state(mesh_utils, G, methods, state_name: str, snapshot_path: str, outdir: Path, seeds, shape_res: int, truncation_cutoff: int, truncation_psi: float, win_size: int, manifest_rows):
    for method in methods:
        state_dir = outdir / method.name / state_name
        state_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            out_path = state_dir / f"seed_{seed}.jpg"
            render_seed_image(
                mesh_utils=mesh_utils,
                G=G,
                seed=seed,
                method=method,
                shape_res=shape_res,
                truncation_cutoff=truncation_cutoff,
                truncation_psi=truncation_psi,
                win_size=win_size,
                out_path=out_path,
            )
            manifest_rows.append(
                {
                    "seed": seed,
                    "method": method.name,
                    "state": state_name,
                    "source_pkl": snapshot_path,
                    "image_path": str(out_path),
                }
            )


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    methods = resolve_methods(args.methods)
    mesh_utils = finetuning_utils.MeshUtilsDataClass()
    manifest_rows = []

    metadata = {
        "tuned_pkl": args.tuned_pkl,
        "baseline_pkl": args.baseline_pkl,
        "outdir": str(outdir),
        "start_seed": args.start_seed,
        "num_seeds": args.num_seeds,
        "shape_res": args.shape_res,
        "truncation_psi": args.truncation_psi,
        "truncation_cutoff": args.truncation_cutoff,
        "methods": [method.__dict__ for method in methods],
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    baseline_G = load_generator(args.baseline_pkl)
    export_state(
        mesh_utils=mesh_utils,
        G=baseline_G,
        methods=methods,
        state_name="untuned",
        snapshot_path=args.baseline_pkl,
        outdir=outdir,
        seeds=seeds,
        shape_res=args.shape_res,
        truncation_cutoff=args.truncation_cutoff,
        truncation_psi=args.truncation_psi,
        win_size=args.win_size,
        manifest_rows=manifest_rows,
    )
    del baseline_G
    torch.cuda.empty_cache()

    tuned_G = load_generator(args.tuned_pkl)
    export_state(
        mesh_utils=mesh_utils,
        G=tuned_G,
        methods=methods,
        state_name="tuned",
        snapshot_path=args.tuned_pkl,
        outdir=outdir,
        seeds=seeds,
        shape_res=args.shape_res,
        truncation_cutoff=args.truncation_cutoff,
        truncation_psi=args.truncation_psi,
        win_size=args.win_size,
        manifest_rows=manifest_rows,
    )
    del tuned_G
    torch.cuda.empty_cache()

    compare_root = outdir / "compare"
    compare_root.mkdir(parents=True, exist_ok=True)
    for method in methods:
        compare_dir = compare_root / method.name
        compare_dir.mkdir(parents=True, exist_ok=True)
        compare_paths = []
        for seed in seeds:
            before_path = outdir / method.name / "untuned" / f"seed_{seed}.jpg"
            after_path = outdir / method.name / "tuned" / f"seed_{seed}.jpg"
            compare_path = compare_dir / f"seed_{seed}_before_after.jpg"
            combine_before_after(before_path, after_path, compare_path)
            compare_paths.append(compare_path)
        make_contact_sheet(compare_paths, compare_dir / "contact_sheet.jpg")

    manifest_path = outdir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "method", "state", "source_pkl", "image_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote mesh bank to: {outdir}")


if __name__ == "__main__":
    main()
