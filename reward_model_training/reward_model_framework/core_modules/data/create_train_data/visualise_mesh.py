import argparse
import sys
import time
from pathlib import Path

import numpy as np
import PIL
import pyvista as pv
import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
EG3D_ROOT = REPO_ROOT / "eg3d"
if str(EG3D_ROOT) not in sys.path:
    sys.path.insert(0, str(EG3D_ROOT))

from shape_utils import convert_sdf_samples_to_ply

from core_modules.data.create_train_data import generation_utils as gen_utils


def visualise_mesh_three(trimesh_object, window_size, zoom=1):
    rot = pv.wrap(trimesh_object)

    # yrot=0
    # zrot=0
    xrot = 90

    # rot = rot.rotate_y(yrot, inplace=False)
    # rot = rot.rotate_z(zrot, inplace=False)
    rot = rot.rotate_x(xrot, inplace=False)

    st = time.time()

    pl = pv.Plotter(window_size=[window_size, window_size], off_screen=True)

    # n_runs=1

    ims = []

    pl.set_background("#363940")
    # _ = pl.add_mesh(rot,pbr=True,metallic=0.1, roughness=0.5,smooth_shading=True)
    mesh1 = pl.add_mesh(rot, smooth_shading=False, color=[220 / 255, 243 / 255, 252 / 255], specular=0.25)

    pl.enable_ssao(kernel_size=64, blur=False)

    pl.set_focus(rot.center)
    pl.camera_position = "yz"
    pl.zoom_camera(1)
    azimuth_angle = -90 - 30  # -60
    pl.camera.Azimuth(azimuth_angle)
    cdim = 100
    azimuth_angle = 30

    # for azimuth_angle in [-45,-30,-15,0,15,30,45]:
    for i in range(4):
        image = pl.screenshot(filename=None, return_img=True)
        ims.append(np.asarray(image)[cdim : window_size - cdim, cdim : window_size - cdim])
        pl.reset_camera()
        pl.camera.Azimuth(azimuth_angle)
        # pl.camera.zoom(zoom)

    ims = ims[1:]
    img = PIL.Image.fromarray(np.hstack(ims)).convert("RGB").copy()  # .save(out_fn)

    pl.close()
    return img


def generate_mesh_visuals(sigma_dir: Path, save_dir: Path, seeds=None, overwrite: bool = False, window_size: int = 2000):
    save_dir.mkdir(parents=True, exist_ok=True)
    sigma_files = list(Path(sigma_dir).glob("entire_sigma_field_256_s_*.pt"))
    if seeds:
        seed_set = {str(s) for s in seeds}
        sigma_files = [f for f in sigma_files if f.stem.split("_")[-1] in seed_set]
    print(f"Found {len(sigma_files)} sigma field files in {sigma_dir}")

    for sf2fn in sigma_files:
        seed = sf2fn.stem.split("_")[-1]
        output_filename = save_dir / f"mesh_cat_s_{seed}.jpg"

        if output_filename.exists() and not overwrite:
            continue

        sf2 = torch.load(sf2fn, map_location="cpu")

        trimesh_object = convert_sdf_samples_to_ply(
            sf2.cpu().numpy(),
            voxel_grid_origin=[-0.5, -0.5, -0.5],
            voxel_size=1.0 / 256,
            ply_filename_out="tmp.obj",
            level=0,
            return_mesh_only=True,
        )

        img = visualise_mesh_three(trimesh_object, window_size=window_size)
        img.save(output_filename, quality=90)
        print(f"[OK] seed {seed} -> {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Visualise sigma field meshes into JPG turntables.")
    parser.add_argument("--sigma_dir", type=Path, default=gen_utils.DEFAULT_SAVE_DIR, help="Directory containing entire_sigma_field_256_s_*.pt")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(gen_utils.DEFAULT_SAVE_DIR) / "visualisations",
        help="Output directory for mesh JPGs",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JPGs")
    parser.add_argument("--window_size", type=int, default=2000, help="PyVista window size (pixels)")
    args = parser.parse_args()

    generate_mesh_visuals(args.sigma_dir, args.out_dir, overwrite=args.overwrite, window_size=args.window_size)


if __name__ == "__main__":
    main()
