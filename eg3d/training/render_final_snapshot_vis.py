import autoroot  # noqa: F401

from pathlib import Path

import numpy as np
import torch
import trimesh
from core_modules.data.create_train_data import generation_utils
from core_modules.utils import finetuning_utils


def render_final_snapshot_vis(
    snapshot_pkl,
    seed=9100080,
    truncation_cutoff=14,
    truncation_psi=0.7,
    level=30,
    shape_res=512,
):
    """Render a single mesh preview for the final training snapshot."""
    da = generation_utils.set_defaults(generation_utils.DArgs())
    da.set_network_pkl(snapshot_pkl)
    da.G = generation_utils.load_pkl_G(da).cuda()

    mesh_utils = finetuning_utils.MeshUtilsDataClass()
    device = torch.device("cuda")
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, 512)).to(device)

    with torch.no_grad():
        ws = da.G.mapping(
            z=z,
            c=mesh_utils.canonical_pose,
            truncation_cutoff=truncation_cutoff,
            truncation_psi=truncation_psi,
        )
        solid_mesh = mesh_utils.sample_sigmas_to_trimesh_from_ws_and_solidify(
            da.G,
            ws,
            conditioning_params=mesh_utils.canonical_pose,
            truncation_cutoff=truncation_cutoff,
            truncation_psi=truncation_psi,
            bordermain=30,
            bordersides=60,
            borderback=80,
            level=level,
            shape_res=shape_res,
        )

    solid_mesh.fix_normals()
    solid_mesh = finetuning_utils.half_unit_scale_center_mesh_for_vis(solid_mesh)

    snapshot_path = Path(snapshot_pkl)
    temp_obj = snapshot_path.with_name(f"{snapshot_path.stem}_vis_tmp.obj")
    out_path = snapshot_path.with_name(f"vis_out_seed_{seed}.jpg")

    try:
        solid_mesh.export(temp_obj)
        finetuning_utils.clean_inverted_mesh(str(temp_obj), tverts=100000)
        cleaned_mesh = trimesh.load(temp_obj)

        vis = mesh_utils.visualise_mesh(
            cleaned_mesh,
            ply_fn="hi.ok",
            save=False,
            azimuth_angle_initial=-40,
            azimuth_angle_interval=80,
            translate=[0.25, 0.0, -0.25],
            zoom=1.4,
            n_angles=3,
            win_size=4096,
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

    return out_path
