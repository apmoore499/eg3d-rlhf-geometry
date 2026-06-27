import autoroot  # noqa: F401

from core_modules.data.create_train_data import generation_utils


#
# import synth_ffhq_meshes


# sigs=torch.load('sigs.pt')


import torch


import torch
import time
from tqdm import tqdm
import os
import mrcfile
import numpy as np
from pathlib import Path as plp


da = generation_utils.DArgs()
da = generation_utils.set_defaults(da)
# da.set_network_pkl('/path/to/eg3d-rlhf-geometry/pkl_pt/eg3d_1/ffhq512-128.pkl')
G_fn = "/path/to/eg3d-rlhf-geometry/000_RLHF_AM/rlhf_training-runs/archived/00070-ffhq-eg3d_rebal_02_07_2023_10k_uniform_yaws-gpus1-batch2-gamma50_mse_pixelwise_sigma_reg/network-snapshot-000200.pkl"


G_fn = "/path/to/data/training_runs_2/01281-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002053.pkl"
# G_fn='/path/to/eg3d-rlhf-geometry/000_RLHF_AM/rlhf_training-runs/archived/00070-ffhq-eg3d_rebal_02_07_2023_10k_uniform_yaws-gpus1-batch2-gamma50_mse_pixelwise_sigma_reg/network-snapshot-000200.pkl'

# G_fn='/path/to/data/training_runs_2/01275-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002053.pkl'
# G_fn='/path/to/data/training_runs_2/01298-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002053.pkl'
# G_fn='/path/to/data/training_runs_2/01299-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002053.pkl'
# G_fn='/path/to/data/training_runs_2/01300-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002053.pkl'

# G_fn='/path/to/data/training_runs_2/01301-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002053.pkl'
# G_fn='/path/to/data/training_runs_2/01302-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002053.pkl'

# G_fn='/path/to/data/training_runs_2/01303-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002053.pkl'

# G_fn='/path/to/data/training_runs_2/01305-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002053.pkl'


G_list = [
    "/path/to/data/training_runs_2/01275-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002053.pkl",
    "/path/to/data/training_runs_2/01298-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002053.pkl",
    "/path/to/data/training_runs_2/01299-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002053.pkl",
    "/path/to/data/training_runs_2/01300-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002053.pkl",
    "/path/to/data/training_runs_2/01301-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002053.pkl",
    "/path/to/data/training_runs_2/01302-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002053.pkl",
    "/path/to/data/training_runs_2/01303-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002053.pkl",
    "/path/to/data/training_runs_2/01305-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002053.pkl",
]


G_list = ["/path/to/data/training_runs_2/01306-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002053.pkl"]
G_list = ["/path/to/data/training_runs_2/01307-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002053.pkl"]

G_list = ["/path/to/data/training_runs_2/01310-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002058_LAST.pkl"]
from core_modules.utils import finetuning_utils

MUDC = finetuning_utils.MeshUtilsDataClass()

seed = 2

# G_list=G_list[1:]

START_SEED = 9100000
N_SEEDS = 40

srange = list(range(START_SEED, N_SEEDS))


LEVEL = 30
SHAPE_RES = 512
SPECULAR = 0.27

WIN_SIZE = 4096

AZA = 35


import sys


pk = {
    "specular": SPECULAR,
    "smooth_shading": False,
    "split_sharp_edges": False,
}
#'enable_anti_aliasing':True,}
#'pbr':True, 'metallic':0.3, 'roughness':0.1}


def main(G_fn):
    # orig_G_fn='/path/to/eg3d-rlhf-geometry/pkl_pt/eg3d_1/ffhq512-128.pkl'

    da.set_network_pkl(G_fn)
    da.G = generation_utils.load_pkl_G(da).cuda()

    seed = 9100080

    z = torch.from_numpy(np.random.RandomState(seed).randn(1, 512)).to(torch.device("cuda"))

    # torch.from_numpy(np.random.normal((1,512),))
    # torch.randn((1,512)
    # sigs = MUDC.sample_sigma_rays_from_z(G=da.G, c=MUDC.canonical_pose, shape_res=256, truncation_cutoff=14, truncation_psi=0.7, noise_mode="const", device=torch.device("cuda"), cl_frac=1.0, z=z)

    with torch.no_grad():
        ws = da.G.mapping(z=z, c=MUDC.canonical_pose, truncation_cutoff=14, truncation_psi=0.7)
        solid_mesh = MUDC.sample_sigmas_to_trimesh_from_ws_and_solidify(da.G, ws, conditioning_params=MUDC.canonical_pose, truncation_cutoff=14, truncation_psi=0.7, bordermain=30, bordersides=60, borderback=80, level=LEVEL, shape_res=SHAPE_RES)
    #
    # MUDC.export_sample_mrc(sigmas=sigs, out_fn=f'/path/to/eg3d-rlhf-geometry/000_RLHF_AM/rlhf_training-runs/archived/00070-ffhq-eg3d_rebal_02_07_2023_10k_uniform_yaws-gpus1-batch2-gamma50_mse_pixelwise_sigma_reg/t1_s_{seed}_G_{os.path.basename(G_fn).replace(".pkl",".mrc")}')
    # vis=MUDC.visualise_mesh(solid_mesh,ply_fn='hi.ok',save=False,azimuth_angle_initial=-35,azimuth_angle_interval=70,translate=[0.5,0.5,0.5],zoom=1.8,n_angles=3,win_size=2048,opacity_cube=0.1)
    # pk['specular']=SPECULAR

    solid_mesh.fix_normals()
    solid_mesh = finetuning_utils.half_unit_scale_center_mesh_for_vis(solid_mesh)

    solid_mesh.export("tmp.obj")
    finetuning_utils.clean_inverted_mesh("tmp.obj", tverts=100000)
    import trimesh

    solid_mesh = trimesh.load("tmp.obj")

    AZA = 40

    bkgd = "#090b0f"
    specular = 0.35

    pk["specular"] = specular

    plotting_kwargs = pk
    translate = [0.25, 0.0, -0.25]
    SPECULAR = specular
    WIN_SIZE = 4096
    vis = MUDC.visualise_mesh(solid_mesh, ply_fn="hi.ok", save=False, azimuth_angle_initial=-AZA, azimuth_angle_interval=AZA * 2, translate=translate, zoom=1.4, n_angles=3, win_size=WIN_SIZE, opacity_cube=0.1, specular=SPECULAR, bkgd=bkgd, plotting_kwargs=pk, offset_vis=130)
    # vis.save(out_fn)

    out_dir = plp(G_fn).parent

    del solid_mesh

    odir_fn = plp(G_fn).parent.joinpath(f"vis_out_seed_{seed}.jpg")

    print(odir_fn)

    vis.save(odir_fn)

    # Your main logic here
    # Example: Do something with the argument
    result = f"save ti: {str(odir_fn).upper()}"
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_template.py <argument>")
        sys.exit(1)

    argument = sys.argv[1]
    result = main(argument)
    print(result)
    sys.exit(0)
