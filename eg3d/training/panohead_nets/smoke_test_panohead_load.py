"""PHASE-1 smoke test for the vendored PanoHead nets.

Builds the vendored PanoHead TriPlaneGenerator + MaskDualDiscriminatorV2
with the exact kwargs the train_rlhf.py `arch=="panohead"` branch uses,
loads the PanoHead checkpoint into them (reporting matched vs skipped
params), then runs the rlhf reward sigma-extraction path
(MeshUtilsDataClass.sample_sigma_rays_from_z) to confirm G.sample works
end to end. No training is performed.

Run from the eg3d/ tree root in the conda env:
    python training/panohead_nets/smoke_test_panohead_load.py
"""
import copy
import os
import sys

# Mimic the training entrypoint: cwd == eg3d/ so imports resolve.
_EG3D_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _EG3D_DIR not in sys.path:
    sys.path.insert(0, _EG3D_DIR)

import autoroot  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import dnnlib  # noqa: E402
import legacy  # noqa: E402
from torch_utils import misc  # noqa: E402
from camera_utils import FOV_to_intrinsics, LookAtPoseSampler  # noqa: E402

PKL = "/home/krillman/Documents/eg3dredo/PanoHead/models/easy-khair-180-gpc0.8-trans10-025000.pkl"

G_CLASS = "training.panohead_nets.triplane.TriPlaneGenerator"
D_CLASS = "training.panohead_nets.dual_discriminator.MaskDualDiscriminatorV2"


def build_rendering_kwargs():
    # Mirror exactly what main_legacy_click builds for arch=="panohead".
    return {
        "image_resolution": 512,
        "disparity_space_sampling": False,
        "clamp_mode": "softplus",
        "superresolution_module": "training.panohead_nets.superresolution.SuperresolutionHybrid8XDC",
        "c_gen_conditioning_zero": False,
        "gpc_reg_prob": 0.8,
        "c_scale": 1.0,
        "superresolution_noise_mode": "none",
        "density_reg": 0.0,
        "density_reg_p_dist": 0.004,
        "reg_type": "l1",
        "decoder_lr_mul": 1.0,
        "decoder_activation": "none",
        "use_torgb_raw": True,
        "triplane_size": 256,
        "triplane_depth": 3,
        "trans_reg": 10.0,
        "use_background": True,
        "sr_antialias": True,
        "depth_resolution": 48,
        "depth_resolution_importance": 48,
        "ray_start": 2.25,
        "ray_end": 3.3,
        "box_warp": 1,
        "avg_camera_radius": 2.7,
        "avg_camera_pivot": [0, 0, 0.2],
    }


def count_match(src, dst):
    """Count dst params/buffers whose name exists in src with same shape."""
    src_named = dict(list(src.named_parameters()) + list(src.named_buffers()))
    matched = 0
    skipped = 0
    skipped_names = []
    for name, tensor in list(dst.named_parameters()) + list(dst.named_buffers()):
        if name in src_named and src_named[name].shape == tensor.shape:
            matched += 1
        else:
            skipped += 1
            if len(skipped_names) < 20:
                skipped_names.append(name)
    return matched, skipped, skipped_names


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    rk = build_rendering_kwargs()
    mapping_kwargs = dnnlib.EasyDict(num_layers=2)
    sr_kwargs = dnnlib.EasyDict(
        channel_base=32768, channel_max=512, fused_modconv_default="inference_only"
    )

    print("\n=== Building vendored PanoHead G ===")
    G = dnnlib.util.construct_class_by_name(
        class_name=G_CLASS,
        z_dim=512,
        c_dim=25,
        w_dim=512,
        img_resolution=512,
        img_channels=3,
        sr_num_fp16_res=4,
        num_fp16_res=0,
        conv_clamp=None,
        channel_base=32768,
        channel_max=512,
        fused_modconv_default="inference_only",
        mapping_kwargs=mapping_kwargs,
        rendering_kwargs=rk,
        sr_kwargs=sr_kwargs,
    )

    print("=== Building vendored PanoHead D ===")
    D = dnnlib.util.construct_class_by_name(
        class_name=D_CLASS,
        c_dim=25,
        img_resolution=512,
        img_channels=3,
        seg_resolution=128,
        seg_channels=1,
        channel_base=32768,
        channel_max=512,
        num_fp16_res=4,
        conv_clamp=256,
        block_kwargs=dnnlib.EasyDict(freeze_layers=0),
        epilogue_kwargs=dnnlib.EasyDict(mbstd_group_size=4),
        disc_c_noise=0.0,
    )

    print("\n=== Loading checkpoint ===")
    with dnnlib.util.open_url(PKL) as f:
        resume = legacy.load_network_pkl(f)

    # training_loop sets requires_grad_(False) on modules before copy.
    G.requires_grad_(False)
    D.requires_grad_(False)
    G_ema = copy.deepcopy(G)
    misc.copy_params_and_buffers(resume["G"], G, require_all=False)
    misc.copy_params_and_buffers(resume["G_ema"], G_ema, require_all=False)
    misc.copy_params_and_buffers(resume["D"], D, require_all=False)

    mg, sg, sgn = count_match(resume["G"], G)
    me, se, _ = count_match(resume["G_ema"], G_ema)
    md, sd, sdn = count_match(resume["D"], D)
    print(f"G    : matched={mg}  skipped={sg}")
    print(f"G_ema: matched={me}  skipped={se}")
    print(f"D    : matched={md}  skipped={sd}")
    if sgn:
        print(f"  G skipped (first {len(sgn)}): {sgn}")
    if sdn:
        print(f"  D skipped (first {len(sdn)}): {sdn}")

    G = G.to(device).eval()

    print("\n=== Reward-path sigma extraction (G.sample via MUDC) ===")
    from core_modules.utils.finetuning_utils import (
        MeshUtilsDataClass,
    )

    MUDC = MeshUtilsDataClass()

    # Valid 25-dim camera/conditioning params (FFHQ pivot [0,0,0.2], r=2.7).
    cam_pivot = torch.tensor(
        G.rendering_kwargs.get("avg_camera_pivot", [0, 0, 0.2]), device=device
    )
    cam_radius = G.rendering_kwargs.get("avg_camera_radius", 2.7)
    intrinsics = FOV_to_intrinsics(18.837, device=device)
    cond_pose = LookAtPoseSampler.sample(
        np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=device
    )
    c = torch.cat([cond_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    z = torch.randn(1, 512, device=device)
    sigmas = MUDC.sample_sigma_rays_from_z(
        G, z=z, c=c, shape_res=64, device=device,
        truncation_psi=0.7, truncation_cutoff=14, noise_mode="const",
    )
    print(f"sigma cube shape = {tuple(sigmas.shape)}")
    sf = sigmas.float()
    print(
        f"sigma min={sf.min().item():.4f} "
        f"max={sf.max().item():.4f} mean={sf.mean().item():.4f}"
    )
    assert tuple(sigmas.shape) == (64, 64, 64), "sigma cube wrong shape"

    print("\nSMOKE TEST PASSED.")


if __name__ == "__main__":
    main()
