"""Backward-compat acceptance check.

Calls main_legacy_click with the EXACT default click_legacy_args set
(the same fixed key set main_hydra builds from cfg_rlhf_tune_AM, with NO
`arch` key) and asserts the resolved G/D class names are the unchanged
EG3D classes. Proves the additive arch branch did not alter default
behaviour. Run from eg3d/ tree root in the conda env:

    python training/panohead_nets/verify_backward_compat.py
"""
import os
import sys

# Mimic the training entrypoint: run as if cwd == eg3d/ so that
# `import train_rlhf`, `dnnlib`, `torch_utils`, `legacy` all resolve.
_EG3D_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _EG3D_DIR not in sys.path:
    sys.path.insert(0, _EG3D_DIR)

import autoroot  # noqa: F401,E402

import train_rlhf  # noqa: E402

# Mirror main_hydra's orig_legacy_args (the fixed default key set), plus
# the required runtime keys from click_legacy_args/defaults.yaml. NO arch.
DEFAULT_ARGS = {
    "cond": True,
    "mirror": False,
    "aug": "noaug",
    "resume": None,
    "freezed": 0,
    "p": 0.2,
    "target": 0.6,
    "batch_gpu": None,
    "cbase": 32768,
    "cmax": 512,
    "glr": None,
    "dlr": 0.002,
    "map_depth": 2,
    "mbstd_group": 4,
    "desc": None,
    "metrics": ["fid1k_partial"],
    "kimg": 25000,
    "tick": 4,
    "snap": 10,
    "seed": 0,
    "nobench": False,
    "workers": 3,
    "dry_run": True,
    "neural_rendering_resolution_initial": 64,
    "neural_rendering_resolution_final": None,
    "neural_rendering_resolution_fade_kimg": 1000,
    "blur_fade_kimg": 200,
    "gen_pose_cond": False,
    "c_scale": 1,
    "c_noise": 0,
    "gpc_reg_prob": 0.5,
    "gpc_reg_fade_kimg": 1000,
    "disc_c_noise": 0,
    "sr_noise_mode": "none",
    "resume_blur": False,
    "sr_num_fp16_res": 4,
    "g_num_fp16_res": 0,
    "d_num_fp16_res": 4,
    "sr_first_cutoff": 2,
    "sr_first_stopband": 2 ** 2.1,
    "style_mixing_prob": 0,
    "sr_module": None,
    "density_reg": 0.25,
    "density_reg_every": 4,
    "density_reg_p_dist": 0.004,
    "reg_type": "l1",
    "decoder_lr_mul": 1,
    "rlhf_config_fn": None,
    "resume_kimg": 0,
    # Runtime keys normally supplied by click_legacy_args/defaults.yaml.
    "cfg": "ffhq",
    "gpus": 1,
    "batch": 4,
    "gamma": 25,
    "data": "/media/krillman/DISK5_1TB/t2_ffhq/eg3d_for_dataset/eg3d_w_mirrore.zip",
    "outdir": "/tmp/_bc_check_outdir",
}


def main():
    c, desc, dry_run = train_rlhf.main_legacy_click(**DEFAULT_ARGS)
    g = c.G_kwargs.class_name
    d = c.D_kwargs.class_name
    print(f"G_kwargs.class_name = {g}")
    print(f"D_kwargs.class_name = {d}")
    print(f"superresolution_module = {c.G_kwargs.rendering_kwargs['superresolution_module']}")
    print(f"D has seg_resolution key: {'seg_resolution' in c.D_kwargs}")
    assert g == "training.triplane.TriPlaneGenerator", f"BC FAIL G: {g}"
    assert d == "training.dual_discriminator.DualDiscriminator", f"BC FAIL D: {d}"
    assert "seg_resolution" not in c.D_kwargs, "BC FAIL: seg kwargs leaked into EG3D D"
    assert c.G_kwargs.rendering_kwargs["superresolution_module"].startswith(
        "training.superresolution"
    ), "BC FAIL: SR module changed for EG3D path"
    print("\nBACKWARD-COMPAT OK: default config still resolves EG3D classes.")


if __name__ == "__main__":
    main()
