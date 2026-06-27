ALL_DATA_TYPES = [
    "triple_dmap",
    "single_dmap",
    "ws_code_view_conditioned",
    "ws_code_unconditioned",
    "triple_rgb_lmks_98",
    "single_rgb_and_dmap",
    "canonical_rgb_lmks_98",
    "aw98_3d_lmks",
    "triple_rgb",
    "single_rgb",
    "mesh_crop_ptgeom",
    "pcd",
    "sigma_field_256",
    "sigma_field_128",
    "sigma_field_64",
    "aw98_patch_normals_nose_8",
    "aw98_patch_geom_nose_8",
    "aw98_patch_geom_all98_8",
    "aw98_patch_rgb_4region_32",
    # NB: experiment that was tried; NO return_single_data dispatch branch exists
    # for it (selecting it would raise). Kept as a record only -- the run configs
    # under core_modules/logs preserve the experiment. Do not select without
    # implementing a branch.
    "pcd_patches",
    "nose_512",
    "point_cloud_entire",
    "pcd_nose_combined",
]
