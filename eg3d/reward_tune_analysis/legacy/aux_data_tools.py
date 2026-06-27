
import autoroot  # noqa: F401

import os
from pathlib import Path

import torch
from training.volumetric_rendering.ray_sampler import RaySampler

ray_sampler_static = RaySampler()
STATIC_CONFIGS_DIR = Path(os.environ["STATIC_CONFIGS_DIR"])









def subset_from_nose_radius(pcd, radius_cutoff=1.1):
    nose_idx = torch.tensor([8127, 8128, 8255, 8256])
    points = pcd[:, [0, 1, 2]]

    nose_mean_point = points[nose_idx].mean(0)
    nose_mask = torch.norm(points - nose_mean_point, dim=1, p=2) < radius_cutoff
    return pcd[nose_mask]


def downsample_pcd_points(ttl, n_points, perm=None):
    ptc = ttl
    if ptc.shape[0] < n_points:
        pad = torch.zeros_like(ptc[0, :][None, :]).expand(n_points - ptc.shape[0], -1)
        ttl = torch.cat([ptc, pad], dim=0)

    if perm == None:
        perm = torch.randperm(ttl.size(0))
    idx = perm[: min(ttl.size(0), n_points)]
    samples = ttl[idx]

    return samples



def depth_map_to_pcd_from_image(
    depth_map_image,
    n_point_samples_per_pcd_batch=2048,
    return_im=False,
    downsample=False,
    gen_c=None,
    nrs=128,
    radius_cutoff=None,
    return_inverted=False,
    center_mean=False,
    **kwargs,
):  # ,canon_cam=None):
    # fn_depth=create_pt_fn(ddir=ddir_func(seed),ot='triple_dmap',seed=seed)
    # depth_map_image=torch.load(fn_depth,map_location=torch.device('cpu'))[1].unsqueeze(0).squeeze(0,1)


    canon_cam = get_canonical_dmap_cams_for_rlhf()
    cam2world_matrix = canon_cam["cam2world_matrix"]
    intrinsics = canon_cam["intrinsics"]

    if depth_map_image.get_device() == -1:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{depth_map_image.get_device()}")

    if gen_c is None:
        cam2world_matrix = canon_cam["cam2world_matrix"].to(device)
        intrinsics = canon_cam["intrinsics"].to(device)

    else:
        c = gen_c
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

    ray_origins, ray_directions = ray_sampler_static(cam2world_matrix, intrinsics, nrs)
    dd, retmask = imd_to_xyz_with_radius_cutoff(
        image_depth=depth_map_image,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        neural_rendering_resolution=nrs,
        radius_cutoff=radius_cutoff,
    )

    dd = dd[:, :, :].reshape(-1, 3)

    if center_mean:
        pcd = center_points(dd)
        pcd = mean_scale_pts(pcd)
    else:
        pcd = dd

    # and now may sample according to cutoff
    # dd=dd[:,retmask[0],:].reshape(-1,3)

    ptc = pcd[retmask[0]]
    ptc_inv = pcd[~retmask[0]]
    if downsample:
        assert False, "error no downsample anymore in this func"
        # dd_idx=torch.randperm(dd.shape[0])[:n_point_samples_per_pcd_batch]
        # ptc=dd[dd_idx]

    if return_im:
        return (ptc, depth_map_image)

    if return_inverted:
        return (ptc, depth_map_image, ptc_inv)
    return ptc


# converts the image to a point cloud given some depth values
def imd_to_xyz_with_radius_cutoff(image_depth, ray_origins, ray_directions, neural_rendering_resolution, radius_cutoff=None):
    final_dim = neural_rendering_resolution * neural_rendering_resolution
    imd_list = image_depth.reshape(final_dim)
    radius_cutoff_mask = torch.where(imd_list <= torch.max(imd_list))

    if radius_cutoff is not None:
        radius_cutoff_mask = torch.where(imd_list <= radius_cutoff)
    # final_dim=neural_rendering_resolution*neural_rendering_resolution

    if ray_origins.get_device() == -1:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{ray_origins.get_device()}")

    imd = image_depth.reshape(1, final_dim).unsqueeze(2).expand(1, final_dim, 3).to(device)
    retval = ray_origins + imd * ray_directions
    return (retval, radius_cutoff_mask)


def get_canonical_dmap_cams_for_rlhf():
    tdmap_cams = torch.load(
        STATIC_CONFIGS_DIR / "triple_dmap_cameras.pt",
        map_location=torch.device("cpu"),
    )
    canon_cam = tdmap_cams[1].unsqueeze(0)
    c = canon_cam
    cam2world_matrix = c[:, :16].view(-1, 4, 4)
    intrinsics = c[:, 16:25].view(-1, 3, 3)

    return dict(cam2world_matrix=cam2world_matrix, intrinsics=intrinsics, gen_c=c)


def get_triple_dmap_cams_for_rlhf():
    tdmap_cams = torch.load(
        STATIC_CONFIGS_DIR / "triple_dmap_cameras.pt",
        map_location=torch.device("cpu"),
    )
    
    intrinsics_list=[]
    gen_c_list=[]
    c2w_mat_list=[]
    
    for i in range(3):
        
        canon_cam = tdmap_cams[i].unsqueeze(0)
        c = canon_cam
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)
        
        intrinsics_list.append(intrinsics)
        
        gen_c_list.append(c)
        c2w_mat_list.append(c)

    return dict(cam2world_matrix=c2w_mat_list, intrinsics=intrinsics_list, gen_c=gen_c_list)

# vertices weightex according to area (sum) of all faces connected to any vertex. Shouold allow us to sample from vertex with large area, more ideal than random sampling!!
def load_vertex_sampling_weights_dmap_128():
    prob_sampling_faces = torch.load(
        STATIC_CONFIGS_DIR / "weight_sampling_for_canon_pcd_depth_map_128.pt",
        map_location="cpu",
    )
    return prob_sampling_faces


def mean_scale_pts(ttl):
    # print(ttl)
    ttl_c = ttl - ttl.mean(dim=0, keepdim=True)
    scale = (1 / ttl_c.abs().max()) * 0.999999
    ttl_c = ttl_c * scale
    return ttl_c


def center_points(ttl):
    # print(ttl)
    ttl_c = ttl - ttl.mean(dim=0, keepdim=True)
    return ttl_c







# class depth_map_to_pcd_as_pt(nn.Module):


# 	def __init__(self,depth_map_image,
#     n_point_samples_per_pcd_batch=2048,
#     return_im=False,
#     downsample=False,
#     #gen_c=None,
#     nrs=128,
#     #radius_cutoff=None,
#     #return_inverted=False,
#     center_mean=False,
#     device=torch.device('cuda')):

# 	self.n_point_samples_per_pcd_batch=n_point_samples_per_pcd_batch
# 	self.nrs=nrs
# 	self.center_mean=center_mean
# 	self.canon_cam=get_canonical_dmap_cams_for_rlhf()
# 	self.cam2world_matrix=canon_cam["cam2world_matrix"]
# 	self.intrinsics=canon_cam["intrinsics"]
# 	self.device=device




# 	def forward
