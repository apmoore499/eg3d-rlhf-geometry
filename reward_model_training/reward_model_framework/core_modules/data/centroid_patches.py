"""Centroid-patch construction for the AW98 keypoint-patch dtypes.

Pure tensor helpers that crop small image-shaped PATCHES of a point cloud around a
set of keypoint centroids, used by the `aw98_patch_*` dtypes in
`dset_loaders.py` (and by the RLHF finetuning path). Extracted out of
`dset_single_stream_ordered_minimal` so the dataset class stays focused on dispatch;
these never used instance state, so they are plain functions.

Pipeline: kNN-gather `point_size` neighbours around each centroid
(`knn_clustering`), optionally append per-patch FFT-magnitude (`frequency_analysis`)
and rgb/normals channels, then reshape each patch to (C, S, S). See each dtype's
branch for the channel/size it requests and the documented return shape.
"""

import torch


def normalize_point_cloud(point_cloud):
    """Normalizes a 3D point cloud to fit within a unit sphere and ensure that each dimension has zero mean and unit variance.

    Args:
        point_cloud (torch.Tensor): A tensor of shape (N, 3) representing a 3D point cloud.

    Returns:
        A normalized point cloud as a torch.Tensor of shape (N, 3).
    """
    point_cloud_np = point_cloud
    # Center the point cloud by subtracting the mean along each dimension
    point_cloud_centered = point_cloud_np - torch.mean(point_cloud_np, axis=0)

    # Scale the point cloud to fit within a unit sphere
    scale = torch.max(torch.sqrt(torch.sum(point_cloud_centered**2, axis=1)))
    point_cloud_normalized = point_cloud_centered / scale

    # Normalize each dimension to have zero mean and unit variance
    point_cloud_normalized_mean = torch.mean(point_cloud_normalized, axis=0)
    point_cloud_normalized_std = torch.std(point_cloud_normalized, axis=0)
    point_cloud_normalized = (point_cloud_normalized - point_cloud_normalized_mean) / point_cloud_normalized_std

    return point_cloud_normalized


def knn_clustering(points, centroids, k=512):
    """Performs K nearest neighbor clustering to group the points around each centroid into patches.

    Args:
        points (torch.Tensor): (N, D) tensor of point cloud coordinates
        centroids (torch.Tensor): (num_centroids, D) tensor of centroid coordinates
        k (int): number of nearest neighbors to select for each centroid

    Returns:
        patches (list of torch.Tensor): list of length num_centroids containing tensors
                                        of size (k, D) representing the patches around
                                        each centroid
    """
    num_centroids = centroids.size(0)
    num_points = points.size(0)

    # Expand centroids and points tensors to compute pairwise distance
    expanded_centroids = centroids.unsqueeze(1).expand(num_centroids, num_points, -1)
    expanded_points = points.unsqueeze(0).expand(num_centroids, num_points, -1)

    # Compute squared Euclidean distances: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    distances = torch.sum((expanded_centroids - expanded_points) ** 2, dim=2)

    # Find the k nearest points for each centroid
    _, indices = torch.topk(distances, k, largest=False, dim=1)

    # Gather the patches for each centroid
    patches = [points[indices[i]] for i in range(num_centroids)]

    return patches


def frequency_analysis(point_cloud):
    # Compute the Fourier transform of the point cloud
    fourier_transform = torch.fft.fftn(point_cloud)

    # Compute the frequency magnitude and shift the zero frequency component to the center of the spectrum
    freq_magnitude = torch.abs(torch.fft.fftshift(fourier_transform))

    return freq_magnitude


def get_processed_patches_rgb(point_cloud_tensor, rgb_data, centroids, patch_size, point_size, input_channel, input_size, add_freq=True):
    patches = knn_clustering(point_cloud_tensor, centroids, k=point_size)
    patches_rgb = knn_clustering(rgb_data, centroids, k=point_size)

    if add_freq:
        for i in range(patch_size):
            freq_mag_shifted = frequency_analysis(patches[i])
            patches[i] = torch.cat([patches[i], patches_rgb[i], freq_mag_shifted], dim=1)
            patches[i] = torch.reshape(patches[i], (input_channel, input_size, input_size))
    else:
        for i in range(patch_size):
            patches[i] = torch.reshape(patches_rgb[i], (input_channel, input_size, input_size))

    patches = [p.unsqueeze(0) for p in patches]

    patches = torch.vstack(patches)
    return patches


def get_processed_patches_normals(point_cloud_tensor, centroids, patch_size, point_size, input_channel, input_size, add_freq=True, center_at_centroid=False):
    from pytorch3d import ops

    patches = knn_clustering(point_cloud_tensor, centroids, k=point_size)

    # calc normals on the fly from the patches, makes training quicker. instead of doing entire point cloud
    patches_normals = ops.points_normals.estimate_pointcloud_normals(torch.cat([p[None, ...] for p in patches], dim=0), neighborhood_size=50)  # 50 is default neighbour size

    if add_freq:
        for i in range(patch_size):
            if center_at_centroid:
                patches[i] = patches[i] - centroids[i]

            freq_mag_shifted = frequency_analysis(patches[i])
            patches[i] = torch.cat([patches[i], patches_normals[i], freq_mag_shifted], dim=1)
            patches[i] = torch.reshape(patches[i], (input_channel, input_size, input_size))
    else:
        for i in range(patch_size):
            patches[i] = torch.reshape(patches_rgb[i], (input_channel, input_size, input_size))  # noqa: F821

    patches = [p.unsqueeze(0) for p in patches]

    patches = torch.vstack(patches)
    return patches


def get_processed_patches_rgb_no_colour(point_cloud_tensor, centroids, patch_size, point_size, input_channel, input_size, add_freq=True, center_at_centroid=False):
    # input size = N,3
    # no batch
    patches = knn_clustering(point_cloud_tensor, centroids, k=point_size)

    if add_freq:
        for i in range(patch_size):
            if center_at_centroid:
                patches[i] = patches[i] - centroids[i]

            freq_mag_shifted = frequency_analysis(patches[i])
            patches[i] = torch.cat([patches[i], freq_mag_shifted], dim=1)

            patches[i] = patches[i].permute(1, 0)

            patches[i] = patches[i].reshape(input_channel, input_size, input_size)
    else:
        assert False, "we are only adding frequency for processed patches..."

    patches = [p.unsqueeze(0) for p in patches]

    patches = torch.vstack(patches)
    return patches


def normalize_pcd_and_get_processed_patches_no_colour(pcds, seed_centroids, rndm_groups):
    pcds = [normalize_point_cloud(p) for p in pcds]

    all_patches = []

    rndm_groups = rndm_groups.flatten()

    for pcd, dict_of_centroids in zip(pcds, seed_centroids):
        centroids_list = []

        for g in rndm_groups:
            centroids_list.append(pcd[dict_of_centroids[g]])
        centroids = torch.vstack(centroids_list)
        patches = get_processed_patches_rgb_no_colour(pcd, centroids=centroids, patch_size=len(centroids), point_size=64, input_channel=6, input_size=8, add_freq=True, center_at_centroid=False)

        all_patches.append(patches.unsqueeze(0))

    return all_patches


__all__ = [
    "normalize_point_cloud",
    "knn_clustering",
    "frequency_analysis",
    "get_processed_patches_rgb",
    "get_processed_patches_normals",
    "get_processed_patches_rgb_no_colour",
    "normalize_pcd_and_get_processed_patches_no_colour",
]
