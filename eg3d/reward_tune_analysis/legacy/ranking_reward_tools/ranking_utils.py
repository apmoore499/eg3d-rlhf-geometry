import copy
import glob
import os
import pickle
import random
import shutil
import sys
import time

import autoroot  # noqa: F401
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont

# pretrained simclr...


# get the training loop


font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf", 500)
font_small = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf", 100)

ffhq_rendering_options = {
    "depth_resolution": 48,  # number of uniform samples to take per ray.
    "depth_resolution_importance": 48,  # number of importance samples to take per ray.
    "ray_start": 2.25,  # near point along each ray to start taking samples.
    "ray_end": 3.3,  # far point along each ray to stop taking samples.
    "box_warp": 1,  # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
    "avg_camera_radius": 2.7,  # used only in the visualizer to specify camera orbit radius.
    "avg_camera_pivot": [
        0,
        0,
        0.2,
    ],  # used only in the visualizer to control center of camera rotation.
}

random.seed(43)
np.random.seed(43)
torch.manual_seed(43)


def compose_images_horizontally(image1_path, image2_path):
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Get the dimensions of the images
    width1, height1 = image1.size
    width2, height2 = image2.size

    gap_width = 200
    # Calculate the new dimensions of the composed image
    new_width = width1 + width2 + gap_width
    new_height = max(height1, height2)

    # Create a new image with the new dimensions
    new_image = Image.new("RGB", (new_width, new_height), color=(255, 0, 0))

    # Paste the first image onto the new image
    new_image.paste(image1, (0, 0))

    # Paste the second image onto the new image
    new_image.paste(image2, (width1 + gap_width, 0))

    # Return the new image
    return new_image


def compose_images_vertically(image1_path, image2_path):
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    # rescale smallest image
    w = 3584 * 2
    h = int(2098 / 2) * 2

    image1 = image1.resize((w, h))
    image2 = image2.resize((w, h))
    # Get the dimensions of the images
    width1, height1 = image1.size
    width2, height2 = image2.size

    gap_height = 100
    # Calculate the new dimensions of the composed image
    new_width = max(width1, width2)
    new_height = height1 + height2 + gap_height

    # Create a new image with the new dimensions
    new_image = Image.new("RGB", (new_width, new_height), color=(255, 0, 0))

    # Paste the first image onto the new image
    new_image.paste(image1, (0, 0))

    # Paste the second image onto the new image
    new_image.paste(image2, (0, height1 + gap_height))

    # Return the new image
    return new_image


def compose_and_save(fn1, fn2, current_id, current_composed_dir, rescale_final=0.25):
    composed_name = os.path.join(current_composed_dir, "composed_" + str(current_id) + ".jpg")

    if os.path.exists(composed_name):
        return composed_name
    else:
        scaling = 1.0
        composed = compose_images_vertically(fn1, fn2)

        composed_height = composed.size[1]
        composed_width = composed.size[0]

        # Create a drawing object
        draw = ImageDraw.Draw(composed, "RGBA")

        # Define the text to be written
        A_text = "A"
        B_text = "B"
        # Define the position where the text should be written
        A_position = (int(1400 * scaling), composed_height / 2 - composed_height / 5)
        B_position = (int(1400 * scaling), composed_height - composed_height / 5)
        # Write the text on the image
        draw.text(A_position, A_text, font=font, fill=(255, 0, 0, 128))
        draw.text(B_position, B_text, font=font, fill=(255, 0, 0, 128))

        # Write file name so we can backtraack later
        fn1_position = A_position - np.array([int(1380 * scaling), -700])
        fn2_position = B_position - np.array([int(1380 * scaling), -700])

        fn1 = fn1.split("/")[-1]
        fn2 = fn2.split("/")[-1]
        draw.text(fn1_position, fn1, font=font_small, fill=(255, 0, 0, 128))
        draw.text(fn2_position, fn2, font=font_small, fill=(255, 0, 0, 128))

        # Define the position where the text should be written
        A_position = (
            composed_width - int(1200 * scaling),
            composed_height / 2 - composed_height / 5,
        )
        B_position = (composed_width - int(1200 * scaling), composed_height - composed_height / 5)

        draw.text(A_position, A_text, font=font, fill=(255, 0, 0, 128))
        draw.text(B_position, B_text, font=font, fill=(255, 0, 0, 128))

        new_size = (
            int(composed_width * rescale_final * 0.5),
            int(composed_height * rescale_final * 0.5),
        )
        composed = composed.resize(new_size)
        composed.save(composed_name, optimize=True, quality=85)
        return composed_name


from PIL import Image
from PIL.Image import Resampling


def convert_png_to_jpg(png_file, im_scale=0.5):
    img = Image.open(png_file)
    csize = img.size
    cs_new = (int(csize[0] * im_scale), int(csize[1] * im_scale))

    new_fn = png_file.replace(".png", ".jpg")

    img.resize(cs_new, Resampling.LANCZOS).save(new_fn, optimize=True, quality=85)  #########
    os.remove(png_file)

    print("successfully create jpg and remove png")


def convert_all_png_to_jpg(png_in_dir):
    print(f"converting {len(png_in_dir)} images")
    for p in png_in_dir:
        convert_png_to_jpg(p)


def get_all_meshes_for_binary(select_only=None):
    rlhf_meshes_dir = "/path/to/eg3d-rlhf-geometry/000_RLHF_AM/rlhf_meshes"
    rlhf_ffhq_folders = [f"{rlhf_meshes_dir}/{f}" for f in os.listdir(rlhf_meshes_dir) if "ffhq" in f]

    print(select_only)

    if select_only is not None:
        for s in select_only:
            rlhf_ffhq_folders = [f"{rlhf_meshes_dir}/{f}" for f in os.listdir(rlhf_meshes_dir) if "ffhq" in f and s in f]

    total_meshes = 0
    seeds_with_meshes = []
    all_dfs = []
    for f in rlhf_ffhq_folders:
        composed_ims = glob.glob(f"{f}/composed_for_ranking/*_composed.jpg")
        total_meshes += len(composed_ims)
        current_seeds = [c.split("/")[-1].split("_")[0] for c in composed_ims]
        single_dmap_fns = [f"{f}/seed{s}.json" for s in current_seeds]
        three_dmap_fns = [f"{f}/seed{s}_three_dmaps.json" for s in current_seeds]
        has_single_dmap = [os.path.exists(f) for f in single_dmap_fns]
        has_three_dmap = [os.path.exists(f) for f in three_dmap_fns]
        has_both = np.logical_and(has_single_dmap, has_three_dmap)

        current_seeds_with_meshes = np.array(current_seeds)[has_both]
        current_dmap_single = np.array(single_dmap_fns)[has_both]
        current_dmap_triple = np.array(three_dmap_fns)[has_both]

        current_composed_ims = np.array(composed_ims)[has_both]
        current_df = pd.DataFrame(
            {
                "seed": current_seeds_with_meshes,
                "composed_im": current_composed_ims,
                "single_dmap": current_dmap_single,
                "three_dmap": current_dmap_triple,
            }
        )

        seeds_with_meshes += [s for s in current_seeds_with_meshes]

        all_dfs.append(current_df)

    print("----")

    print("total meshes avail")
    print(total_meshes)  # 994 meshes
    print("total meshes with depth map")
    print(len(seeds_with_meshes))  # 994 meshes
    joined_all_dfs = pd.concat(all_dfs)

    return joined_all_dfs


def create_pairs_for_ranking_df():
    joined_orig_df = joined_all_dfs_for_merge.copy()
    meshes_for_ranking = joined_all_dfs_for_merge.copy()

    # split into pairs
    all_unique_idx = [i for i in meshes_for_ranking.unique_idx]
    n_idx = len(all_unique_idx)  # all images for composition
    half_idx = int(n_idx / 2)  # half for binary ranking

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    first_pair_idx = random.sample(population=all_unique_idx, k=half_idx)  # should be 0,2,4,6,8,etc
    remaining_idx = [i for i in all_unique_idx if i not in first_pair_idx]  # should be 1,3,5,7,,etc
    second_pair_idx = random.sample(remaining_idx, half_idx)

    has_both_idx = [i for i in all_unique_idx if (i in first_pair_idx or i in second_pair_idx)]

    joined_pair = pd.DataFrame({"first_pair_idx": first_pair_idx, "second_pair_idx": second_pair_idx})

    joined_pair_first_fn = [meshes_for_ranking.loc[i, "composed_im"] for i in joined_pair.first_pair_idx]
    joined_pair_sec_fn = [meshes_for_ranking.loc[i, "composed_im"] for i in joined_pair.second_pair_idx]

    joined_pair["joined_pair_first_fn"] = joined_pair_first_fn
    joined_pair["joined_pair_sec_fn"] = joined_pair_sec_fn

    joined_pair.reset_index(inplace=True, drop=True)
    joined_pair["unique_idx"] = joined_pair.index
    joined_pair["composed_fn"] = "-"  # initialise composed_fn column

    joined_pair.to_csv(pairs_for_ranking_df_fn, index=False)  # save out first


def get_cposed_fn(seed):
    cposed_fn = f"/path/to/eg3d-rlhf-geometry/000_RLHF_AM/rlhf_meshes/rlhf_meshes_ffhq512-128_const_noise_t1/composed_for_ranking/{seed}_composed.jpg"
    return cposed_fn


def make_binary_comparisons_07_08_2023(joined_all_dfs):
    c = [
        "first_pair_idx",
        "second_pair_idx",
        "joined_pair_first_fn",
        "joined_pair_sec_fn",
        "unique_idx",
        "composed_fn",
    ]

    new_dict = {}

    for k in c:
        new_dict[k] = []

    new_dict["first_pair_idx"] = [i for i in range(100, 20100, 2)]
    new_dict["second_pair_idx"] = [i for i in range(101, 20101, 2)]
    new_dict["joined_pair_first_fn"] = [get_cposed_fn(i) for i in new_dict["first_pair_idx"]]
    new_dict["joined_pair_sec_fn"] = [get_cposed_fn(i) for i in new_dict["second_pair_idx"]]
    new_dict["unique_idx"] = [i for i in range(0, 10000)]
    new_dict["composed_fn"] = ["-" for i in range(0, 10000)]

    entire_df = pd.DataFrame.from_dict(new_dict)

    available = [int(s) for s in joined_all_dfs.seed]
    available = np.array(available)

    # checking whether the seed has been synthesised
    entire_df["first_pair_idx_in_seed"] = [np.array(i) in available for i in entire_df["first_pair_idx"]]
    entire_df["second_pair_idx_in_seed"] = [np.array(i) in available for i in entire_df["second_pair_idx"]]
    entire_df = entire_df[(entire_df["first_pair_idx_in_seed"] == True) & (entire_df["second_pair_idx_in_seed"] == True)]

    # check that the composed images exist
    first_pair_idx_exists = [os.path.exists(i) for i in entire_df["joined_pair_first_fn"]]
    second_pair_idx_exists = [os.path.exists(i) for i in entire_df["joined_pair_sec_fn"]]

    # return(entire_df['joined_pair_first_fn'])#,second_pair_idx_exists,entire_df)

    entire_df = entire_df[(np.array(first_pair_idx_exists) == True) & (np.array(second_pair_idx_exists) == True)]

    print("n cases for binary compare")
    print(entire_df.shape)

    return entire_df


np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


RLHF_DIR = "/path/to/eg3d-rlhf-geometry/000_RLHF_AM"
RLHF_MODELS_DIR_SIMCLR = f"{RLHF_DIR}/models_simclr_pretrained"
IMAGE_SCALING = 0.5
RATINGS_fn = "ratings_01_08_2023.ods"
RWD_DATA_FORMATTED_fn = "rwd_training_data_single_dmap_extra_cases_02_08_2023_expanded.csv"
current_composition_name = "rlhf_meshes_ffhq512-128_const_noise_t1"
composed_dir = os.path.join("/path/to/eg3d-rlhf-geometry/000_RLHF_AM", "composed_for_binary_ranking")
current_composed_dir = os.path.join(composed_dir, current_composition_name)
pairs_for_ranking_df_fn = os.path.join(current_composed_dir, "pairs_to_be_ranked.csv")

os.makedirs(composed_dir, exist_ok=True)
os.makedirs(current_composed_dir, exist_ok=True)


@torch.jit.script
def rescale_dmaps(dmaps, rescale_size: int):  # rescaaaling 128x128 dempth map for input to resnet, ie 224x224
    dxm = dmaps.reshape(-1, 3, 128, 128)
    dxm = F.interpolate(dxm, size=(rescale_size, rescale_size), mode="bilinear")
    dxm = dxm.reshape(-1, 2, 3, rescale_size, rescale_size)
    return dxm


# train_dict['X_dmaps'].shape


def create_new_joined_binary_df(overwrite=True):
    joined_all_dfs = get_all_meshes_for_binary(select_only="rlhf_meshes_ffhq512-128_const_noise_t1")
    entire_df = make_binary_comparisons_07_08_2023(joined_all_dfs)

    if overwrite:
        entire_df.to_csv(pairs_for_ranking_df_fn, index=False)


def create_new_rankings_spec(pairs_for_ranking_df_fn):
    joined_pair = pd.read_csv(pairs_for_ranking_df_fn)  # read in saved df
    empty_idx = np.where(joined_pair["composed_fn"] == "-")[0]

    st = time.time()
    for k, idx in enumerate(empty_idx):
        joined_pair = pd.read_csv(pairs_for_ranking_df_fn)  # read in saved df
        current_id = joined_pair["unique_idx"].iloc[idx]
        fn1 = joined_pair["joined_pair_first_fn"].iloc[idx]
        fn2 = joined_pair["joined_pair_sec_fn"].iloc[idx]

        # ** this is where the composition happens, create new image **
        composed_name = compose_and_save(
            fn1=fn1,
            fn2=fn2,
            current_id=current_id,
            current_composed_dir=current_composed_dir,
            rescale_final=IMAGE_SCALING,
        )

        composed_fns = [c for c in joined_pair["composed_fn"].values]
        composed_fns[idx] = composed_name

        joined_pair["composed_fn"] = composed_fns

        if k % 20 == 0:
            et = time.time()
            tt = et - st
            n_remaining_images = len(empty_idx) - k
            estimated_finish = tt * n_remaining_images / 20
            eta_min = estimated_finish / 60

            # print(f'Estimated time to finish is {eta_min:.3f} minute')
            print(f"Composed {k} images in {tt:.3f} seconds\t idx {idx}\t finish in {eta_min:.3f} minute")
            # print(f'Current idx is {idx}')

            st = time.time()
            joined_pair.to_csv(pairs_for_ranking_df_fn, index=False)  # save out first

        joined_pair.to_csv(pairs_for_ranking_df_fn, index=False)  # save out first
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
