"""
Helpers for generating, formatting, and visualising ranking data.
These utilities cover composing mesh/rgb images, mapping seeds to paths,
and preparing ranking CSVs for downstream loaders.
"""

import copy
import glob
import itertools
import os

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import PIL
import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageOps

from core_modules.data import io_geometry_utils as io_utils


idx_for_grey = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
}


seed_letter_concordance = {
    "A": "seed1",
    "B": "seed2",
    "C": "seed3",
    "D": "seed4",
    "E": "seed5",
    "F": "seed6",
}


def add_alpha_channel_to_im_as_np_array(iic):
    iic = copy.deepcopy(iic)
    iic_tp = np.dstack((iic, np.zeros((iic.shape[0], iic.shape[1]))))
    iic_tp[:, :, 3] = 225
    iic_tpi = iic_tp.astype(np.uint8)
    return iic_tpi


def assemble_triple_rgb(seed, data_dir):
    return io_utils.assemble_triple_rgb(seed, data_dir).permute(1, 2, 0).numpy()


def combine_vertically(im1, im2):
    im1_height = im1.shape[0]
    im2_height = im2.shape[0]

    if im1_height > im2_height:
        im2 = im2[:im1_height, :, :]
    elif im2_height > im1_height:
        im1 = im1[:im2_height, :, :]

    combined_im = np.vstack([im1, im2])
    return combined_im


def compose_vertical_from_np(mesh_file, image_file):
    im1 = Image.fromarray(mesh_file)
    im2 = Image.fromarray(image_file)
    mesh_file_width, mesh_file_height = im1.size
    image_file_width, image_file_height = im2.size

    ratio_for_image_file_height = mesh_file_height / image_file_height
    im2 = im2.resize(
        (
            int(image_file_width * ratio_for_image_file_height),
            int(image_file_height * ratio_for_image_file_height),
        )
    )

    im3 = Image.new("RGB", (max(im1.width, im2.width), im1.height + im2.height))

    im3.paste(im1, (0, 0))
    starting_position = int(im3.width / 2 - im2.width / 2)
    im3.paste(im2, (starting_position, im1.height))
    return im3


def convert_seed_to_path(seed, composed_dir):
    if seed is None or str(seed).lower() == "none" or np.isnan(seed):
        return None
    seed = int(seed)
    return os.path.join(composed_dir, "composed_s_" + str(seed) + ".jpg")


def combine_image_stack(iic):
    res_4k = {
        "height": 2160,
        "width": 3840,
    }

    iic_height = iic.shape[0]
    iic_width = iic.shape[1]
    new_height = res_4k["height"]
    ratio = new_height / iic_height
    new_width = int(iic_width * ratio)

    new_width, new_height = get_new_dims(iic, res_4k["height"])
    iic = PIL.Image.fromarray(iic).resize((new_width, new_height))

    return iic


def combine_row(row):
    read_ims = read_in_row(row)
    read_ims = [put_red_border_around_img(ri) for ri in read_ims]
    list_of_seeds = ["A", "B", "C", "D", "E", "F", "G"]
    seed_names = [list_of_seeds[i] for i in range(len(read_ims))]
    read_ims = [text_on_image(ri, sn) for ri, sn in zip(read_ims, seed_names)]

    list_of_ims = [copy.deepcopy(read_ims[0]) for i in range(6)]
    for l in list_of_ims:
        l.fill(0)

    for i in range(len(read_ims)):
        list_of_ims[i] = read_ims[i]

    extr = extract_ranks(row)

    for letter in extr:
        idx = idx_for_grey[letter]
        list_of_ims[idx] = return_greyed_image(list_of_ims[idx])

    for k, i in enumerate(list_of_ims):
        if i.shape[2] != 4:
            i = add_alpha_channel_to_im_as_np_array(i)
            list_of_ims[k] = i

    combined_h1 = np.hstack(list_of_ims[:2])
    combined_h2 = np.hstack(list_of_ims[2:4])
    combined_h3 = np.hstack(list_of_ims[4:])

    combined_v = np.vstack([combined_h1, combined_h2, combined_h3])

    return combined_v


def read_in_row(row):
    read_ims = []
    for c in row.index:
        if c.startswith("seed"):
            if type(row[c]) == str and len(row[c]) > 0:
                im = mpimg.imread(row[c])
                read_ims.append(im)
            else:
                pass
    return read_ims


def extract_ranks(current_row):
    rank_cols = [c for c in current_row.index if "rank" in c]
    rankings_from_row = current_row[rank_cols]
    ranked_letters = np.array(rankings_from_row)[np.array([is_not_none(k) for k in rankings_from_row.values])]
    return ranked_letters


def return_greyed_image(image_as_np_array):
    iic = image_as_np_array
    myBackgroundImage = PIL.Image.fromarray(iic)
    iic = copy.deepcopy(iic)

    iic.fill(100)

    iic_tp = np.dstack((iic, np.zeros((iic.shape[0], iic.shape[1]))))
    iic_tp[:, :, 3] = 225
    iic_tpi = PIL.Image.fromarray(iic_tp.astype(np.uint8))

    myForegroundImage = iic_tpi
    myMerged_image = Image.new("RGBA", myBackgroundImage.size)
    myMerged_image.paste(myBackgroundImage, (0, 0))
    _, _, _, mask = myForegroundImage.split()
    myMerged_image.paste(myForegroundImage, (0, 0), mask)

    return np.asarray(myMerged_image)


def is_not_none(val):
    if type(val) == str and len(val) > 0:
        return True
    else:
        return False


def compose_meshes_for_ranking():
    composed_dir = os.path.join(data_dir, "composed_for_ranking")

    seeds = return_ffhq_seeds()

    ims = globdir(composed_dir, "composed_s_*.jpg")

    s_completed = [s.split("composed_s_")[-1].split(".jpg")[0] for s in ims]

    sc = [int(s) for s in s_completed]

    seeds_remain = [s for s in seeds if s not in sc]
    print("seeds remaining to synthesise")
    print(len(seeds_remain))

    for s in tqdm.tqdm(seeds):
        a = assemble_triple_rgb(s, data_dir)
        b = get_mesh_im(s, data_dir)
        c = compose_vertical_from_np(b, a)
        c.save(os.path.join(composed_dir, "composed_s_" + str(s) + ".jpg"))


def format_rankings_csv_for_seeds(data_dir, saving_file=False):
    rankings_dir = os.path.join(data_dir, "rankings_data")
    records_save_name = os.path.join(rankings_dir, "mesh_seed_choices.csv")
    path_seed_choices = pd.read_csv(records_save_name).set_index("index")
    composed_dir = os.path.join(data_dir, "composed_for_ranking")

    for c in path_seed_choices.columns:
        path_seed_choices[c] = path_seed_choices[c].apply(lambda seed: convert_seed_to_path(seed, composed_dir))

    path_seed_choices_records_name = os.path.join(rankings_dir, "path_seed_choices.csv")
    path_seed_choices.to_csv(path_seed_choices_records_name)
    pdr = pd.read_csv(path_seed_choices_records_name).set_index("index")
    pdr.head()

    pdr_rankings = pdr.copy()

    pdr_rankings["rank1"] = None
    pdr_rankings["rank2"] = None
    pdr_rankings["rank3"] = None
    pdr_rankings["rank4"] = None
    pdr_rankings["rank5"] = None
    pdr_rankings["rank6"] = None
    pdr_rankings["rank7"] = None
    pdr_rankings["completed"] = False

    pdr_rankings["n_in_row"] = None
    n_in_each_row = []

    for i in pdr_rankings.index:
        rowvals = pdr_rankings.loc[i].values
        n_in_row = sum([is_not_none(k) for k in rowvals])
        pdr_rankings.loc[i, "n_in_row"] = n_in_row

    n_in_each_row = pdr_rankings.n_in_row.values
    min_n_in_each_row = min(n_in_each_row)
    print(min_n_in_each_row)
    max_n_in_each_row = max(n_in_each_row)
    print(max_n_in_each_row)

    rankings_records_name = os.path.join(rankings_dir, "rankings_records.csv")

    if not os.path.isfile(rankings_records_name) and saving_file:
        pdr_rankings.to_csv(rankings_records_name)


def list_all_files_in_dir_and_create_composed(data_dir):
    all_in_dir = os.listdir(data_dir)
    all_in_dir = [a for a in all_in_dir if a.endswith(".jpg")]
    print("n files in current data dir")
    print(len(all_in_dir))

    composed_dir = os.path.join(data_dir, "composed_for_ranking")
    os.makedirs(composed_dir, exist_ok=True)


def join_all_ims_together(pdr_rankings, data_dir):
    pdr_rankings_c = pdr_rankings.copy()
    os.makedirs(os.path.join(data_dir, "rankings", "joined_ims"), exist_ok=True)

    for i in tqdm.tqdm(pdr_rankings_c.index):
        current_im = combine_row(pdr_rankings_c.loc[i])
        out_fn = os.path.join(data_dir, "rankings", "joined_ims", "joined_idx_" + str(i) + ".jpg")
        current_im = PIL.Image.fromarray(current_im).convert("RGB")
        current_im.save(out_fn)


def view_first_example_of_joined_im(data_dir):
    tims = glob.glob(os.path.join(data_dir, "rankings", "joined_ims", "joined_idx_*.jpg"))
    print(tims[0])
    tt = PIL.Image.open(tims[0])
    tt.show()


def get_n_combinations(n):
    retval = len(list(itertools.combinations(range(n), 2)))
    return retval


def get_row_ranks(index, ranked_c):
    row_rank = ranked_c.loc[index][["rank1", "rank2", "rank3", "rank4", "rank5", "rank6", "rank7"]]
    n_in_row = ranked_c.loc[index].n_in_row.item()
    return dict(n_in_row=n_in_row, row_rank=row_rank)


def get_just_ranks(df_row):
    row_rank = df_row[["rank1", "rank2", "rank3", "rank4", "rank5", "rank6", "rank7"]]
    return row_rank


def get_mesh_im(seed, data_dir):
    mesh_fn = os.path.join(data_dir, "mesh_cat_s_" + str(seed) + ".jpg")
    read_im = mpimg.imread(mesh_fn)
    return read_im


def get_new_dims(im, new_height):
    iic_height = im.shape[0]
    iic_width = im.shape[1]
    ratio = new_height / iic_height
    new_width = int(iic_width * ratio)
    return (new_width, new_height)


def globdir(in_path, extension):
    search_str = os.path.join(in_path, extension)
    return glob.glob(search_str)


def put_red_border_around_img(current_img_np_array):
    current_img = PIL.Image.fromarray(current_img_np_array)
    img_with_border = ImageOps.expand(current_img, border=15, fill="red")
    img_with_border = np.asarray(img_with_border)
    return img_with_border


def text_on_image(img, text, fontsize=100):
    img = Image.fromarray(img)
    im_width, im_height = img.size
    font_location = (int(4 * im_width / 5) + 100, int(im_height - 160))
    I1 = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf", fontsize)
    I1.text(font_location, text, fill=(255, 0, 0), font=font)
    img = np.asarray(img)
    return img


def get_seed_from_composed(in_str):
    retval = in_str.split("/")[-1].split("_s_")[-1].split(".")[0]
    return retval


def insert_seed_to_rankings(row_for_clone):
    row_of_it_all = row_for_clone.copy()
    n_in_row = row_of_it_all.n_in_row
    ranks = get_just_ranks(row_of_it_all)

    for idx in range(n_in_row):
        current_rank = ranks[f"rank{idx + 1}"]
        current_seed_rank = seed_letter_concordance[current_rank]
        current_np_seed = get_seed_from_composed(row_of_it_all[current_seed_rank])
        row_of_it_all[f"rank{idx + 1}"] = current_np_seed

    return row_of_it_all


def dataframe_to_ranked_seeds(ranked_c_df):
    ranked_c_for_seeds = ranked_c_df.copy()
    ranked_c_df = ranked_c_df.copy()
    if ranked_c_df.shape[1] == 2:
        ranked_c_df.columns = ["seed1", "seed2"]
        return ranked_c_df[["seed1", "seed2"]]

    for idx in tqdm.tqdm(ranked_c_for_seeds.index):
        ranked_c_for_seeds.loc[idx] = insert_seed_to_rankings(ranked_c_for_seeds.loc[idx])

    ranked_seeds_for_dataloader = ranked_c_for_seeds[["rank1", "rank2", "rank3", "rank4", "rank5", "rank6", "rank7"]]
    return ranked_seeds_for_dataloader


def get_list_of_rankings_minimal_rows(ranked_seeds_for_dataloader):
    list_of_rankings = []
    for idx in tqdm.tqdm(ranked_seeds_for_dataloader.index):
        list_of_seeds = ranked_seeds_for_dataloader.loc[idx].fillna(-1).astype(np.int32).values
        ordered_seeds = np.array(list_of_seeds)
        list_of_rankings.append(ordered_seeds)

    assert len(list_of_rankings) == ranked_seeds_for_dataloader.shape[0], "error list of rankigns not same as n row in ranked seeds"
    return np.array(list_of_rankings)


def create_list_of_rankings_minimal(ranked_c_df):
    ranked_seeds_for_dataloader = dataframe_to_ranked_seeds(ranked_c_df)
    list_of_rankings = get_list_of_rankings_minimal_rows(ranked_seeds_for_dataloader)
    return list_of_rankings


def check_row(idx_of, ranked_c):
    possible_rank_letters = np.array(["A", "B", "C", "D", "E", "F"])
    rr = get_row_ranks(idx_of, ranked_c)
    vv = rr["row_rank"].values.astype("str")
    vv = vv[vv != "nan"]
    all_in_row = len(vv) == rr["n_in_row"]
    ranks = rr["row_rank"]
    necessary_rank_letters = possible_rank_letters[: rr["n_in_row"]]
    ranked_cases = ranks[: rr["n_in_row"]]
    length_and_unique_check = set(necessary_rank_letters.tolist()) == set(ranked_cases.tolist())
    if all_in_row and length_and_unique_check:
        return True
    else:
        return False
