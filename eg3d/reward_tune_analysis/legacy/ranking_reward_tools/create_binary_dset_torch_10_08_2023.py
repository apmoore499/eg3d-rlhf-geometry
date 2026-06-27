# to make dataset so that jupyter dont freeze


import copy
import glob
import shutil
import time

import numpy as np
import pandas as pd
from ranking_utils import *
from sklearn.model_selection import train_test_split

ranking_idx = 0


def get_composed_fn(i):
    return f"/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/000_RLHF_AM/composed_for_binary_ranking/rlhf_meshes_ffhq512-128_const_noise_t1/composed_{i}.jpg"


def copy_im(current_im, ranking_idx):
    new_folder_name = f"/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/000_RLHF_AM/composed_for_binary_ranking/rlhf_meshes_ffhq512-128_const_noise_t1/binary_ranking_{ranking_idx}"
    try:
        shutil.copy(current_im, new_folder_name)
    except:
        pass


def batch_tensor(in_tensor):
    if in_tensor.shape[0] != 1:
        in_tensor = in_tensor.unsqueeze(0)
    return in_tensor


def copy_code(ranking_idx):
    new_folder_name = f"/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/000_RLHF_AM/composed_for_binary_ranking/rlhf_meshes_ffhq512-128_const_noise_t1/binary_ranking_{ranking_idx}"
    all_to_copy = glob.glob(boilerplate_dir + "*")
    for fn in all_to_copy:
        shutil.copy(fn, new_folder_name)


def get_seed_from_jpg(fn):
    return int(fn.split("/")[-1].split("_")[1].split(".")[0])


def init_rankings_csv(ranking_idx):
    new_folder_name = f"/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/000_RLHF_AM/composed_for_binary_ranking/rlhf_meshes_ffhq512-128_const_noise_t1/binary_ranking_{ranking_idx}"

    all_in_folder = glob.glob(new_folder_name + "/*.jpg")

    all_in_folder = [p for p in all_in_folder if "completed_page.jpg" not in p]

    folder_seeds = [get_seed_from_jpg(fn) for fn in all_in_folder]

    orig_spec = pd.read_csv(pairs_for_ranking_df_fn)

    print(len(all_in_folder))

    new_spec = orig_spec.set_index("unique_idx").loc[folder_seeds]

    new_spec["composed_fn"] = all_in_folder

    new_spec.head()

    new_spec = new_spec.reset_index()
    modified_seeds = [get_seed_from_jpg(fn) for fn in new_spec["composed_fn"].values]

    print("does new seed idx and path name match")
    import numpy as np

    match_idx_pn = np.all(modified_seeds == new_spec["unique_idx"].values)

    print(match_idx_pn)

    if match_idx_pn:
        new_spec.to_csv(f"{new_folder_name}/pairs_to_be_ranked.csv", index=False)

        print("idx matched for ranking idx ", ranking_idx)
    else:
        print("idx not matched, csv not created, for ranking idx ", ranking_idx)


# get all the ranigns...
def get_ranking_folder(r):
    return f"/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/000_RLHF_AM/composed_for_binary_ranking/rlhf_meshes_ffhq512-128_const_noise_t1/binary_ranking_{r}/"


def get_rankings_csv(r):
    folder = get_ranking_folder(r)
    rfn = "results_of_ranking.csv"
    rfn = os.path.join(folder, rfn)
    if not os.path.exists(rfn):
        pass
    else:
        return pd.read_csv(rfn)


def get_expname_from_path(path):
    return path.split("/")[-2].replace("rlhf_meshes_", "")


def rescale_dmap_single(dmap, rescale_size, mode="nearest", aa_mode=True):
    dxm = dmap  # .reshape(-1,1,128,128)
    dmp = F.interpolate(dxm.unsqueeze(0), size=(rescale_size, rescale_size), mode=mode, antialias=aa_mode)
    # dxm=dxm.reshape(-1,1,rescale_size,rescale_size).cuda()

    dmp = dmp.squeeze(0)
    return dmp


def normalise_dmap_vals_ffhq(dmap, start=2.25, end=3.3, min=0.0, check_lims=True):
    rmin = start
    rmax = end
    if min == 0.0:
        dmap = (dmap - rmin) / (rmax - rmin)
        dm_min = 0.0
    elif min == -1.0:
        dmap = (((dmap - rmin) / (rmax - rmin)) * 2) - 1
        dm_min = -1.0

    dmap[dmap < dm_min] = dm_min
    dmap[dmap > 1.0] = 1.0
    # dmap.clamp_(min=dm_min,max=1.0) #clamp between vals

    if check_lims:
        assert dmap.max() <= 1.0
        assert dmap.min() >= dm_min
    return dmap


def mask_depth_map(dmap_good):
    dmaps = dmap_good.squeeze(0)

    ups = [rescale_dmap_single(dmap=d, rescale_size=224, mode="bilinear", aa_mode=True) for d in dmaps]

    dmi = [d.unsqueeze(0) for d in ups]

    dmaps = torch.cat(dmi, 0)
    dmaps[0][dmaps[0] > 2.9] = 3.3
    dmaps[2][dmaps[2] > 2.9] = 3.3
    dmaps[1][dmaps[1] > 2.7] = 3.3

    # dmaps=[normalise_dmap_vals_ffhq(d).unsqueeze(0) for d in dmaps]

    # dmaps=torch.cat(dmaps,0).unsqueeze(0)

    return dmaps.unsqueeze(0)


def create_instance_pair_binary_from_pt(
    input_pair_win,
    dict_of_depth_pt,
    dict_of_stylecodes_pt=None,
    dict_of_lmks_pt=None,
    dict_of_lmks_pt_triple=None,
    rendering_options=None,
    normalise=False,
    norm_min=None,
):
    input_pair = input_pair_win[0]
    idx1 = input_pair[0]
    idx2 = input_pair[1]
    winning_idx = input_pair_win[1]

    dmap1_fn = dict_of_depth_pt[idx1]
    dmap2_fn = dict_of_depth_pt[idx2]
    dmap1 = torch.load(dmap1_fn, map_location=torch.device("cuda"))  # .squeeze(2)
    dmap2 = torch.load(dmap2_fn, map_location=torch.device("cuda"))  # .squeeze(2)

    dmap1 = mask_depth_map(dmap1).half()
    dmap2 = mask_depth_map(dmap2).half()

    rescale_size = 224
    # dmap1=F.interpolate(dmap1,size=(rescale_size,rescale_size),mode='bilinear').unsqueeze(2).half()
    # dmap2=F.interpolate(dmap2,size=(rescale_size,rescale_size),mode='bilinear').unsqueeze(2).half()

    # stylecode1_fn=dict_of_stylecodes_pt[idx1]
    # stylecode2_fn=dict_of_stylecodes_pt[idx2]

    # stylecode1=torch.load(stylecode1_fn)
    # stylecode2=torch.load(stylecode2_fn)

    # lmks_2d_1_fn=dict_of_lmks_pt[idx1]
    # lmks_2d_2_fn=dict_of_lmks_pt[idx2]

    # lmks_2d_1=torch.load(lmks_2d_1_fn)#.view(1,2,98)
    # lmks_2d_2=torch.load(lmks_2d_2_fn)#.view(1,2,98)

    # lmks_2d_triple_1_fn=dict_of_lmks_pt_triple[idx1]
    # lmks_2d_triple_2_fn=dict_of_lmks_pt_triple[idx2]

    # lmks_2d_triple_1=torch.load(lmks_2d_triple_1_fn)
    # lmks_2d_triple_2=torch.load(lmks_2d_triple_2_fn)

    # lmks_2d_triple_1=batch_tensor(lmks_2d_triple_1)
    # lmks_2d_triple_2=batch_tensor(lmks_2d_triple_2)

    unique_id_1 = torch.tensor(idx1).unsqueeze(0)
    unique_id_2 = torch.tensor(idx2).unsqueeze(0)

    assert unique_id_1 != unique_id_2
    assert unique_id_1 == idx1
    assert unique_id_2 == idx2

    first_combination = dmap1  # , unique_id_1,stylecode1,lmks_2d_1,lmks_2d_triple_1)
    second_combination = dmap2  # , unique_id_2,stylecode2,lmks_2d_2,lmks_2d_triple_2)
    combo_list = [first_combination, second_combination]

    high_score_idx = np.where(np.array(input_pair) == winning_idx)[0][0]
    low_score_idx = np.where(np.array(input_pair) != winning_idx)[0][0]
    high_comb = combo_list[high_score_idx]
    low_comb = combo_list[low_score_idx]
    combo = (high_comb, low_comb)
    high_dmap = combo[0][0].squeeze().unsqueeze(0)
    low_dmap = combo[1][0].squeeze().unsqueeze(0)

    if normalise:
        low_dmap = normalise_dmap_vals(low_dmap, rendering_options, min=norm_min)
        high_dmap = normalise_dmap_vals(high_dmap, rendering_options, min=norm_min)

    depths = [high_dmap, low_dmap]
    scores = [1000, -1000]
    scores = [torch.as_tensor(s).unsqueeze(0) for s in scores]
    # unique_ids=[combo[0][1],combo[1][1]]
    # stylecodes=[combo[0][2],combo[1][2]]
    # lmks_2d=[combo[0][3],combo[1][3]]
    # lmks_2d_triple=[combo[0][4],combo[1][4]]
    return dict(depths=depths, scores=scores)  # ,stylecodes=stylecodes,lmks_2d=lmks_2d,lmks_2d_triple=lmks_2d_triple))


def create_dset_dict_binary_from_pt(
    rankings_df,
    dict_of_depth_pt,
    dict_of_stylecodes_pt,
    dict_of_lmks_pt,
    dict_of_lmks_pt_triple,
    normalise=False,
    norm_min=None,
):
    pairs_win_binary = get_pairs_and_win_idx_binary(rankings_df)

    combos = []
    for p in tqdm(pairs_win_binary):
        combo = create_instance_pair_binary_from_pt(
            input_pair_win=p,
            dict_of_depth_pt=dict_of_depth_pt,
            dict_of_stylecodes_pt=dict_of_stylecodes_pt,
            dict_of_lmks_pt=dict_of_lmks_pt,
            dict_of_lmks_pt_triple=dict_of_lmks_pt_triple,
            rendering_options=ffhq_rendering_options,
            normalise=normalise,
            norm_min=norm_min,
        )

        combos.append(combo)

    # combos=[create_instance_pair_binary_from_pt(input_pair_win=p,
    #                                             dict_of_depth_pt=dict_of_depth_pt,
    #                                             dict_of_stylecodes_pt=dict_of_stylecodes_pt,
    #                                             dict_of_lmks_pt=dict_of_lmks_pt,
    #                                             dict_of_lmks_pt_triple=dict_of_lmks_pt_triple,
    #                                             rendering_options=ffhq_rendering_options,
    #                                             normalise=normalise,norm_min=norm_min) for p in pairs_win_binary]

    depth_maps = [torch.cat(dd, dim=0).unsqueeze(0) for dd in [c["depths"] for c in list(combos)]]
    X_dmaps = torch.cat(depth_maps, dim=0)

    scores = [torch.cat(dd, dim=0).unsqueeze(0) for dd in [c["scores"] for c in list(combos)]]
    yall = torch.cat(scores, dim=0)

    # ids=[torch.cat(dd,dim=0).unsqueeze(0) for dd in [c['unique_ids'] for c in list(combos)]]
    # ids=torch.cat(ids,dim=0)

    # stylecodes=[torch.cat(dd,dim=0).unsqueeze(0) for dd in [c['stylecodes'] for c in list(combos)]]
    # stylecodes=torch.cat(stylecodes,dim=0)

    # dd = [c['lmks_2d'] for c in list(combos)]

    # lmks_2d=[]

    # for i,d in enumerate(dd):
    #     print(i)
    #     print(ids[i])
    #     print(d)
    #     print(d[0].shape)
    #     print(d[1].shape)

    #     if d[0].shape!=(1,2,98):
    #         d[0]=d[0].view(1,2,98)

    #     if d[1].shape!=(1,2,98):
    #         d[1]=d[1].view(1,2,98)

    #     lmks_2d.append(torch.cat(d,dim=0).unsqueeze(0))

    # #lmks_2d=[torch.cat(dd,dim=0).unsqueeze(0) for dd in [c['lmks_2d'] for c in list(combos)]]
    # lmks_2d=torch.cat(lmks_2d,dim=0)

    # dd = [c['lmks_2d_triple'] for c in list(combos)]

    # lmks_2d_triple=[]

    # for i,d in enumerate(dd):
    #     print(i)
    #     print(ids[i])
    #     print(d)
    #     print(d[0].shape)
    #     print(d[1].shape)

    #     if d[0].shape!=(1,3,2,98):
    #         d[0]=d[0].view(1,3,2,98)

    #     if d[1].shape!=(1,3,2,98):
    #         d[1]=d[1].view(1,3,2,98)

    #     lmks_2d_triple.append(torch.cat(d,dim=0).unsqueeze(0))

    # #lmks_2d_triple=[torch.cat(dd,dim=0).unsqueeze(0) for dd in [c['lmks_2d_triple'] for c in list(combos)]]
    # lmks_2d_triple=torch.cat(lmks_2d_triple,dim=0)

    return dict(X_dmaps=X_dmaps, yall=yall)  # ,stylecodes=stylecodes,lmks_2d=lmks_2d,lmks_2d_triple=lmks_2d_triple))


# def make_dset_naked(X_dmaps,yall,ids,nrs,stylecodes,lmks_2d,lmks_2d_triple,batch_size=None):
def make_dset_naked(X_dmaps, yall, nrs, batch_size=None):
    if nrs == 64:
        X_dmaps = torch.nn.functional.interpolate(X_dmaps, size=[3, 64, 64], mode="nearest")

    # dataset = TensorDataset( Tensor(X_dmaps).to(device).float(),
    #                         Tensor(yall.float()).to(device).float(),
    #                         Tensor(ids).to(device).int(),
    #                         Tensor(stylecodes).to(device).float(),
    #                         Tensor(lmks_2d).to(device).float(),
    #                         Tensor(lmks_2d_triple).to(device).float())

    dataset = TensorDataset(Tensor(X_dmaps), Tensor(yall.float()))
    # Tensor(ids).to(device).int(),
    # Tensor(stylecodes).to(device).float(),
    # Tensor(lmks_2d).to(device).float(),
    # Tensor(lmks_2d_triple).to(device).float())

    if batch_size is None:
        batch_size = len(dataset)

    if batch_size > len(dataset):
        batch_size = len(dataset)
    print("len dataset")
    print(len(dataset))
    print("----------")
    dloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dloader


def set_winning_idx(rankings_df):
    rankings_df["winning_idx"] = 0
    rankings_df.loc[rankings_df.rankings == "A", "winning_idx"] = rankings_df.loc[rankings_df.rankings == "A", "first_pair_idx"]
    rankings_df.loc[rankings_df.rankings == "B", "winning_idx"] = rankings_df.loc[rankings_df.rankings == "B", "second_pair_idx"]
    return rankings_df


def format_rankings(names=None):
    if names is not None and type(names) is not list:
        names = [names]

    joined_all_dfs = pd.read_csv(pairs_for_ranking_df_fn)
    joined_all_dfs.head()
    joined_all_dfs = get_all_meshes_for_binary(select_only=names)
    joined_all_dfs["seed"] = joined_all_dfs["seed"].astype(int)
    joined_all_dfs.set_index("seed", inplace=True)
    print(f" sanity check for index: 9424")
    joined_all_dfs["three_dmap"][9424]
    print("shape all examples")
    print(joined_all_dfs.shape)
    rdfs = [get_rankings_csv(r) for r in range(0, 40)]
    rdfs_joined = pd.concat(rdfs)
    # filter out rankings not '-'
    rdfs_joined = rdfs_joined[rdfs_joined["rankings"] != "-"]

    keep_cols = [
        "first_pair_idx",
        "second_pair_idx",
        "joined_pair_first_fn",
        "joined_pair_sec_fn",
        "unique_idx",
        "composed_fn",
        "rankings",
    ]

    rdf = rdfs_joined[keep_cols]
    print("shape of total rdf concat")
    print(rdf.shape)

    all_rankings_df = rdf
    all_rankings_df.reset_index(drop=True, inplace=True)
    all_rankings_df.head()
    print("shape of all rankings now merged (paired)")
    print(all_rankings_df.shape)
    print(joined_all_dfs.shape)

    expname = get_expname_from_path(joined_all_dfs["single_dmap"].iloc[0])
    print("for expname:")
    print(expname)

    # all_rankings_df=all_rankings_df[all_rankings_df['rankings']!='C']

    print(all_rankings_df.shape)

    print("summary fo rankings")

    print(all_rankings_df.groupby("rankings").count() / all_rankings_df.shape[0])

    print("subset to ranked pairs only, exclude C")
    all_rankings_df = all_rankings_df[all_rankings_df["rankings"] != "C"]

    all_idx = [all_rankings_df.first_pair_idx.values] + [all_rankings_df.second_pair_idx.values]

    keep_idx = np.unique(all_idx)

    print("max idx")
    print(max(keep_idx))

    print("len idx (ie n individual dmaps)")
    print(len(keep_idx))
    print("min idx")
    print(min(keep_idx))

    # all_rankings_df[all_rankings_df.second_pair_idx.isin(keep_idx)]

    joined_all_dfs["pt_name_three_dmap"] = joined_all_dfs["three_dmap"].str.replace(".json", ".pt")
    joined_all_dfs["pt_three_dmap_exists"] = joined_all_dfs["pt_name_three_dmap"].apply(lambda x: os.path.exists(x))

    joined_all_dfs["w_code"] = joined_all_dfs["pt_name_three_dmap"].apply(lambda x: x.replace("_three_dmaps", "_style_code"))

    # joined_all_dfs.groupby(['pt_three_dmap_exists']).size()
    # joined_all_dfs[joined_all_dfs.index.isin(keep_idx)].shape

    # convert any dmap to .pt files if not

    new_idx = joined_all_dfs[joined_all_dfs["pt_three_dmap_exists"] == False].index.values

    print(f" number of new idx to export: {len(new_idx)}")

    print("exporting .pt files...")
    list_of_depths = {}

    # for k,i in enumerate(new_idx):
    # assume already converted AM 29_09_2023, change so that we can run 3dmm semi sup on the fly

    #     print(i)

    #     print(joined_all_dfs['three_dmap'].loc[i])

    #     current_dmap=read_dmap_three_to_tensor(joined_all_dfs['three_dmap'].loc[i])
    #     new_pt_name=joined_all_dfs['three_dmap'].loc[i].replace('.json','.pt')
    #     #new_pt_name
    #     torch.save(current_dmap,new_pt_name)
    #     #list_of_depths[i]=read_dmap_three_to_tensor(joined_all_dfs['three_dmap'][i])
    #     if k%100==0:
    #         print(f'finished {k} of {len(new_idx)}')

    dict_of_depth_pt = joined_all_dfs["pt_name_three_dmap"].to_dict()

    print("set rankings_df_sub to be all_rankings_df")
    rankings_df_sub = all_rankings_df

    return (rankings_df_sub, dict_of_depth_pt)


def convert_df_to_pt_dataset(
    df,
    dict_of_depth_pt,
    dict_of_stylecodes_pt,
    dict_of_lmks_pt,
    dict_of_lmks_pt_triple,
    normalise=False,
    norm_min=None,
    rescale_size=None,
    nrs=128,
    batch_size=64,
    savename="name_me.pt",
):
    if savename == "name_me.pt":
        print("error you must specify name for dataset, exiting")
        return None

    dset_dict = create_dset_dict_binary_from_pt(
        df,
        dict_of_depth_pt=dict_of_depth_pt,
        dict_of_stylecodes_pt=dict_of_stylecodes_pt,
        dict_of_lmks_pt=dict_of_lmks_pt,
        dict_of_lmks_pt_triple=dict_of_lmks_pt_triple,
        normalise=normalise,
        norm_min=norm_min,
    )

    # if rescale_size is not None:
    #    dset_dict['X_dmaps']=rescale_dmaps(dset_dict['X_dmaps'],rescale_size=rescale_size)

    dset_kwargs = {
        "X_dmaps": dset_dict["X_dmaps"],
        "yall": dset_dict["yall"],
        "nrs": nrs,
        "batch_size": batch_size,
    }

    dloader = make_dset_naked(**dset_kwargs)

    torch.save(dloader, savename)

    # return(dset_dict)


def get_all_idx(sdf):
    all_rank_idx = [sdf.first_pair_idx.values] + [sdf.second_pair_idx.values]

    ari = np.array(all_rank_idx)

    ari = ari.flatten()
    return ari


def split_partitions(names_of_pt, rankings_df_sub, dict_of_depth_pt, dset_kwargs):
    n_examples = rankings_df_sub.shape[0]
    props = dset_kwargs["props"]

    # vtscale=0.6
    # train_val, test = train_test_split(rankings_df_sub, test_size=int(250*n_examples/2000*vtscale), random_state=42)
    # train, val = train_test_split(train_val, test_size=int(250*n_examples/2000*vtscale), random_state=42)
    test_split = props[1] + props[2]

    val_test_prop = props[2] / (test_split)

    train, test_val = train_test_split(rankings_df_sub, test_size=test_split, random_state=42)
    val, test = train_test_split(test_val, test_size=val_test_prop, random_state=42)

    test.reset_index(drop=True, inplace=True)
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)

    train["winning_idx"] = -1
    test["winning_idx"] = -1
    val["winning_idx"] = -1

    train = set_winning_idx(train)
    test = set_winning_idx(test)
    val = set_winning_idx(val)

    return (train, test, val)


def split_out_and_save(names_of_pt, rankings_df_sub, dict_of_depth_pt, m_psi_partitions=None, dset_kwargs=None):
    train, test, val = split_partitions(names_of_pt, rankings_df_sub, dict_of_depth_pt, dset_kwargs=dset_kwargs)

    if m_psi_partitions is not None:
        train, tr_dict = append_to_rankings(
            train,
            m_psi_partitions["train"],
            n_good_mesh_to_use="all",
            n_for_each_psi=m_psi_partitions["n_for_each_psi"],
        )
        val, va_dict = append_to_rankings(
            val,
            m_psi_partitions["val"],
            n_good_mesh_to_use="all",
            n_for_each_psi=m_psi_partitions["n_for_each_psi"],
        )
        test, te_dict = append_to_rankings(
            test,
            m_psi_partitions["test"],
            n_good_mesh_to_use="all",
            n_for_each_psi=m_psi_partitions["n_for_each_psi"],
        )
        dict_of_depth_pt.update(tr_dict)
        dict_of_depth_pt.update(va_dict)
        dict_of_depth_pt.update(te_dict)

    using_3dmm_gt = True

    if using_3dmm_gt:
        mm_partitions_df = [
            999998,
            "/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/000_RLHF_AM/rlhf_meshes/3dmm_mean_shape/999998_composed.jpg",
            "/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/000_RLHF_AM/rlhf_meshes/3dmm_mean_shape/seed999998.json",
            "/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/000_RLHF_AM/rlhf_meshes/3dmm_mean_shape/seed999998_three_dmaps.json",
        ]

        mm_partitions_df = pd.DataFrame(mm_partitions_df).transpose()

        mm_partitions_df.columns = m_psi_partitions["train"].columns

        mm_partitions_df.set_index("seed", inplace=True)

        mm_partitions_df["seed"] = ["999998"]

        train, tr_dict = append_to_rankings(train, mm_partitions_df, n_good_mesh_to_use="all", n_for_each_psi=train.shape[0])
        val, va_dict = append_to_rankings(val, mm_partitions_df, n_good_mesh_to_use="all", n_for_each_psi=val.shape[0])
        test, te_dict = append_to_rankings(test, mm_partitions_df, n_good_mesh_to_use="all", n_for_each_psi=test.shape[0])
        dict_of_depth_pt.update(tr_dict)
        dict_of_depth_pt.update(va_dict)
        dict_of_depth_pt.update(te_dict)

    norm_dmaps = True
    norm_min = -1.0
    rescale_size = 224

    dict_of_stylecodes_pt = {}

    for k in dict_of_depth_pt.keys():
        dict_of_stylecodes_pt[k] = dict_of_depth_pt[k].replace("_three_dmaps", "_style_code")

    dict_of_lmks_pt = {}
    # etc AAAAAM 14:44
    for k in dict_of_depth_pt.keys():
        dict_of_lmks_pt[k] = dict_of_depth_pt[k].replace("_three_dmaps", "_kpts_awloss_98").replace("seed", "seed_")

    dict_of_lmks_pt_triple = {}
    # etc AAAAAM 14:44
    for k in dict_of_depth_pt.keys():
        dict_of_lmks_pt_triple[k] = dict_of_depth_pt[k].replace("_three_dmaps", "awloss_lmks_3_ims_98")

    conv_cols = ["first_pair_idx", "second_pair_idx", "unique_idx", "winning_idx"]

    dfs = [train, val, test]

    for df in dfs:
        for conv_col in conv_cols:
            df[conv_col] = df[conv_col].astype(int)

    default_kwargs = {
        "normalise": True,
        "dict_of_depth_pt": dict_of_depth_pt,
        "dict_of_stylecodes_pt": dict_of_stylecodes_pt,
        "dict_of_lmks_pt": dict_of_lmks_pt,
        "dict_of_lmks_pt_triple": dict_of_lmks_pt_triple,
        "norm_min": -1.0,
        "rescale_size": 224,
        "nrs": 128,
    }

    tkwargs = copy.deepcopy(default_kwargs)

    print("creating val:")
    st = time.time()

    tkwargs["df"] = val
    tkwargs["savename"] = names_of_pt["val"]
    tkwargs["batch_size"] = dset_kwargs["batch_size"]

    convert_df_to_pt_dataset(**tkwargs)

    ft = time.time()

    print("time taken: ", ft - st)
    st = time.time()

    tkwargs["df"] = test
    tkwargs["savename"] = names_of_pt["test"]
    tkwargs["batch_size"] = dset_kwargs["batch_size"]

    convert_df_to_pt_dataset(**tkwargs)
    ft = time.time()

    print("time taken: ", ft - st)

    print("creating train:")
    st = time.time()

    tkwargs["df"] = train
    tkwargs["savename"] = names_of_pt["train"]
    tkwargs["batch_size"] = dset_kwargs["batch_size"]

    convert_df_to_pt_dataset(**tkwargs)

    ft = time.time()

    print("time taken: ", ft - st)


def append_to_rankings(rankings_df_sub, m_psi_best, n_good_mesh_to_use, n_for_each_psi):
    print(n_good_mesh_to_use)
    # put in a new ranking
    if n_good_mesh_to_use == "all":
        n_good_mesh_to_use = min(2000, len(m_psi_best.index))
        # m_psi_best=m_psi_best

    # assert n_good_mesh_to_use>0, 'error need at least positive n or all for n_good_mesh_to_use'

    print(n_good_mesh_to_use)

    # randomly sample n_good_mesh_to_use rows from m_psi_best
    m_psi_best_sample = m_psi_best.sample(n=n_good_mesh_to_use)
    new_rankings_list = []

    m_psi_best_sample["pt_name_three_dmap"] = ""

    for k, i in enumerate(m_psi_best_sample.index):
        print(m_psi_best_sample["three_dmap"].loc[i])
        # current_dmap=read_dmap_three_to_tensor(m_psi_best_sample['three_dmap'].loc[i]) #all json already convert to pt, skip thise
        new_pt_name = m_psi_best_sample["three_dmap"][i].replace(".json", ".pt")
        # new_pt_name
        # torch.save(current_dmap,new_pt_name)

        m_psi_best_sample["pt_name_three_dmap"].loc[i] = new_pt_name
        # list_of_depths[i]=read_dmap_three_to_tensor(joined_all_dfs['three_dmap'][i])
        if k % 100 == 0:
            print(f"finished {k} of {len(m_psi_best_sample.index)}")

    dict_of_depth_pt_msi = m_psi_best_sample["pt_name_three_dmap"].to_dict()

    for i in m_psi_best_sample.index:
        sel_m_psi = m_psi_best_sample.loc[i]

        rankings_df = copy.deepcopy(rankings_df_sub)

        rdf_s = rankings_df.sample(n_for_each_psi)
        idx_from = np.unique(get_all_idx(rdf_s))
        np.sort(idx_from)
        sel_idx = random.sample(idx_from.tolist(), n_for_each_psi)

        rdfe = rdf_s[["first_pair_idx", "second_pair_idx", "joined_pair_first_fn", "joined_pair_sec_fn"]]
        rdfe.columns = ["fi", "si", "fp_fn", "sp_fn"]
        rdf_1 = rdfe[["fi", "fp_fn"]]
        rdf_2 = rdfe[["si", "sp_fn"]]
        rdf_1.columns = ["seed", "fn"]
        rdf_2.columns = ["seed", "fn"]

        rdf_vals = pd.concat([rdf_1, rdf_2], axis=0, ignore_index=True)

        rdf_vals.seed = rdf_vals.seed.astype(int)
        rdf_vals.index = rdf_vals.seed
        rdf_vals = rdf_vals.drop(columns=["seed"]).reset_index().drop_duplicates(subset="seed", keep="first").set_index("seed")

        rdf_s.reset_index(drop=True, inplace=True)

        rdf_s["second_pair_idx"] = sel_idx

        sec_rdf = rdf_vals.loc[sel_idx].fn.values

        rdf_s["joined_pair_sec_fn"] = sec_rdf
        rdf_s["first_pair_idx"] = int(sel_m_psi.seed)
        rdf_s["joined_pair_sec_fn"] = sel_m_psi.three_dmap
        rdf_s.composed_fn = "-----"
        rdf_s.rankings = "A"
        new_unique_idx = range(rankings_df.index.max() + 1, rankings_df.index.max() + 1 + n_for_each_psi)

        # print(new_unique_idx)

        rdf_s.unique_idx = [r for r in new_unique_idx]

        rdf_s.index = rdf_s.unique_idx

        rdf_s.winning_idx = rdf_s.first_pair_idx

        new_rankings_list.append(rdf_s)

    if len(new_rankings_list) > 0:
        rdf_s = pd.concat(new_rankings_list, axis=0, ignore_index=True)

        new_unique_idx = range(rankings_df_sub.index.max() + 1, rankings_df_sub.index.max() + 1 + rdf_s.shape[0])

        rdf_s.unique_idx = [r for r in new_unique_idx]

        rdf_s.index = rdf_s.unique_idx

        rankings_df_sub = pd.concat([rankings_df_sub, rdf_s], axis=0)

        rankings_df_sub.shape

    return rankings_df_sub, dict_of_depth_pt_msi


def create_new_dset(names_of_pt, sel_dsets=None):
    rankings_df_sub, dict_of_depth_pt = format_rankings(names=sel_dsets)

    split_out_and_save(names_of_pt, rankings_df_sub, dict_of_depth_pt)


def create_new_dset_w_extra(names_of_pt, m_psi_best, n_good_mesh_to_use=5, n_for_each_psi=30, sel_dsets=None):
    print(sel_dsets)
    rankings_df_sub, dict_of_depth_pt = format_rankings(names=sel_dsets)
    rankings_df_sub, new_pt = append_to_rankings(rankings_df_sub, m_psi_best, n_good_mesh_to_use, n_for_each_psi)  # <---- do this one AFTER we have test,val,train

    dict_of_depth_pt.update(new_pt)
    split_out_and_save(names_of_pt, rankings_df_sub, dict_of_depth_pt)


def create_new_dset_w_extra_separated(
    dset_kwargs,
):  # ,names_of_pt,m_psi_best,n_for_each_psi=30,sel_dsets=None):
    n_for_each_psi = dset_kwargs["n_for_each_psi"]
    n_train = dset_kwargs["good_mesh_train"]
    n_val = dset_kwargs["good_mesh_val"]
    n_test = dset_kwargs["good_mesh_test"]
    # batch_size=dset_kwargs['batch_size']
    names_of_pt = dset_kwargs["names_of_pt"]
    sel_dsets = dset_kwargs["sel_dsets"]

    m_psi_best = dset_kwargs["m_psi_best"]

    # format proportions

    props = dset_kwargs["props"]

    # normalise proportions if necessary

    assert len(props) == 3
    assert sum(props) == 1.0

    print(sel_dsets)
    rankings_df_sub, dict_of_depth_pt = format_rankings(names=sel_dsets)
    # rankings_df_sub,new_pt=append_to_rankings(rankings_df_sub,m_psi_best,n_good_mesh_to_use,n_for_each_psi) #<---- do this one AFTER we have test,val,train

    # dict_of_depth_pt.update(new_pt)

    msi_partitions = {}

    # n_train=15
    # n_test=5
    # n_val=7

    msi_partitions["train"] = m_psi_best.sample(n=n_train)
    m_psi_best.drop(msi_partitions["train"].index, inplace=True)
    msi_partitions["test"] = m_psi_best.sample(n=n_test)
    m_psi_best.drop(msi_partitions["test"].index, inplace=True)
    msi_partitions["val"] = m_psi_best.sample(n=n_val)

    msi_partitions["n_for_each_psi"] = n_for_each_psi

    split_out_and_save(
        names_of_pt,
        rankings_df_sub,
        dict_of_depth_pt,
        m_psi_partitions=msi_partitions,
        dset_kwargs=dset_kwargs,
    )


def read_pseudo_meshes(pseudo_mesh_dir, exclude_meshes=None):
    pseudo_mesh_dir = [pseudo_mesh_dir]
    m_psi_best = get_all_meshes_for_binary(select_only=pseudo_mesh_dir)

    if exclude_meshes is None:
        return m_psi_best

    with open(exclude_meshes) as f:
        exclude_meshes = f.readlines()

    exclude_meshes = [x.strip() for x in exclude_meshes]
    exclude_meshes = [int(x) for x in exclude_meshes]
    exclude_meshes = np.array(exclude_meshes)
    m_psi_best.seed = m_psi_best.seed.astype(int)
    m_psi_best.index = m_psi_best.seed
    m_psi_best = m_psi_best[~m_psi_best.seed.isin(exclude_meshes)]
    m_psi_best.seed = m_psi_best.seed.astype(str)

    return m_psi_best


# names_of_pt=dict(train='train_dset_13_08_2023.pt',val='val_dset_13_08_2023.pt',test='test_dset_13_08_2023.pt')


# create_new_dset(names_of_pt,sel_dsets=sel_dsets)
