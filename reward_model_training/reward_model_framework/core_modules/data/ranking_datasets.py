"""
Builders for ranking-based datasets and helpers to prepare ranking CSVs.

This module centralizes the legacy `rld` dataset wiring so call sites can use
clear, descriptive imports without digging through the old monolith.
"""

import re
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import pandas as pd
from sklearn.model_selection import train_test_split

from core_modules.data import all_data_types
from core_modules.data import dset_loaders as dl
from core_modules.data import misc_small_utils as msu
from core_modules.data import ranking_utils as r_utils
from core_modules.utils import misc_helpers as mh
from core_modules.utils.pylogger_c import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)
PROJECT_ROOT = Path(__import__("os").environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[5]))
SRC_RLHF_ROOT = PROJECT_ROOT / "reward_model_training" / "reward_model_framework" / "core_modules"


def create_dset_from_types_and_dsets_dict(dict_of_dsets, selected_dtypes, goodseeds_list, badseeds_dict):
    """Combine selected single-stream datasets into a multi-stream wrapper."""
    sel_dsets_dict = {k: dict_of_dsets[k] for k in selected_dtypes}
    sel_dsets = [sel_dsets_dict[i] for i in sel_dsets_dict.keys()]

    for s in sel_dsets:
        s.list_of_good_seeds = goodseeds_list
        s.badseeds_dict = badseeds_dict

    return dl.dset_smulti_stream(sel_dsets)


def get_dsets_dict_for_list_of_rankings_minimal(
    list_of_rankings,
    ddir_func,
    seed_func,
    using_transforms=False,
    augmentations=None,
    goodmesh_augment=None,
    dset_partition="",
    for_contrastive=False,
    include_goodseed=False,
    batch_augmentations=None,
    map_on=None,
):
    """Instantiate datasets for every dtype given the rankings array."""
    dict_of_dsets = {}
    for dt in all_data_types.ALL_DATA_TYPES:
        if not for_contrastive:
            dset = dl.dset_single_stream_ordered_minimal(
                all_combined_rankings=list_of_rankings,
                dtype=dt,
                ddir_func=ddir_func,
                seed_func=seed_func,
                using_transforms=using_transforms,
                augmentations=augmentations,
                goodmesh_augment=goodmesh_augment,
                dset_partition=dset_partition,
                include_goodseed=include_goodseed,
                batch_augmentations=batch_augmentations,
                map_on=map_on,
            )
        else:
            dset = dl.dset_contrastive_second_stage(
                all_combined_rankings=list_of_rankings,
                dtype=dt,
                ddir_func=ddir_func,
                seed_func=seed_func,
                using_transforms=using_transforms,
                augmentations=augmentations,
                goodmesh_augment=goodmesh_augment,
                dset_partition=dset_partition,
                include_goodseed=include_goodseed,
                batch_augmentations=batch_augmentations,
            )

        dict_of_dsets[dt] = dset

    return dict_of_dsets


def set_up_dataloaders(cfg, collate_fn):
    """Instantiate datasets via Hydra config and propagate run-time attributes."""
    data = hydra.utils.instantiate(cfg.data)

    dloader_params = {
        "num_workers": cfg.dloader.num_workers,
        "pin_memory": cfg.dloader.pin_memory,
        "collate_fn": collate_fn,
    }

    for i, dk in enumerate(cfg.data.dset_dict.selected_dtypes):
        for k in data.finished_dset.keys():
            data.finished_dset[k].dsets[i].update_attrs_from_run_dict(cfg.data.dset_dict[dk])

    return dict(data=data)


def return_checked_combos(data_dir, dset_dict_config):
    """Load ranking CSVs, validate rows, optionally trim/augment them, and return a filtered dataframe."""
    ranked = pd.read_csv(SRC_RLHF_ROOT / "data" / "create_train_data" / "rankedseedsall.csv", index_col=0)

    if ranked.shape[0] > 0:
        return ranked  # hack  13122025

    ranked_c = ranked[ranked.completed == True]
    ranked.head()

    all_idx = [i for i in ranked_c.index]
    checked_rows = [r_utils.check_row(idx, ranked_c) for idx in all_idx]
    invert_idx = [not c for c in checked_rows]
    idx_with_mistakes = np.array(all_idx)[invert_idx]
    print("mistake rankings idx")
    print(len(idx_with_mistakes))

    checked_rankings = ranked_c[checked_rows]

    remove_middle_data = dset_dict_config.remove_middle_data

    print("-------------------------")
    print("")
    print("remove middle data")
    print(remove_middle_data)
    print("")
    print("-------------------------")

    checked_rankings[(checked_rankings != -1)]

    cr = checked_rankings[["rank1", "rank2", "rank3", "rank4", "rank5", "rank6", "rank7"]]

    row_size_to_keep = dset_dict_config.row_size_to_keep
    keep_idx = cr[(cr.isna()).sum(1) <= 7 - row_size_to_keep].index
    keep_idx2 = [l for l in keep_idx]
    checked_rankings = checked_rankings.loc[keep_idx2]

    if remove_middle_data:
        print("removing middle rankings data")

        all_rankings = checked_rankings[["rank1", "rank2", "rank3", "rank4", "rank5", "rank6", "rank7"]]
        aux_data = checked_rankings[[c for c in checked_rankings.columns if c not in ["rank1", "rank2", "rank3", "rank4", "rank5", "rank6", "rank7"]]]

        ar_np = all_rankings.values
        ri = all_rankings.isna()
        first_entry = [a for a in ar_np[:, 0]]
        rev_vals = ar_np[:, ::-1]
        last_entry = []
        for r in rev_vals:
            conv_str = [str(rr) for rr in r]
            not_nan = [c != "nan" for c in conv_str]
            last_let = r[not_nan][0]
            last_entry.append(last_let)
            rev_str = conv_str[::-1]
            idx_match = np.array([r == last_let for r in rev_str])
            idx_num = np.where(idx_match)[0][0]

        ar_np.fill(np.nan)

        ap = pd.DataFrame(ar_np)
        ap.columns = all_rankings.columns

        ap["rank1"].loc[:] = first_entry
        ap["rank2"].loc[:] = last_entry
        ap.index = all_rankings.index

        aux_data.loc[:, "n_in_row"] = [2 for k in range(aux_data.shape[0])]

        remerge_ranks = pd.concat([aux_data, ap], axis=1)
        checked_rankings = remerge_ranks

    total_combos = [r_utils.get_n_combinations(nrow) for nrow in checked_rankings.n_in_row]
    print("total n combo")
    print(np.sum(total_combos))

    return checked_rankings


class RankingDataBuilder:
    """Legacy dset_helper rewritten with clearer naming."""

    def __init__(self, dset_dict, augmentations=None, batch_augmentations=None, map_on=None, **kwargs):
        self.dset_dict = dset_dict
        self.badseeds_models = self.dset_dict.badseeds_models if self.dset_dict.get("badseeds_models") else []
        self.double_batch_size = False
        self.badseeds_dict = {}

        self.setup_and_save(augmentations, batch_augmentations, map_on)

    def setup_and_save(self, augmentations, batch_augmentations, map_on):
        self.get_checked_rankings()

        if self.dset_dict.dset_version != "one":
            self.split_partitions()

        self.split_seeds_from_partitions()
        self.get_goodmeshes_and_split()

        self.set_ddir_and_seed_func()

        self.save_finished_dset(augmentations=augmentations, for_contrastive=False, include_goodseed=self.dset_dict.include_goodseed, batch_augmentations=batch_augmentations, map_on=map_on)

        if self.dset_dict.get("augmentations"):
            self.save_finished_dset(augmentations=augmentations, for_contrastive=True, include_goodseed=self.dset_dict.include_goodseed, batch_augmentations=batch_augmentations, map_on=map_on)

        return self

    def set_ddir_and_seed_func(self):
        if self.dset_dict.dset_version in {"three", "inverted"}:
            self.ddir_func = msu.ddir_func
            self.seed_func = msu.seed_func_default

    def find_seed_from_new_fn(self, fn):
        pattern = r"s_\d+"
        matches = re.findall(pattern, fn)
        assert len(matches) == 1, "error more than1 match here"
        return matches[0]

    def get_checked_rankings(self):
        mh.print_break("getting checked rankings")

        if self.dset_dict.dset_version == "three":
            print("dset version three, ie multi pairs from latest, only sample ffhq no trunctation random noise")

            data_dir = self.dset_dict.data_dir
            self.checked_rankings = return_checked_combos(data_dir=data_dir, dset_dict_config=self.dset_dict)
            proportion_of_data_to_use = self.dset_dict.proportion_of_data_to_use

            if proportion_of_data_to_use < 1:
                self.checked_rankings = self.checked_rankings.sample(frac=proportion_of_data_to_use)

        return self

    def split_partitions(self):
        mh.print_break("split into testi train val")

        n_train = self.dset_dict.partition_splits.n_train
        n_val = self.dset_dict.partition_splits.n_val
        n_test = self.dset_dict.partition_splits.n_test

        train, val, test = mh.split_dset_partitions(self.checked_rankings, n_train, n_val, n_test)
        self.train = train
        self.val = val
        self.test = test

        return self

    def split_seeds_from_partitions(self):
        mh.print_break("splitting into seeds")

        self.train_ranked_seeds = r_utils.create_list_of_rankings_minimal(self.train)
        self.val_ranked_seeds = r_utils.create_list_of_rankings_minimal(self.val)
        self.test_ranked_seeds = r_utils.create_list_of_rankings_minimal(self.test)

        return self

    def get_goodmeshes_and_split(self):
        good_spec = self.dset_dict.goodmeshes

        n_train = good_spec.n_train
        n_test = good_spec.n_test
        n_val = good_spec.n_val

        seeds_good_filtered = [i for i in range(100000, 101000)]

        self.goodmesh_train = self.goodmesh_val = self.goodmesh_test = []

        if isinstance(n_train, float) and isinstance(n_test, float) and isinstance(n_val, float):
            print("using percentages for goodmeshes")
            assert False, "error cant use percentages for goodmeshes. use integer values to specify how many goodmeshes to include in the dataset"
        elif isinstance(n_train, int) and isinstance(n_test, int) and isinstance(n_val, int):
            print("using int amounts for goodmeshses,correct")

        no_test_val = (n_test == 0) and (n_val == 0)

        if n_train > 0 and n_val > 0 and n_test > 0:
            self.goodmesh_train, vt = train_test_split(seeds_good_filtered, test_size=n_test + n_val, train_size=n_train, random_state=42)
            (
                self.goodmesh_val,
                self.goodmesh_test,
            ) = train_test_split(vt, test_size=n_test, train_size=n_val, random_state=42)
        elif n_train > 0 and no_test_val:
            self.goodmesh_train, _ = train_test_split(seeds_good_filtered, train_size=n_train, random_state=42)
        elif n_train > 0 and n_val > 0:
            self.goodmesh_train, self.goodmesh_val = train_test_split(seeds_good_filtered, test_size=n_val, train_size=n_train, random_state=42)
        elif n_train > 0 and n_test > 0:
            self.goodmesh_train, self.goodmesh_test = train_test_split(seeds_good_filtered, test_size=n_test, train_size=n_train, random_state=42)
        elif n_train == 0 and n_val > 0 and n_test == 0:
            self.goodmesh_val, _ = train_test_split(seeds_good_filtered, train_size=n_val, random_state=42)
        elif n_train == 0 and n_val == 0 and n_test > 0:
            self.goodmesh_test, _ = train_test_split(seeds_good_filtered, train_size=n_test, random_state=42)
        elif n_train == 0 and n_val > 0 and n_test > 0:
            (
                self.goodmesh_val,
                self.goodmesh_test,
            ) = train_test_split(seeds_good_filtered, test_size=n_test, train_size=n_val, random_state=42)

        return self

    def save_finished_dset(self, augmentations, for_contrastive, include_goodseed, batch_augmentations, map_on):
        using_transforms = self.dset_dict.using_transforms

        fc = for_contrastive

        if fc:
            aug = augmentations.contrastive.train
        else:
            aug = augmentations.train

        ba = batch_augmentations
        print(ba)

        if isinstance(ba, str):
            if not ba.endswith(".yaml"):
                ba += ".yaml"

            ba_fn = SRC_RLHF_ROOT / "configs" / "data" / "batch_augmentations" / Path(ba).name
            ba_cfg = omegaconf.OmegaConf.load(ba_fn)
            batch_augmentations = ba_cfg

        train_dsets_dict = get_dsets_dict_for_list_of_rankings_minimal(
            list_of_rankings=self.train_ranked_seeds,
            ddir_func=self.ddir_func,
            seed_func=self.seed_func,
            using_transforms=using_transforms.train,
            augmentations=aug,
            goodmesh_augment=augmentations.goodmesh_augment,
            dset_partition="train",
            for_contrastive=fc,
            include_goodseed=include_goodseed,
            batch_augmentations=batch_augmentations.train,
            map_on=map_on,
        )

        if fc:
            aug = augmentations.contrastive.val
        else:
            aug = augmentations.val

        val_dsets_dict = get_dsets_dict_for_list_of_rankings_minimal(
            list_of_rankings=self.val_ranked_seeds,
            ddir_func=self.ddir_func,
            seed_func=self.seed_func,
            using_transforms=using_transforms.val,
            augmentations=aug,
            goodmesh_augment=augmentations.goodmesh_augment,
            dset_partition="val",
            for_contrastive=fc,
            include_goodseed=include_goodseed,
            batch_augmentations=batch_augmentations.val,
            map_on=map_on,
        )

        if fc:
            aug = augmentations.contrastive.test
        else:
            aug = augmentations.test

        test_dsets_dict = get_dsets_dict_for_list_of_rankings_minimal(
            list_of_rankings=self.test_ranked_seeds,
            ddir_func=self.ddir_func,
            seed_func=self.seed_func,
            using_transforms=using_transforms.test,
            augmentations=aug,
            dset_partition="test",
            for_contrastive=fc,
            include_goodseed=include_goodseed,
            batch_augmentations=batch_augmentations.test,
            map_on=map_on,
        )

        selected_dtypes = self.dset_dict.selected_dtypes

        train_dset = create_dset_from_types_and_dsets_dict(
            dict_of_dsets=train_dsets_dict,
            selected_dtypes=selected_dtypes,
            goodseeds_list=self.goodmesh_train,
            badseeds_dict=self.badseeds_dict,
        )
        val_dset = create_dset_from_types_and_dsets_dict(
            dict_of_dsets=val_dsets_dict,
            selected_dtypes=selected_dtypes,
            goodseeds_list=self.goodmesh_val,
            badseeds_dict=self.badseeds_dict,
        )
        test_dset = create_dset_from_types_and_dsets_dict(
            dict_of_dsets=test_dsets_dict,
            selected_dtypes=selected_dtypes,
            goodseeds_list=self.goodmesh_test,
            badseeds_dict=self.badseeds_dict,
        )

        finished_dset = dict(train=train_dset, test=test_dset, val=val_dset)

        if for_contrastive:
            self.finished_dset_contrastive = finished_dset
        else:
            self.finished_dset = finished_dset
        return self


# Backwards compatibility: legacy name used throughout configs/code.
dset_helper = RankingDataBuilder

__all__ = [
    "RankingDataBuilder",
    "create_dset_from_types_and_dsets_dict",
    "dset_helper",
    "get_dsets_dict_for_list_of_rankings_minimal",
    "return_checked_combos",
    "set_up_dataloaders",
]
