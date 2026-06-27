import itertools
import random

import cv2
import hydra
import numpy as np
import skimage.exposure as exposure
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core_modules.models.utils_transformer import CrossAttentionModule, SequenceEncoderWithFourierSigma, TransformerGPT
from core_modules.models.base import UniversalRWDModel
from core_modules.models.utils_base import ce_loss, get_ncomb2, get_rand_reordered_pair, reorder_pair_by_idx


def get_gaussian_msk_centered(delta_offset=0, patchsize=(32, 32)):
    dims = 30
    dims2 = 30 // 2
    delta = np.zeros((dims, dims, 3), dtype=np.float32)
    sigmax = int(random.random() * 5) + 1
    sigmay = int(random.random() * 5) + 1
    delta[dims2 + delta_offset : dims2 + 1 + delta_offset, dims2 + delta_offset : dims2 + 1 + delta_offset] = (255, 255, 255)
    blur = cv2.GaussianBlur(delta, (0, 0), sigmaX=sigmax, sigmaY=sigmay)
    dims4x = dims * 16
    resized = cv2.resize(blur, (dims4x, dims4x), interpolation=cv2.INTER_AREA)
    result = exposure.rescale_intensity(resized, in_range="image", out_range=(0, 255)).astype(np.uint8)
    result = cv2.resize(result, patchsize)
    msk = (result / 255.0).astype(np.float32)
    msk = msk[:, :, 0]
    ttr = torch.from_numpy(msk)
    return ttr


class sigma_rays_transformer(UniversalRWDModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        embed_dim = kwargs["embed_dim"]
        num_heads = kwargs["num_heads"]
        cross_attention_dropout = kwargs["cross_attention_dropout"]
        encoder_dropout = kwargs["encoder_dropout"]
        self_attention_dropout = kwargs["self_attention_dropout"]
        self.global_feature_size = kwargs["global_feature_size"]
        self.encoder = SequenceEncoderWithFourierSigma(input_dim=1, embed_dim=embed_dim, num_heads=num_heads, forward_expansion=4, dropout=encoder_dropout, max_length=128, num_features=1)
        self.self_attention = TransformerGPT(embed_size=embed_dim, heads=4, depth=4, forward_expansion=4, dropout=self_attention_dropout)
        self.cross_attention = CrossAttentionModule(embed_dim=embed_dim, num_heads=num_heads, cross_attention_dropout=cross_attention_dropout)

        self.dp3d = nn.Dropout3d(p=0.15)
        self.conv1 = nn.Conv3d(1, 8, kernel_size=5)
        self.norm1 = nn.BatchNorm3d(8)
        self.pool1 = nn.MaxPool3d([5, 1, 1], stride=1)

        self.conv2 = nn.Conv3d(8, 8, kernel_size=[6, 1, 1], stride=1, padding=0)
        self.norm2 = nn.BatchNorm3d(8)
        self.pool2 = nn.MaxPool3d([5, 3, 3], stride=1)

        self.conv3 = nn.Conv3d(8, 8, kernel_size=[8, 1, 1])
        self.norm3 = nn.BatchNorm3d(8)
        self.pool3 = nn.MaxPool3d([32, 1, 1], stride=[5, 1, 1])

        self.attn_msk_list = None
        self.n_kpts = kwargs["n_kpts"]
        self.MINSCALE = kwargs.get("MINSCALE", 1.0)

        self.remove_MLPs()
        self.comparator = hydra.utils.instantiate(kwargs["mlp_transformer"])
        self.sm = torch.nn.Softmax(1)

        self.external = None
        self.external_init_dict = kwargs["external"]
        if self.external_init_dict is not None:
            self.set_external()

        self.hparams.optimizer = kwargs["optimizer"]
        if "scheduler" in kwargs.keys():
            self.hparams.scheduler = kwargs["scheduler"]

        self.train_step = 0
        self.setup_opt_sched_init()

        self.dict_of_regions = {
            "LHS_face_idx": list(range(0, 16)),
            "nose_idx": [54],
            "RHS_face_idx": list(range(17, 33)),
            "mouth_face_idx": list(range(76, 96)),
            "nose_ridge_idx": list(range(51, 55)),
            "left_eye_idx": list(range(60, 68)) + [97],
            "right_eye_idx": list(range(60, 68)) + [96],
            "septum_idx": list(range(55, 59)),
            "left_brow_idx": list(range(33, 42)),
            "right_brow_idx": list(range(42, 51)),
            "all_idx": list(range(98)),
        }
        sel_regions = ["LHS_face_idx", "RHS_face_idx", "nose_ridge_idx", "nose_ridge_idx", "nose_idx", "nose_idx", "left_eye_idx", "right_eye_idx"]
        attn_msk_dict = {}
        for s in sel_regions:
            template_msk = torch.ones((self.n_kpts, self.n_kpts)) - 1.0
            msk = self.dict_of_regions[s]
            for row in msk:
                for col in msk:
                    template_msk[row, col] = 0.0
            attn_msk_dict[s] = template_msk
        self.attn_msk_dict = attn_msk_dict
        self.sel_region_for_masking = kwargs.get("sel_region_for_masking")

    def remove_MLPs(self):
        for attr_l in ["MLP", "scalar_rwd_head_BT", "scalar_rwd_head", "scalar_rwd_head_pairs"]:
            if hasattr(self, attr_l):
                delattr(self, attr_l)
        return self

    def forward_to_global_feature_vec(self, x):
        if len(x.shape) == 5:
            xs1, xs2, xs3, xs4, xs5 = x.shape
            x = x.reshape(xs1 * xs2, xs3, xs4, xs5)
        embds_for_in = self.cutout_nose_region(x)
        embds_for_in = torch.cat([self.random_pad_input(cv).unsqueeze(0) for cv in embds_for_in], 0)
        embds_for_in[embds_for_in == -1] = 0.0
        x = embds_for_in + torch.randn_like(embds_for_in) * 1e-2

        x1 = self.pool1(self.dp3d(F.relu(self.norm1(self.conv1(x.unsqueeze(1))))))
        x2 = self.pool2(self.dp3d(F.relu(self.norm2(self.conv2(x1)))))
        x3 = self.pool3(self.dp3d(F.relu(self.norm3(self.conv3(x2)))))
        x3 = x3.squeeze(-1, -2)
        ret_emb = x3.reshape(-1, 16 * 8).unsqueeze(2)

        feature_vec = self.encoder(ret_emb)
        encoded_seq = self.self_attention(feature_vec)
        return encoded_seq

    def on_before_optimizer_step(self, optimizer: optim.Optimizer) -> None:
        self.train_step = self.train_step + 1
        return super().on_before_optimizer_step(optimizer)

    def calc_scaling_factor(self, keys, just_scaling_factor=False):
        if not self.training:
            scaling_factor = self.MINSCALE
        else:
            total_stepping_batches = self.trainer.estimated_stepping_batches
            current_step = self.train_step
            scaling_factor = max(1 - current_step * 1.0 / total_stepping_batches, self.MINSCALE)
        if just_scaling_factor:
            return scaling_factor
        scale = torch.ones_like(keys)
        if self.sel_region_for_masking is not None:
            s = self.sel_region_for_masking
            msk = self.dict_of_regions[s]
            keys_to_scale = [i for i in torch.arange(keys.shape[1]) if i not in msk]
            scale[:, keys_to_scale, :] *= scaling_factor
        else:
            msk = scaling_factor + (1 - scaling_factor) * get_gaussian_msk_centered()
            mskf = msk.flatten().to(keys.device)
            mskf = mskf[None, :, None]
            mskf = mskf.expand(*keys.shape)
            scale = mskf
        return scale

    def random_pad_input(self, cv3_input, pad_start_only=True):
        rays_start = torch.randint(low=0, high=7, size=(1,), device=cv3_input.device)
        pad = torch.zeros_like(cv3_input)[:7, :, :] - 1
        if rays_start == 0 or pad_start_only:
            cv3_input_padded = torch.cat((cv3_input, pad), dim=0)
        elif rays_start == 7:
            cv3_input_padded = torch.cat((pad, cv3_input), dim=0)
        else:
            pad_start, pad_end = torch.split(pad, [rays_start, 7 - rays_start], dim=0)
            cv3_input_padded = torch.cat((pad_start, cv3_input, pad_end), dim=0)
        return cv3_input_padded

    def forward_from_cat_global_vectors(self, enc_seq1, enc_seq2, with_softmax=False, attn_mask=None):
        enc_seq1_self = enc_seq1
        enc_seq2_self = enc_seq2
        if attn_mask is not None:
            attn_seq1_seq2 = self.cross_attention.forward(enc_seq1_self, enc_seq2_self, enc_seq2_self, attn_mask=attn_mask.squeeze(0).to(self.device))
            attn_seq2_seq1 = self.cross_attention.forward(enc_seq2_self, enc_seq1_self, enc_seq1_self, attn_mask=attn_mask.squeeze(0).to(self.device))
        else:
            attn_seq1_seq2 = self.cross_attention.forward(enc_seq1_self, enc_seq2_self, enc_seq2_self)
            attn_seq2_seq1 = self.cross_attention.forward(enc_seq2_self, enc_seq1_self, enc_seq1_self)
        combined_features_seq1_seq2 = torch.cat((attn_seq1_seq2, attn_seq2_seq1), dim=-1)
        combined_features_seq2_seq1 = torch.cat((attn_seq2_seq1, attn_seq1_seq2), dim=-1)
        combined_features_seq1_seq2 = torch.mean(combined_features_seq1_seq2, dim=1)
        combined_features_seq2_seq1 = torch.mean(combined_features_seq2_seq1, dim=1)
        logits_seq1_seq2 = self.comparator(combined_features_seq1_seq2)
        logits_seq2_seq1 = self.comparator(combined_features_seq2_seq1)
        logits_seq2_seq1_rev = logits_seq2_seq1.flip(-1)
        logits = (logits_seq1_seq2 + logits_seq2_seq1_rev) / 2
        if with_softmax:
            logits = self.sm(logits)
        return logits

    def get_activation_maps(self, seq1, seq2, attn_mask=None):
        enc_seq1 = self.encoder(seq1)
        enc_seq2 = self.encoder(seq2)
        key_scale = self.calc_scaling_factor(enc_seq2)
        attn_seq1_maps = self.cross_attention.get_attention_maps(enc_seq1, enc_seq2, enc_seq2, keyscale=key_scale)
        attn_seq2_maps = self.cross_attention.get_attention_maps(enc_seq2, enc_seq1, enc_seq1, keyscale=key_scale)
        return dict(attn_seq1_maps=attn_seq1_maps, attn_seq2_maps=attn_seq2_maps)

    def get_activation_maps_from_embedded(self, enc_seq1, enc_seq2, attn_mask=None):
        key_scale = self.calc_scaling_factor(enc_seq2)
        attn_seq1_maps = self.cross_attention.get_attention_maps(enc_seq1, enc_seq2, enc_seq2, keyscale=key_scale)
        attn_seq2_maps = self.cross_attention.get_attention_maps(enc_seq2, enc_seq1, enc_seq1, keyscale=key_scale)
        return dict(attn_seq1_maps=attn_seq1_maps, attn_seq2_maps=attn_seq2_maps)

    def forward_to_scalar_reward_from_single_global(self, x):
        return super().forward_to_scalar_reward_from_single_global(x)

    def forward_to_BT_lambda_from_single_global(self, x, mult=1):
        return super().forward_to_BT_lambda_from_single_global(x, mult)

    def return_windowed_attention_mask(self, seq_len=128, window_size=7):
        assert window_size % 2 == 1, "error need to have odd for window size"
        b = torch.ones((seq_len, seq_len))
        win_half = int(window_size / 2)
        win_at = torch.triu(b, diagonal=-win_half) * torch.tril(b, diagonal=win_half)
        return win_at

    def cutout_nose_region(self, embds_for_in):
        X = Y = 63
        OFF = 3
        xsqu3d = embds_for_in[:, X - OFF : X + OFF + 1, Y - OFF : Y + OFF + 1, :]
        cv3_input = xsqu3d.permute(0, 3, 1, 2)
        return cv3_input

    def run_forward_pass(self, batch, generate_heatmaps=False, return_global_vector=False, return_preds=True):
        device = batch.file_batch.device
        X_dmap = batch.file_batch.to(device, non_blocking=False).squeeze(2)
        Lengths = batch.lens_batch.to(device, non_blocking=False)
        seeds = batch.ordered_seeds.to(device, non_blocking=False)
        heatmap_images = []
        if self.external is not None:
            Lengths_idx = [torch.arange(L) for L in Lengths]
            feature_embeddings = [self.external.forward(xd[L]) for xd, L in zip(X_dmap, Lengths_idx)]
            feature_embeddings = [self.forward_to_global_feature_vec(x) for x in feature_embeddings]
            cpreds = feature_embeddings
        else:
            Lengths_idx = [torch.arange(L) for L in Lengths]
            permutation_of_lengths = [torch.arange(L) for L in Lengths]
            inverse_permutation_of_lengths = [torch.argsort(p) for p in permutation_of_lengths]
            Lengths_reordered = [L[p] for L, p in zip(Lengths_idx, permutation_of_lengths)]
            embds_for_in = [xd[L] for xd, L in zip(X_dmap, Lengths_reordered)]
            embds_for_in = torch.vstack(embds_for_in)
            embds_for_in = embds_for_in.chunk(max(1, int(embds_for_in.shape[0] / 8)))
            feature_emb_chunk = [self.forward_to_global_feature_vec(x) for x in embds_for_in]
            feature_emb_chunk = torch.vstack(feature_emb_chunk)
            feature_embeddings = torch.split(feature_emb_chunk, Lengths.tolist(), dim=0)
            cpreds = [cp[i] for cp, i in zip(feature_embeddings, inverse_permutation_of_lengths)]
        cseeds = [s[:L].reshape(-1, 1) for s, L in zip(seeds, Lengths)]
        global_feature_preds = cpreds
        dict_of_global_feature = {"-10000": torch.tensor(1)}
        CPU_SEEDS = torch.vstack(cseeds)
        attn_mask = self.return_windowed_attention_mask()
        if return_global_vector and return_preds:
            global_embeddings = torch.vstack(global_feature_preds).view(CPU_SEEDS.shape[0], -1)
            dict_of_global_feature = {s: v[None, ...].detach() for s, v in zip(CPU_SEEDS, global_embeddings)}

        ordered_combos = [list(itertools.combinations(range(L), 2)) for L in Lengths]
        binary_idx_order_rand = [[get_rand_reordered_pair(p=1) for _ in oc] for oc in ordered_combos]
        ordered_combos_rand = [[reorder_pair_by_idx(o, p) for o, p in zip(oc, pc)] for oc, pc in zip(ordered_combos, binary_idx_order_rand)]
        Lengths_np = Lengths.detach().cpu().numpy()
        sum_in_batch = np.sum([get_ncomb2(l) for l in Lengths_np])
        batches = [torch.cat([torch.cat((cp[o[0]].unsqueeze(0), cp[o[1]].unsqueeze(0)), 1) for o in oc]) for oc, cp in zip(ordered_combos_rand, cpreds)]
        seeds_batches = [torch.cat([torch.cat((cp[o[0]].unsqueeze(0), cp[o[1]].unsqueeze(0)), 1) for o in oc]) for oc, cp in zip(ordered_combos_rand, cseeds)]
        seeds_batches_rev = [torch.cat([torch.cat((cp[o[1]].unsqueeze(0), cp[o[0]].unsqueeze(0)), 1) for o in oc]) for oc, cp in zip(ordered_combos_rand, cseeds)]
        targets = [torch.tensor([0 for _ in bi], dtype=torch.uint8, device=device) for bi in binary_idx_order_rand]
        targets_rev = [torch.tensor([1 for _ in bi], dtype=torch.uint8, device=device) for bi in binary_idx_order_rand]
        cat_targ = torch.cat(targets, 0)
        cat_targ_rev = torch.cat(targets_rev, 0)
        all_targ = torch.cat((cat_targ, cat_targ_rev), dim=0).view(2, -1).t().reshape(-1, 1).split(2)
        cat_seeds = torch.cat(seeds_batches, 0)
        cat_seeds_rev = torch.cat(seeds_batches_rev, 0)
        return_for_seeds = torch.cat((cat_seeds, cat_seeds_rev), dim=0)
        return_for_logits = torch.ones_like(return_for_seeds)

        paired_loss = torch.tensor(0.0, device=device)
        if self.hparams.loss["lambda_pairs"] != 0.0:
            batches_pred = [
                self.forward_from_cat_global_vectors(
                    b[:, : self.global_feature_size],
                    b[:, self.global_feature_size :],
                    with_softmax=False,
                )
                for b in batches
            ]
            batches_rev_pred = [
                self.forward_from_cat_global_vectors(
                    b[:, self.global_feature_size :],
                    b[:, : self.global_feature_size],
                    with_softmax=False,
                )
                for b in batches
            ]
            cat_batches = torch.cat(batches_pred, 0)
            cat_batches_rev = torch.cat(batches_rev_pred, 0)
            all_batches = torch.stack((cat_batches, cat_batches_rev), dim=1).view(-1, 2).split(2)
            weights = torch.tensor([1.0, 1.0]).to(all_batches[0].device)
            sel_idx = torch.multinomial(weights, len(all_targ), replacement=True)
            paired_loss = torch.hstack([ce_loss(a, t.flatten()) for a, t in zip(all_batches, all_targ)]).mean() * self.hparams.loss["lambda_pairs"]
            batches_rev = torch.cat(batches_rev_pred, 0)
            batches = torch.cat(batches_pred, 0)
            return_for_logits = torch.cat((batches_rev[:, 1:], batches_rev[:, :1]), 1)
            return_for_logits = torch.cat((batches, return_for_logits), 0).detach()

        scalar_rwd_loss = torch.tensor(0.0, device=device)
        sigmoid_comparison_values = [-1]
        scalar_rwds_dist_dict_scalar = {}
        rwd_vals_for_ret_scalar = torch.tensor(-1.0, device=device)
        if self.hparams.loss["lambda_scalar_rwd"] != 0.0:
            globals_for_scalar = cpreds
            ordered_combos = [list(itertools.combinations(range(L), 2)) for L in Lengths]
            cpreds_scalar_rwd = [self.forward_to_scalar_reward_from_single_global(x) for x in globals_for_scalar]
            intermediate_losses = [[-torch.log(torch.sigmoid(cp[o[0]] - cp[o[1]])) for o in oc] for oc, cp in zip(ordered_combos, cpreds_scalar_rwd)]
            il = [torch.mean(torch.cat(im)).unsqueeze(0) for im in intermediate_losses]
            scalar_rwd_loss = torch.cat(il).mean() * self.hparams.loss["lambda_scalar_rwd"]
            rwd_vals_for_ret_scalar = torch.vstack(cpreds_scalar_rwd).detach()
            scalar_rwds_dist_dict_scalar = {s: v for s, v in zip(CPU_SEEDS, rwd_vals_for_ret_scalar.flatten())}
            cpreds_scalar_rwd = [c.detach() for c in cpreds_scalar_rwd]
            sigmoid_comparison_values = [torch.hstack([cp[o[0]] - cp[o[1]] for o in oc]) for oc, cp in zip(ordered_combos, cpreds_scalar_rwd)]

        BT_rwd_loss = torch.tensor(0.0, device=device)
        BT_comparison_values = [-1]
        rwd_vals_for_ret_BT = torch.tensor(-1.0, device=device)
        rwds_dist_dict_BT = {}
        if self.hparams.loss["lambda_BT"] != 0.0:
            globals_for_BT = cpreds
            ordered_combos = [list(itertools.combinations(range(L), 2)) for L in Lengths]
            BT_vals = [self.forward_to_BT_lambda_from_single_global(x) for x in globals_for_BT]
            intermediate_losses = [[-torch.log(cp[o[0]] / (cp[o[0]] + cp[o[1]])) for o in oc] for oc, cp in zip(ordered_combos, BT_vals)]
            il = [torch.mean(torch.cat(im)).unsqueeze(0) for im in intermediate_losses]
            BT_rwd_loss = torch.cat(il).mean()
            rwd_vals_for_ret_BT = torch.vstack(BT_vals).flatten().detach()
            BT_rwd_loss = BT_rwd_loss * self.hparams.loss["lambda_BT"]
            rwds_dist_dict_BT = {s: v for s, v in zip(CPU_SEEDS, rwd_vals_for_ret_BT.flatten().detach())}
            BT_vals = [b.detach() for b in BT_vals]
            BT_comparison_values = [torch.hstack([(cp[o[0]] / (cp[o[0]] + cp[o[1]])) for o in oc]) for oc, cp in zip(ordered_combos, BT_vals)]

        if return_preds:
            preds = dict(
                dict_of_global_feature=dict_of_global_feature,
                pred_logits=return_for_logits,
                seeds=return_for_seeds,
                sum_in_batch=sum_in_batch,
                rwd_vals_scalar=rwd_vals_for_ret_scalar,
                sigmoid_comparison_values=sigmoid_comparison_values,
                scalar_rwds_dist_dict_scalar=scalar_rwds_dist_dict_scalar,
                BT_comparison_values=BT_comparison_values,
                rwd_vals_BT=rwd_vals_for_ret_BT,
                rwds_dist_dict_BT=rwds_dist_dict_BT,
                heatmap_images=heatmap_images,
            )
        else:
            preds = {}
        preds.update(dict(sum_in_batch=sum_in_batch))

        l2_reg_lambda = torch.tensor(0.0, device=device)
        if self.hparams.loss["lambda_BT"] != 0.0 and self.hparams.loss["lambda_reg_rwd_vals"] != 0.0:
            l2_reg_lambda = torch.mean(torch.norm(rwd_vals_for_ret_BT.flatten())) * self.hparams.loss["lambda_reg_rwd_vals"]

        abs_val_loss = torch.tensor(0.0, device=device)
        l2_reg_agg_features = torch.tensor(0.0, device=device)

        losses = dict(
            paired_loss=paired_loss.detach(),
            BT_rwd_loss=BT_rwd_loss.detach(),
            abs_val_loss=abs_val_loss.detach(),
            scalar_rwd_loss=scalar_rwd_loss.detach(),
            l2_reg_lambda=l2_reg_lambda.detach(),
            l2_reg_agg_features=l2_reg_agg_features.detach(),
            total_loss=paired_loss + BT_rwd_loss + abs_val_loss + scalar_rwd_loss + l2_reg_lambda + l2_reg_agg_features,
        )
        return (losses, preds)


class sigma_rays_3dconv(UniversalRWDModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        embed_dim = kwargs["embed_dim"]
        num_heads = kwargs["num_heads"]
        cross_attention_dropout = kwargs["cross_attention_dropout"]
        encoder_dropout = kwargs["encoder_dropout"]
        self_attention_dropout = kwargs["self_attention_dropout"]
        self.global_feature_size = kwargs["global_feature_size"]
        self.n_3d_fmaps = kwargs["n_3d_fmaps"]
        self.encoder = SequenceEncoderWithFourierSigma(input_dim=128, embed_dim=128, num_heads=8, forward_expansion=4, dropout=0.0, max_length=1024, num_features=1)
        self.self_attention = TransformerGPT(embed_size=128, heads=4, depth=4, forward_expansion=4, dropout=self_attention_dropout)
        self.cross_attention = CrossAttentionModule(embed_dim=128, num_heads=num_heads, cross_attention_dropout=cross_attention_dropout)

        self.dp3d = nn.Dropout3d(p=0.15)
        self.conv1 = nn.Conv3d(1, self.n_3d_fmaps, kernel_size=[5, 7, 7], bias=True)
        self.norm1 = nn.BatchNorm3d(self.n_3d_fmaps)
        self.conv2 = nn.Conv1d(self.n_3d_fmaps, self.n_3d_fmaps * 4, 4)
        self.nmp = nn.MaxPool1d(131)
        self.meanp = torch.nn.AvgPool1d(131)

        self.attn_msk_list = None
        self.n_kpts = kwargs["n_kpts"]
        self.MINSCALE = kwargs.get("MINSCALE", 1.0)

        self.remove_MLPs()
        self.comparator = hydra.utils.instantiate(kwargs["mlp_transformer"])
        self.sm = torch.nn.Softmax(1)

        self.external = None
        self.external_init_dict = kwargs["external"]
        if self.external_init_dict is not None:
            self.set_external()

        self.hparams.optimizer = kwargs["optimizer"]
        if "scheduler" in kwargs.keys():
            self.hparams.scheduler = kwargs["scheduler"]

        self.train_step = 0
        self.setup_opt_sched_init()

        self.dict_of_regions = {
            "LHS_face_idx": list(range(0, 16)),
            "nose_idx": [54],
            "RHS_face_idx": list(range(17, 33)),
            "mouth_face_idx": list(range(76, 96)),
            "nose_ridge_idx": list(range(51, 55)),
            "left_eye_idx": list(range(60, 68)) + [97],
            "right_eye_idx": list(range(60, 68)) + [96],
            "septum_idx": list(range(55, 59)),
            "left_brow_idx": list(range(33, 42)),
            "right_brow_idx": list(range(42, 51)),
            "all_idx": list(range(98)),
        }
        sel_regions = ["LHS_face_idx", "RHS_face_idx", "nose_ridge_idx", "nose_ridge_idx", "nose_idx", "nose_idx", "left_eye_idx", "right_eye_idx"]
        attn_msk_dict = {}
        for s in sel_regions:
            template_msk = torch.ones((self.n_kpts, self.n_kpts)) - 1.0
            msk = self.dict_of_regions[s]
            for row in msk:
                for col in msk:
                    template_msk[row, col] = 0.0
            attn_msk_dict[s] = template_msk
        self.attn_msk_dict = attn_msk_dict
        self.sel_region_for_masking = kwargs.get("sel_region_for_masking")

    def remove_MLPs(self):
        for attr_l in ["MLP", "scalar_rwd_head_BT", "scalar_rwd_head", "scalar_rwd_head_pairs"]:
            if hasattr(self, attr_l):
                delattr(self, attr_l)
        return self

    def forward_to_global_feature_vec(self, x):
        if len(x.shape) == 5:
            xs1, xs2, xs3, xs4, xs5 = x.shape
            x = x.reshape(xs1 * xs2, xs3, xs4, xs5)
        embds_for_in = self.cutout_nose_region(x)
        embds_for_in = torch.cat([self.random_pad_input(cv).unsqueeze(0) for cv in embds_for_in], 0)
        embds_for_in[embds_for_in == -1] = 0.0
        x = embds_for_in + torch.randn_like(embds_for_in) * 1e-2

        x1 = self.dp3d(F.relu(self.norm1(self.conv1(x.unsqueeze(1))))).squeeze(-1, -2)
        x2 = self.conv2(x1)
        feature_vec = self.encoder(x2)
        encoded_seq = self.self_attention(feature_vec)
        return encoded_seq

    def on_before_optimizer_step(self, optimizer: optim.Optimizer) -> None:
        self.train_step = self.train_step + 1
        return super().on_before_optimizer_step(optimizer)

    def calc_scaling_factor(self, keys, just_scaling_factor=False):
        if not self.training:
            scaling_factor = self.MINSCALE
        else:
            total_stepping_batches = self.trainer.estimated_stepping_batches
            current_step = self.train_step
            scaling_factor = max(1 - current_step * 1.0 / total_stepping_batches, self.MINSCALE)
        if just_scaling_factor:
            return scaling_factor
        scale = torch.ones_like(keys)
        if self.sel_region_for_masking is not None:
            s = self.sel_region_for_masking
            msk = self.dict_of_regions[s]
            keys_to_scale = [i for i in torch.arange(keys.shape[1]) if i not in msk]
            scale[:, keys_to_scale, :] *= scaling_factor
        else:
            msk = scaling_factor + (1 - scaling_factor) * get_gaussian_msk_centered()
            mskf = msk.flatten().to(keys.device)
            mskf = mskf[None, :, None]
            mskf = mskf.expand(*keys.shape)
            scale = mskf
        return scale

    def random_pad_input(self, cv3_input, pad_start_only=True):
        rays_start = torch.randint(low=0, high=7, size=(1,), device=cv3_input.device)
        pad = torch.zeros_like(cv3_input)[:7, :, :] - 1
        if rays_start == 0 or pad_start_only:
            cv3_input_padded = torch.cat((cv3_input, pad), dim=0)
        elif rays_start == 7:
            cv3_input_padded = torch.cat((pad, cv3_input), dim=0)
        else:
            pad_start, pad_end = torch.split(pad, [rays_start, 7 - rays_start], dim=0)
            cv3_input_padded = torch.cat((pad_start, cv3_input, pad_end), dim=0)
        return cv3_input_padded

    def forward_from_cat_global_vectors(self, enc_seq1, enc_seq2, with_softmax=False, attn_mask=None):
        enc_seq1_self = enc_seq1
        enc_seq2_self = enc_seq2
        if attn_mask is not None:
            attn_seq1_seq2 = self.cross_attention.forward(enc_seq1_self, enc_seq2_self, enc_seq2_self, attn_mask=attn_mask.squeeze(0).to(self.device))
            attn_seq2_seq1 = self.cross_attention.forward(enc_seq2_self, enc_seq1_self, enc_seq1_self, attn_mask=attn_mask.squeeze(0).to(self.device))
        else:
            attn_seq1_seq2 = self.cross_attention.forward(enc_seq1_self, enc_seq2_self, enc_seq2_self)
            attn_seq2_seq1 = self.cross_attention.forward(enc_seq2_self, enc_seq1_self, enc_seq1_self)
        combined_features_seq1_seq2 = torch.cat((attn_seq1_seq2, attn_seq2_seq1), dim=-1)
        combined_features_seq2_seq1 = torch.cat((attn_seq2_seq1, attn_seq1_seq2), dim=-1)
        combined_features_seq1_seq2 = torch.mean(combined_features_seq1_seq2, dim=1)
        combined_features_seq2_seq1 = torch.mean(combined_features_seq2_seq1, dim=1)
        logits_seq1_seq2 = self.comparator(combined_features_seq1_seq2)
        logits_seq2_seq1 = self.comparator(combined_features_seq2_seq1)
        logits_seq2_seq1_rev = logits_seq2_seq1.flip(-1)
        logits = (logits_seq1_seq2 + logits_seq2_seq1_rev) / 2
        if with_softmax:
            logits = self.sm(logits)
        return logits

    def get_activation_maps(self, seq1, seq2, attn_mask=None):
        enc_seq1 = self.encoder(seq1)
        enc_seq2 = self.encoder(seq2)
        key_scale = self.calc_scaling_factor(enc_seq2)
        attn_seq1_maps = self.cross_attention.get_attention_maps(enc_seq1, enc_seq2, enc_seq2, keyscale=key_scale)
        attn_seq2_maps = self.cross_attention.get_attention_maps(enc_seq2, enc_seq1, enc_seq1, keyscale=key_scale)
        return dict(attn_seq1_maps=attn_seq1_maps, attn_seq2_maps=attn_seq2_maps)

    def get_activation_maps_from_embedded(self, enc_seq1, enc_seq2, attn_mask=None):
        key_scale = self.calc_scaling_factor(enc_seq2)
        attn_seq1_maps = self.cross_attention.get_attention_maps(enc_seq1, enc_seq2, enc_seq2, keyscale=key_scale)
        attn_seq2_maps = self.cross_attention.get_attention_maps(enc_seq2, enc_seq1, enc_seq1, keyscale=key_scale)
        return dict(attn_seq1_maps=attn_seq1_maps, attn_seq2_maps=attn_seq2_maps)

    def forward_to_scalar_reward_from_single_global(self, x):
        return super().forward_to_scalar_reward_from_single_global(x)

    def forward_to_BT_lambda_from_single_global(self, x, mult=1):
        return super().forward_to_BT_lambda_from_single_global(x, mult)

    def return_windowed_attention_mask(self, seq_len=128, window_size=7):
        assert window_size % 2 == 1, "error need to have odd for window size"
        b = torch.ones((seq_len, seq_len))
        win_half = int(window_size / 2)
        win_at = torch.triu(b, diagonal=-win_half) * torch.tril(b, diagonal=win_half)
        return win_at

    def cutout_nose_region(self, embds_for_in):
        X = Y = 63
        OFF = 3
        xsqu3d = embds_for_in[:, X - OFF : X + OFF + 1, Y - OFF : Y + OFF + 1, :]
        cv3_input = xsqu3d.permute(0, 3, 1, 2)
        return cv3_input


__all__ = ["sigma_rays_transformer", "sigma_rays_3dconv", "get_gaussian_msk_centered"]
