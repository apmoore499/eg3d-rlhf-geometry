import hydra
import torch
from torch import optim

from core_modules.models.base import UniversalRWDModel, log
from core_modules.models.utils_transformer import CrossAttentionModule, SequenceEncoderWithFourier, TransformerGPT


class aw98_3d_lmks_MLP(UniversalRWDModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.external = None
        self.external_init_dict = kwargs["external"]

        if self.external_init_dict is not None:
            self.set_external()

    def forward_to_global_feature_vec(self, x):
        feature_vec = self.MLP(x.squeeze(1).flatten(start_dim=1))  # (B,98,C) -> (B,98*C) for the MLP
        return feature_vec


class aw98_2d_lmks_MLP(UniversalRWDModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.external = None
        self.external_init_dict = kwargs["external"]

        if self.external_init_dict is not None:
            self.set_external()

    def forward_to_global_feature_vec(self, x):
        feature_vec = self.MLP(x.squeeze(1).flatten(start_dim=1))  # (B,98,C) -> (B,98*C) for the MLP
        return feature_vec


class aw98_transformer(UniversalRWDModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # del self.MLP #for our model, the self.MLP is actually the self.encoder object of the class instance

        embed_dim = kwargs["embed_dim"]  # 512
        num_heads = kwargs["num_heads"]
        cross_attention_dropout = kwargs["cross_attention_dropout"]
        encoder_dropout = kwargs["encoder_dropout"]
        n_kpts = kwargs["n_kpts"]

        self.encoder = SequenceEncoderWithFourier(input_dim=3, embed_dim=embed_dim, num_heads=num_heads, forward_expansion=4, dropout=encoder_dropout, max_length=n_kpts, num_features=embed_dim)

        self.self_attention = TransformerGPT()
        self.cross_attention = CrossAttentionModule(embed_dim=embed_dim, num_heads=num_heads, cross_attention_dropout=cross_attention_dropout)
        self.attn_msk_list = None
        self.n_kpts = kwargs["n_kpts"]
        self.MINSCALE = 1.0

        if "MINSCALE" in kwargs.keys():
            self.MINSCALE = kwargs["MINSCALE"]  # MINSCALE FOR THROTTLING THE ATTENTION WEIGHTS TOWARDS CENTER

        self.remove_MLPs()

        self.comparator = hydra.utils.instantiate(kwargs["mlp_transformer"])
        self.sm = torch.nn.Softmax(1)  # i think should be 1 or 0, check

        self.external = None
        self.external_init_dict = kwargs["external"]  # should be omegaconf...

        # self.external = hydra.utils.instantiate(self.external_init_dict)

        if self.external_init_dict is not None:
            self.set_external()

        self.hparams.optimizer = kwargs["optimizer"]  # ,convert='partial')
        if "scheduler" in kwargs.keys():
            self.hparams.scheduler = kwargs["scheduler"]  # ,convert='partial')

        self.train_step = 0  # use this as a step counter isntead of relying on global step

        self.setup_opt_sched_init()  # global feature embedding i think might need to be changed but let's just see first

        self.dict_of_regions = {
            "LHS_face_idx": list(range(0, 16)),  # 0 to 15
            "nose_idx": [54],  # Single value, no change needed
            "RHS_face_idx": list(range(17, 33)),  # 17 to 32
            "mouth_face_idx": list(range(76, 96)),  # 76 to 95
            "nose_ridge_idx": list(range(51, 55)),  # 51 to 54
            "left_eye_idx": list(range(60, 68)) + [97],  # 60 to 67, plus 97
            "right_eye_idx": list(range(60, 68)) + [96],  # 60 to 67, plus 96
            "septum_idx": list(range(55, 59)),  # 55 to 58
            "left_brow_idx": list(range(33, 42)),  # 33 to 41
            "right_brow_idx": list(range(42, 51)),  # 42 to 50
            "all_idx": list(range(98)),  # 42 to 50
        }

        sel_regions = ["LHS_face_idx", "RHS_face_idx", "nose_ridge_idx", "nose_ridge_idx", "nose_idx", "nose_idx", "left_eye_idx", "right_eye_idx"]

        # msk_list=[self.attn_mask_template]*8
        attn_msk_dict = {}
        for s in sel_regions:
            template_msk = torch.ones((self.n_kpts, self.n_kpts)) - 1.0  # try attention weight 1.0? not sure
            msk = self.dict_of_regions[s]

            for row in msk:
                for col in msk:
                    template_msk[row, col] = 0.0

            attn_msk_dict[s] = template_msk

        self.attn_msk_dict = attn_msk_dict

        self.sel_region_for_masking = None
        if kwargs["sel_region_for_masking"] is not None:
            self.sel_region_for_masking = kwargs["sel_region_for_masking"]

    def remove_MLPs(self, mlp_list=["MLP", "scalar_rwd_head_pairs"]):
        for attr_l in mlp_list:
            if hasattr(self, attr_l):
                delattr(self, attr_l)

        return self

    def forward_to_global_feature_vec(self, x):
        feature_vec = self.encoder(x)

        encoded_seq = self.self_attention(feature_vec)

        return encoded_seq

    def on_validation_start(self) -> None:
        atn_scale = self.calc_scaling_factor(torch.tensor([1.0]).unsqueeze(0), just_scaling_factor=True)  # $(1,1))#$(1,1))

        log.info(f"On Val Start: atn key scale (VAL): {atn_scale} ")
        return super().on_validation_start()

    def on_test_start(self) -> None:
        atn_scale = self.calc_scaling_factor(torch.tensor([1.0]).unsqueeze(0), just_scaling_factor=True)  # $(1,1))

        log.info(f"On Test Start: atn key scale (TEST): {atn_scale} ")

        return super().on_test_start()

    def on_before_optimizer_step(self, optimizer: optim.Optimizer, *args) -> None:
        self.train_step = self.train_step + 1  # increment the train step
        return super().on_before_optimizer_step(optimizer, *args)

    def calc_scaling_factor(self, keys, just_scaling_factor=False):
        if not self.training:
            scaling_factor = self.MINSCALE

        else:
            # training..........
            total_stepping_batches = self.trainer.estimated_stepping_batches  # *(self.trainer.num_training_batches-self.trainer.num_val_batches)
            current_step = self.train_step
            scaling_factor = max(1 - current_step * 1.0 / total_stepping_batches, self.MINSCALE)

        if just_scaling_factor:
            return scaling_factor
        # return(scaling_factor)

        scale = torch.ones_like(keys)

        if self.sel_region_for_masking is not None:
            s = self.sel_region_for_masking
            msk = self.dict_of_regions[s]

            keys_to_scale = [i for i in torch.arange(keys.shape[1]) if i not in msk]  # will select all keys not

            scale[:, keys_to_scale, :] *= scaling_factor

        else:
            msk = scaling_factor + (1 - scaling_factor) * torch.ones((1))  # *get_gaussian_msk_centered() dont' know why gaussian mask originaly required. removed

            mskf = msk.flatten().to(keys.device)

            mskf = mskf[None, :, None]

            mskf = mskf.expand(*keys.shape)

            scale = mskf

        return scale

    def forward_from_cat_global_vectors(self, enc_seq1, enc_seq2, with_softmax=False, attn_mask=None):
        key_scale = self.calc_scaling_factor(enc_seq2)
        # enc_seq2=enc_seq2*key_scale

        # attention ramp?
        #

        enc_seq1_self = enc_seq1  # self.self_attention(enc_seq1)
        enc_seq2_self = enc_seq2  # self.self_attention(enc_seq2)

        attn_seq1_seq2 = self.cross_attention.forward(enc_seq1_self, enc_seq2_self, enc_seq2_self, keyscale=key_scale)  # ,attn_mask=batch_
        attn_seq2_seq1 = self.cross_attention.forward(enc_seq2_self, enc_seq1_self, enc_seq1_self, keyscale=key_scale)  # ,attn_mask=batch_msk)

        # Combine and process features for both orders
        combined_features_seq1_seq2 = torch.cat((attn_seq1_seq2, attn_seq2_seq1), dim=-1)
        combined_features_seq2_seq1 = torch.cat((attn_seq2_seq1, attn_seq1_seq2), dim=-1)  # Note the reversed order

        combined_features_seq1_seq2 = torch.mean(combined_features_seq1_seq2, dim=1)  # Reduce over sequence length
        combined_features_seq2_seq1 = torch.mean(combined_features_seq2_seq1, dim=1)  # Reduce over sequence length

        # Compute probabilities for both sequences
        logits_seq1_seq2 = self.comparator(combined_features_seq1_seq2)
        logits_seq2_seq1 = self.comparator(combined_features_seq2_seq1)  # .flip(-1)
        logits_seq2_seq1_rev = logits_seq2_seq1.flip(-1)

        # then average the logits? but reverse second?

        logits = (logits_seq1_seq2 + logits_seq2_seq1_rev) / 2
        # logits=logits_seq1_seq2

        if with_softmax:
            logits = self.sm(logits)
        return logits

    def get_activation_maps(self, seq1, seq2, attn_mask=None):
        enc_seq1 = self.encoder(seq1)
        enc_seq2 = self.encoder(seq2)

        key_scale = self.calc_scaling_factor(enc_seq2)

        # Cross-attention where each sequence attends to the other
        attn_seq1_maps = self.cross_attention.get_attention_maps(enc_seq1, enc_seq2, enc_seq2, keyscale=key_scale)  # ,attn_mask=batch_msk)
        attn_seq2_maps = self.cross_attention.get_attention_maps(enc_seq2, enc_seq1, enc_seq1, keyscale=key_scale)  # ,attn_mask=batch_msk)

        return dict(attn_seq1_maps=attn_seq1_maps, attn_seq2_maps=attn_seq2_maps)

    def get_activation_maps_from_embedded(self, enc_seq1, enc_seq2, attn_mask=None):
        key_scale = self.calc_scaling_factor(enc_seq2)

        # Cross-attention where each sequence attends to the other
        attn_seq1_maps = self.cross_attention.get_attention_maps(enc_seq1, enc_seq2, enc_seq2, keyscale=key_scale)  # ,attn_mask=batch_msk)
        attn_seq2_maps = self.cross_attention.get_attention_maps(enc_seq2, enc_seq1, enc_seq1, keyscale=key_scale)  # ,attn_mask=batch_msk)

        return dict(attn_seq1_maps=attn_seq1_maps, attn_seq2_maps=attn_seq2_maps)

    def forward_to_scalar_reward_from_single_global(self, x):
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], -1)
        return super().forward_to_scalar_reward_from_single_global(x)

    def forward_to_BT_lambda_from_single_global(self, x, mult=1):
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], -1)
        return super().forward_to_BT_lambda_from_single_global(x, mult)


__all__ = ["aw98_3d_lmks_MLP", "aw98_2d_lmks_MLP", "aw98_transformer"]
