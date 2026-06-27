"""
Custom data transforms used for sigma volumes, depth maps, and RGB data.
Note: Several transforms are legacy/experimental; common usage centers on the
composition helpers and sigma normalization/clamping.
"""

from typing import List
import autoroot  # noqa: F401
import kornia
import monai
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# read in a sample input....
from monai.networks.layers import median_filter
from torchvision.transforms import v2


# ROOT CLASS THAT IS USED TO COMPOSE ALL OTHER TRANSFORMS
class transforms_composition_helper(nn.Module):
    def __init__(self, instantiated_transforms, dtype, **kwargs):
        super().__init__()

        # print(instantiated_transforms)

        self.dtype = dtype
        list_of_transforms = []

        for _, transform in instantiated_transforms.items():
            list_of_transforms.append(transform)

        transforms = v2.Compose(list_of_transforms)
        self.transforms = transforms

    def forward(self, x):
        return self.transforms(x)


# ROOT CLASS THAT IS USED TO COMPOSE ALL OTHER TRANSFORMS
class transforms_composition_helper_random(nn.Module):
    def __init__(self, instantiated_transforms, dtype, p_apply=0.5, **kwargs):
        super().__init__()

        # print(instantiated_transforms)

        self.dtype = dtype

        self.p_apply = p_apply
        self.instantiated_transforms = instantiated_transforms

        # for _,transform in instantiated_transforms.items():
        # self.list_of_transforms.append(transform)

        # transforms = v2.Compose(list_of_transforms)
        # self.transforms=transforms

    def forward(self, x):
        for k, t in enumerate(self.instantiated_transforms.keys()):
            p_t = self.p_apply[k]
            if p_t == 1:
                x = self.instantiated_transforms[t](x)
            elif p_t == 0:
                continue
            else:
                rsel = np.random.rand()
                if rsel > p_t:
                    x = self.instantiated_transforms[t](x)

        return x


# --------------------------------------------------------------------------------
# FOR 3D SIGMA VALS OVER 3D VOLUME
# Note: Many of the sigma_* transforms below are legacy/experimental and are rarely
# used in current pipelines. They are kept for backward compatibility; prefer
# adding new transforms near the top-level helpers if needed.


class clamp_sigma(nn.Module):
    def __init__(self, min_clamp=None, max_clamp=None):
        super().__init__()

        self.min = min_clamp
        self.max = max_clamp

    def forward(self, x):
        x = torch.clamp(input=x, min=self.min, max=self.max)
        return x


class clamp_scale_sigma_fixed(nn.Module):
    def __init__(self, min_clamp=0.0, max_clamp=100.0, out_min=0.0, out_max=100.0):
        super().__init__()

        self.min_clamp = min_clamp
        self.max_clamp = max_clamp
        self.out_min = out_min
        self.out_max = out_max

        assert self.max_clamp > self.min_clamp, "error max_clamp <= min_clamp for clamp_scale_sigma_fixed"
        assert self.out_max > self.out_min, "error out_max <= out_min for clamp_scale_sigma_fixed"

    def forward(self, x):
        x = torch.clamp(input=x, min=self.min_clamp, max=self.max_clamp)
        x = (x - self.min_clamp) / (self.max_clamp - self.min_clamp)
        x = x * (self.out_max - self.out_min) + self.out_min
        return x


# helper module just to passs thru input...
class identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# normalises sigma to itself, doesn't care about absolute values
class normalise_sigma_self(nn.Module):
    def __init__(self, min_norm=0.0, max_norm=1.0):
        super().__init__()

        self.min = min_norm
        self.max = max_norm

        assert self.max > self.min, "error min >= max for normalise_sigma transform"
        # assert self.max_sigma>self.min_sigma, 'error min_sigma >= max_sigma for normalise_sigma transform'

    def forward(self, x):
        return (x - x.min()) / (x.max() - x.min()) * (self.max - self.min) + self.min


# normalise sigma to within some range scalar values min/max
class normalise_sigma(nn.Module):
    def __init__(self, min_norm=0.0, max_norm=1.0, min_sigma=0.0, max_sigma=125.0):
        super().__init__()

        self.min = min_norm
        self.max = max_norm
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        assert self.max > self.min, "error min >= max for normalise_sigma transform"
        assert self.max_sigma > self.min_sigma, "error min_sigma >= max_sigma for normalise_sigma transform"

    def forward(self, x):
        # x=(torch.max(x)-x)/(torch.max(x)-torch.min(x))*(self.max-self.min)+self.min

        x = (x - self.min_sigma) / (self.max_sigma - self.min_sigma) * (self.max - self.min) + self.min

        return x

    # class sigma_compress_and_jitter (removed as legacy)(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        nose_start = x[:, :, :20]
        nose_mid = x[:, :, 20:-20]
        nose_end = x[:, :, -20:]

        rand_offset_start = torch.randint(low=0, high=40, size=(1, 1))
        rand_offset_end = torch.randint(low=0, high=40, size=(1, 1))

        old_shape = list(x.shape)

        X, Y, Z = old_shape

        nstart = (
            torch.nn.functional.interpolate(
                nose_start.unsqueeze(0).unsqueeze(0),
                size=(X, Y, 20 + rand_offset_start),
            )
            .squeeze(0)
            .squeeze(0)
        )
        nmid = (
            torch.nn.functional.interpolate(
                nose_mid.unsqueeze(0).unsqueeze(0),
                size=(X, Y, Z - 20 - 20 - rand_offset_start - rand_offset_end),
            )
            .squeeze(0)
            .squeeze(0)
        )
        nend = torch.nn.functional.interpolate(nose_end.unsqueeze(0).unsqueeze(0), size=(X, Y, 20 + rand_offset_end)).squeeze(0).squeeze(0)

        x = torch.cat((nstart, nmid, nend), -1)

        return x


class sigma_extend_nose(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # make work with any shape
        # we asusme input is only [X,Y,Z] and Z is the dimension along which we want to do some perturbations

        old_shape = list(x.shape)

        X, Y, Z = old_shape

        # this essentially defines the padding transform
        nose_mid = x[:, :, 20:-20]
        nose_end = x[:, :, -20:]

        # try concat it along the z dim....
        rand_offset = torch.randint(low=2, high=38, size=(1, 1))

        nmid = torch.nn.functional.interpolate(nose_mid.unsqueeze(0).unsqueeze(0), size=(X, Y, Z - 20 - 20)).squeeze(0).squeeze(0)
        tmm = torch.sum((nmid.flatten(0, 1) > 20.0).to(torch.float), 0)
        first_idx = torch.where(torch.flip(tmm, dims=[0]))[0][0]
        first_idx = tmm.shape[0] - first_idx - 15  # 5 for cutoff...

        new_start = nmid[:, :, :first_idx]
        rand_offset = torch.randint(low=10, high=30, size=(1, 1))  # .cuda()
        new_mid = nmid[:, :, first_idx : first_idx + rand_offset]

        new_mid = (
            torch.nn.functional.interpolate(
                new_mid.unsqueeze(0).unsqueeze(0),
                size=(X, Y, 154 - new_start.shape[-1] - rand_offset),
            )
            .squeeze(0)
            .squeeze(0)
        )
        new_end = torch.nn.functional.interpolate(nose_end.unsqueeze(0).unsqueeze(0), size=(X, Y, rand_offset)).squeeze(0).squeeze(0)

        x = torch.cat((new_start, new_mid, new_end), -1)  # .shape
        return x


class sigma_jitter_depth(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        old_shape = list(x.shape)

        X, Y, Z = old_shape

        nose_start = x[:, :, :20]
        nose_mid = x[:, :, 20:-20]
        nose_end = x[:, :, -20:]

        # try concat it along the z dim....
        rand_offset = torch.randint(low=2, high=38, size=(1, 1))

        nstart = torch.nn.functional.interpolate(nose_start.unsqueeze(0).unsqueeze(0), size=(X, Y, rand_offset)).squeeze(0).squeeze(0)
        nmid = torch.nn.functional.interpolate(nose_mid.unsqueeze(0).unsqueeze(0), size=(X, Y, Z - 20 - 20)).squeeze(0).squeeze(0)
        nend = torch.nn.functional.interpolate(nose_end.unsqueeze(0).unsqueeze(0), size=(X, Y, 40 - rand_offset)).squeeze(0).squeeze(0)

        x = torch.cat((nstart, nmid, nend), -1)

        return x


class threshold_ray_sigma_after_surface(nn.Module):
    def __init__(self, threshold=20.0):
        super().__init__()

        self.threshold = threshold

    def forward(self, x):
        x_for_thresh = x.detach().cpu().numpy()[:, :, ::-1]  # >=threshold # works
        x_for_thresh[x_for_thresh >= self.threshold] = self.threshold
        accumulated = np.maximum.accumulate(x_for_thresh, axis=-1)[:, :, ::-1]  # rev ti be orig otder...
        x[accumulated >= self.threshold] = self.threshold

        return x


class pad_sigma_before_and_after_random(nn.Module):
    def __init__(self, max_len=40):
        super().__init__()

        self.len = max_len

    def forward(self, x):
        offset = torch.randint(-self.len, self.len, size=(1, 1)).flatten()

        if offset > 0:
            x = torch.cat((x[:, :, offset:], torch.zeros_like(x)[:, :, :offset]), -1)

        elif offset < 0:
            x = torch.cat((torch.ones_like(x)[:, :, :-offset], x[:, :, :offset]), -1)

        return x


class remove_bg_to_zeros(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1].item())
        x[:, x.min(0)[0] == 1.0] = 0.0
        x = x.view(orig_shape)

        return x


class scale_over_threshold(nn.Module):
    def __init__(self, threshold=100.0, upper_lim=125.0):
        super().__init__()
        self.threshold = threshold
        self.upper_lim = upper_lim

    def forward(self, x):
        mask = (x > self.threshold).to(torch.float)  # use as binary mask to select only elements > threshold
        mask_under = 1 - mask
        x_unmasked = x * mask_under

        x_masked = x * mask
        x_masked = kornia.enhance.normalize_min_max(x_masked, min_val=self.threshold, max_val=self.upper_lim) * mask  # multiply by mask to cancel out elements that should be zero
        x = x_unmasked + x_masked

        return x


# @torch.jit.script
# on a type of sigma_field_256 data
class ensemble_mouth_eyes_nose_smooth(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, run_cfg_kwargs=None):
        # -------------------------------------------------------------------

        run_cfg_default = self.get_default_kwargs()

        if run_cfg_kwargs is not None:
            run_cfg_default.update(run_cfg_kwargs)

        rcfg = run_cfg_default

        WITH_NOSE = rcfg["FILTER_NOSE"]
        WITH_LEFT_EYE = rcfg["FILTER_LEFT_EYE"]
        WITH_RIGHT_EYE = rcfg["FILTER_RIGHT_EYE"]
        WITH_MOUTH = rcfg["FILTER_MOUTH"]
        WITH_FINAL_MEDIAN = rcfg["FINAL_MEDIAN"]

        sf_orig = x  # .clone()

        # -------------------------------------------------------------------

        # NOSE FILTERING
        sfs = torch.tensor(list(x.shape))
        hf = (sfs / 2).to(torch.int)
        pcx = 0.18
        pcy = 0.35
        output_nose = sf_orig[
            hf[0] - int(pcx * hf[0]) : hf[0] + int(pcx * hf[0]),
            hf[0] - int(pcy * hf[1]) : hf[0] + int(pcy * hf[1]),
            64:,
        ]

        # LEFT EYE
        hf = sfs.to(torch.int)
        hf[0] = hf[0] * 0.65
        hf[1] = hf[1] * 0.35
        pcx = 0.1
        pcy = 0.15
        output_eyes_l = sf_orig[
            hf[0] - int(pcx * hf[0]) : hf[0] + int(pcx * hf[0]),
            hf[0] - int(pcy * hf[1]) : hf[0] + int(pcy * hf[1]),
            32:,
        ]

        # RIGHT EYE
        hf[1] = hf[0]
        hf[0] = sfs[0] - hf[0]
        pcx = 0.17
        pcy = 0.09
        output_eyes_r = sf_orig[
            hf[0] - int(pcx * hf[0]) : hf[0] + int(pcx * hf[0]),
            hf[1] - int(pcy * hf[1]) : hf[1] + int(pcy * hf[1]),
            32:,
        ]

        # MOUTH
        sfs = torch.tensor(list(x.shape))
        hf = (sfs / 2).to(torch.int)
        pcx = 0.40
        pcy = 0.26
        hf[1] = sfs[1] - hf[1] * 1.5
        output_mouth = sf_orig[
            hf[0] - int(pcx * hf[0]) : hf[0] + int(pcx * hf[0]),
            hf[1] - int(pcy * hf[1]) : hf[1] + int(pcy * hf[1]),
            64:,
        ]

        if WITH_NOSE:
            sfs = torch.tensor(list(x.shape))
            hf = (sfs / 2).to(torch.int)
            pcx = 0.18
            pcy = 0.35

            output_nose = median_filter(output_nose, (5, 5, 9))  # this one is best, but its very strong
            output_nose = median_filter(output_nose, (3, 3, 5))  # this one is best, but its very strong
            x[
                hf[0] - int(pcx * hf[0]) : hf[0] + int(pcx * hf[0]),
                hf[0] - int(pcy * hf[1]) : hf[0] + int(pcy * hf[1]),
                64:,
            ] = output_nose  # [hf[0]-int(pcx*hf[0]):hf[0]+int(pcx*hf[0]),hf[0]-int(pcy*hf[1]):hf[0]+int(pcy*hf[1]),64:] #actually last dim is Z, but ned to be reversed

        # -------------------------------------------------------------------
        if WITH_LEFT_EYE:
            hf = sfs.to(torch.int)
            hf[0] = hf[0] * 0.65
            hf[1] = hf[1] * 0.35
            pcx = 0.1
            pcy = 0.15
            output_eyes_l = median_filter(output_eyes_l, (3, 3, 3))  # this one is best, but its very strong
            output_eyes_l = median_filter(output_eyes_l, (3, 3, 5))  # this one is best, but its very strong
            x[
                hf[0] - int(pcx * hf[0]) : hf[0] + int(pcx * hf[0]),
                hf[0] - int(pcy * hf[1]) : hf[0] + int(pcy * hf[1]),
                32:,
            ] = output_eyes_l  # [hf[0]-int(pcx*hf[0]):hf[0]+int(pcx*hf[0]),hf[0]-int(pcy*hf[1]):hf[0]+int(pcy*hf[1]),32:]

        # RIGHT EYE
        if WITH_RIGHT_EYE:
            # RIGHT EYE
            hf[1] = hf[0]
            hf[0] = sfs[0] - hf[0]
            pcx = 0.17
            pcy = 0.09
            output_eyes_r = median_filter(output_eyes_r, (3, 3, 3))  # this one is best, but its very strong
            output_eyes_r = median_filter(output_eyes_r, (3, 3, 5))  # this one is best, but its very strong
            x[
                hf[0] - int(pcx * hf[0]) : hf[0] + int(pcx * hf[0]),
                hf[1] - int(pcy * hf[1]) : hf[1] + int(pcy * hf[1]),
                32:,
            ] = output_eyes_r  # [hf[0]-int(pcx*hf[0]):hf[0]+int(pcx*hf[0]),hf[1]-int(pcy*hf[1]):hf[1]+int(pcy*hf[1]),32:]

        # -------------------------------------
        # MOUTH
        if WITH_MOUTH:
            sfs = torch.tensor(list(x.shape))
            hf = (sfs / 2).to(torch.int)
            pcx = 0.40
            pcy = 0.26
            hf[1] = sfs[1] - hf[1] * 1.5

            output_mouth = median_filter(output_mouth, (3, 3, 3))  # this one is best, but its very strong
            x[
                hf[0] - int(pcx * hf[0]) : hf[0] + int(pcx * hf[0]),
                hf[1] - int(pcy * hf[1]) : hf[1] + int(pcy * hf[1]),
                64:,
            ] = output_mouth  # actually last dim is Z, but ned to be reversed

        if WITH_FINAL_MEDIAN:
            x = median_filter(x, (3, 3, 3))  # this one is best, but its very strong

        return x

    def get_default_kwargs(self):
        run_cfg_default = dict(
            FILTER_NOSE=True,
            FILTER_LEFT_EYE=True,
            FILTER_RIGHT_EYE=True,
            FILTER_MOUTH=True,
            FINAL_MEDIAN=True,
        )

        return run_cfg_default


class smooth_nose_only(ensemble_mouth_eyes_nose_smooth):
    def __init__(self):
        super().__init__()

    def get_default_kwargs(self):
        run_cfg_default = dict(
            FILTER_NOSE=True,
            FILTER_LEFT_EYE=False,
            FILTER_RIGHT_EYE=False,
            FILTER_MOUTH=False,
            FINAL_MEDIAN=False,
        )

        return run_cfg_default


class smooth_no_eyes(ensemble_mouth_eyes_nose_smooth):
    def __init__(self):
        super().__init__()

    def get_default_kwargs(self):
        run_cfg_default = dict(
            FILTER_NOSE=True,
            FILTER_LEFT_EYE=False,
            FILTER_RIGHT_EYE=False,
            FILTER_MOUTH=True,
            FINAL_MEDIAN=True,
        )

        return run_cfg_default


class smooth_median_entire(ensemble_mouth_eyes_nose_smooth):
    def __init__(self):
        super().__init__()

    def get_default_kwargs(self):
        run_cfg_default = dict(
            FILTER_NOSE=False,
            FILTER_LEFT_EYE=False,
            FILTER_RIGHT_EYE=False,
            FILTER_MOUTH=False,
            FINAL_MEDIAN=True,
        )

        return run_cfg_default


# sigma field 256
class smooth_front_of_face_transform(nn.Module):
    def __init__(self, k=-2, ps=0.1, quantile=60, p=1.0, **kwargs):
        super().__init__()

        self.quantile = quantile
        self.ps = ps
        self.k = k
        self.p_apply = p

        if "quantile" in kwargs.keys():
            self.quantile = quantile

        if "k" in kwargs.keys():
            self.k = k

        if "ps" in kwargs.keys():
            self.ps = ps

    def forward(self, field_data, quantile=60, smooth_isolevel=False, k=-2, ps=0.1):
        X, Y, Z = field_data.shape

        p_cur = torch.rand((1), device=field_data.device)

        if p_cur > self.p_apply:
            return field_data

        original_isolevels = self.get_first_isolevel_crossing(field_data, isolevel=20)

        min_h_idx = max_h_idx = 1
        n_iter_for_quantile = 0
        while min_h_idx == max_h_idx:
            fd_rs, m_idx, ss = self.prepare_field_for_transform(field_data)  # preparation step

            if n_iter_for_quantile > 0:
                k = k - 1
            tfo, midx_diff = self.get_tfo_from_surface_idx(fd_rs, X, Y, Z, m_idx, quantile=quantile, k=k)  # get idx of max rays and transform

            idx_to_shift = torch.argwhere(tfo[:, 0] < 0)

            idx_qual = torch.where(tfo[:, 0].view(X, Y) < 0)

            if idx_qual[0].flatten().shape[0] <= 2 and idx_qual[1].flatten().shape[0] <= 2:
                return field_data

            xlist = idx_qual[0].flatten().clamp(20, X - 20)
            ylist = idx_qual[1].flatten().clamp(20, Y - 20)

            xmi = xlist.min().int().item()
            xma = xlist.max().int().item()
            ymi = ylist.min().int().item()
            yma = ylist.max().int().item()

            min_h_idx = m_idx[idx_to_shift].min()
            max_h_idx = m_idx[idx_to_shift].max()

            n_iter_for_quantile += 1

            if n_iter_for_quantile > 10:
                import sys

                print("error wcoudl not find quantile for this one...")
                sys.exit()

        x = field_data.clone()

        nose_start = x[:, :, :min_h_idx]
        nose_mid = x[:, :, min_h_idx:max_h_idx]
        nose_end = x[:, :, max_h_idx:]
        compress_range = max(1, int((max_h_idx - min_h_idx) / 1.1))

        nstart = nose_start
        nmid = torch.nn.functional.interpolate(nose_mid.unsqueeze(0).unsqueeze(0), size=(X, Y, compress_range)).squeeze(0).squeeze(0)
        end_range = Z - (nose_start.shape[-1]) - nmid.shape[-1]
        nend = torch.nn.functional.interpolate(nose_end.unsqueeze(0).unsqueeze(0), size=(X, Y, end_range)).squeeze(0).squeeze(0)
        fd_transformed = torch.cat((nstart, nmid, nend), -1)

        # ddk=median_filter(fd_transformed.clone(), (5, 5, 5))
        # fd_transformed[xmi:xma,ymi:yma,min_h_idx:]=fd_transformed[xmi:xma,ymi:yma,min_h_idx:]*0.6+(1-0.6)*ddk[xmi:xma,ymi:yma,min_h_idx:]

        x = fd_transformed.clone()
        fd_transformed_bk = fd_transformed.clone()

        # x=field_data.clone()

        nose_start = x[:, :, :min_h_idx]
        nose_mid = x[:, :, min_h_idx:max_h_idx]
        nose_end = x[:, :, max_h_idx:]

        compress_range = max(1, int((max_h_idx - min_h_idx) / 1.2))

        nstart = nose_start
        nmid = torch.nn.functional.interpolate(nose_mid.unsqueeze(0).unsqueeze(0), size=(X, Y, compress_range)).squeeze(0).squeeze(0)
        end_range = Z - (nose_start.shape[-1]) - nmid.shape[-1]
        nend = torch.nn.functional.interpolate(nose_end.unsqueeze(0).unsqueeze(0), size=(X, Y, end_range)).squeeze(0).squeeze(0)
        fd_transformed = torch.cat((nstart, nmid, nend), -1)

        # return(x)

        assert fd_transformed.shape == field_data.shape, print(fd_transformed.shape)

        ddk = median_filter(fd_transformed.clone(), (7, 7, 3))

        if ps is not None:
            p_smooth = ps
        else:
            p_smooth = torch.rand((1), device=field_data.device).flatten()[0] * 0.6 + 0.1

        fd_transformed[xmi:xma, ymi:yma, min_h_idx - 1 :] = fd_transformed[xmi:xma, ymi:yma, min_h_idx - 1 :] * p_smooth + (1 - p_smooth) * ddk[xmi:xma, ymi:yma, min_h_idx - 1 :]

        if ps is not None:
            p_smooth = ps
        else:
            p_smooth = torch.rand((1), device=field_data.device).flatten()[0] * 0.4 + 0.3

        p_os = torch.randint(3, 7, [1], device=field_data.device).flatten()[0]

        ddk = median_filter(fd_transformed.clone(), (5, 5, 3))

        fd_transformed[xmi:xma, ymi:yma, min_h_idx - p_os :] = fd_transformed[xmi:xma, ymi:yma, min_h_idx - p_os :] * p_smooth + (1 - p_smooth) * ddk[xmi:xma, ymi:yma, min_h_idx - p_os :]

        fd_transformed = median_filter(fd_transformed.clone(), (1, 1, 1))

        if smooth_isolevel:
            fd_transformed = self.smooth_on_isolevel(fd_transformed, original_isolevels)

        # finally blend it again

        # fd_transformed=field_data*0.6+(1-0.6)*fd_transformed

        return fd_transformed

    def forward_for_bad_augmentation(self, field_data, quantile=60, smooth_isolevel=False, k=-2):
        X, Y, Z = field_data.shape

        p_cur = torch.rand((1), device=field_data.device)

        if p_cur > self.p_apply:
            return field_data

        original_isolevels = self.get_first_isolevel_crossing(field_data, isolevel=20)

        fd_rs, m_idx, ss = self.prepare_field_for_transform(field_data)  # preparation step

        tfo, midx_diff = self.get_tfo_from_surface_idx(fd_rs, X, Y, Z, m_idx, quantile=quantile, k=k)  # get idx of max rays and transform

        idx_to_shift = torch.argwhere(tfo[:, 0] < 0)

        idx_qual = torch.where(tfo[:, 0].view(X, Y) < 0)

        if idx_qual[0].flatten().shape[0] <= 2 and idx_qual[1].flatten().shape[0] <= 2:
            return field_data

        xlist = idx_qual[0].flatten().clamp(20, X - 20)
        ylist = idx_qual[1].flatten().clamp(20, Y - 20)

        xmi = xlist.min().int().item()
        xma = xlist.max().int().item()
        ymi = ylist.min().int().item()
        yma = ylist.max().int().item()

        min_h_idx = m_idx[idx_to_shift].min()
        max_h_idx = m_idx[idx_to_shift].max()
        x = field_data.clone()

        nose_start = x[:, :, :min_h_idx]
        nose_mid = x[:, :, min_h_idx:max_h_idx]
        nose_end = x[:, :, max_h_idx:]
        compress_range = max(1, int((max_h_idx - min_h_idx) / 1.1))

        nstart = nose_start
        nmid = torch.nn.functional.interpolate(nose_mid.unsqueeze(0).unsqueeze(0), size=(X, Y, compress_range)).squeeze(0).squeeze(0)
        end_range = Z - (nose_start.shape[-1]) - nmid.shape[-1]
        nend = torch.nn.functional.interpolate(nose_end.unsqueeze(0).unsqueeze(0), size=(X, Y, end_range)).squeeze(0).squeeze(0)
        fd_transformed = torch.cat((nstart, nmid, nend), -1)

        # ddk=median_filter(fd_transformed.clone(), (5, 5, 5))
        # fd_transformed[xmi:xma,ymi:yma,min_h_idx:]=fd_transformed[xmi:xma,ymi:yma,min_h_idx:]*0.6+(1-0.6)*ddk[xmi:xma,ymi:yma,min_h_idx:]

        x = fd_transformed.clone()
        fd_transformed_bk = fd_transformed.clone()

        # x=field_data.clone()

        nose_start = x[:, :, :min_h_idx]
        nose_mid = x[:, :, min_h_idx:max_h_idx]
        nose_end = x[:, :, max_h_idx:]

        compress_range = max(1, int((max_h_idx - min_h_idx) / 1.2))

        nstart = nose_start
        nmid = torch.nn.functional.interpolate(nose_mid.unsqueeze(0).unsqueeze(0), size=(X, Y, compress_range)).squeeze(0).squeeze(0)
        end_range = Z - (nose_start.shape[-1]) - nmid.shape[-1]
        nend = torch.nn.functional.interpolate(nose_end.unsqueeze(0).unsqueeze(0), size=(X, Y, end_range)).squeeze(0).squeeze(0)
        fd_transformed = torch.cat((nstart, nmid, nend), -1)

        # return(x)

        assert fd_transformed.shape == field_data.shape, print(fd_transformed.shape)

        fd_transformed = median_filter(fd_transformed.clone(), (1, 1, 1))

        return fd_transformed

    def get_first_isolevel_crossing(self, x, isolevel=20):
        # use this syntax to get index of first 20 iso level surf for each X*Y rays...
        np_field = x.cpu().numpy()
        shp = np_field.shape  # X,Y,Z
        X, Y, Z = shp
        isolev = np.argmax((np.flip(np_field, -1).reshape([X * Y, Z]) > isolevel).astype(float), -1)
        isolev = Z - isolev.reshape((X, Y, 1))
        return isolev

    def smooth_on_isolevel(self, sf_orig, isolevel_locations):
        # will apply median kernel filter (3dconv) on the coordinate which define the isosurface
        x_mp = 64
        y_mp = 70

        mcd_y = 60
        mcd_x = 30
        coords = []
        for xc in range(x_mp - mcd_x, x_mp - mcd_x):
            for yc in range(y_mp - mcd_y, y_mp + mcd_y):
                coords.append((xc, yc))

        coords = [coords[i] for i in torch.randperm(len(coords))]
        for c in coords:
            xc, yc = c
            zc = isolevel_locations[xc, yc][0]
            sf_orig[xc - 2 : xc + 3, yc - 2 : yc + 3, zc - 2 : zc + 3] = median_filter(sf_orig[xc - 2 : xc + 3, yc - 2 : yc + 3, zc - 2 : zc + 3], (5, 5, 3))

        return sf_orig

    def get_tfo_from_surface_idx(self, fd_rs, X, Y, Z, m_idx, quantile, k):
        if k == None:
            k = self.k
        max_idx_list = m_idx.reshape(X, Y)

        # print(f'quantile: {quantile}')

        # print('all such quantile')
        # print(torch.quantile(m_idx.flatten().float(),torch.linspace(0.0,1.0,steps=quantile,device=m_idx.device)))#[-1]#.unsqueeze(0).expand(X*Y).reshape(X,Y)

        mean_of = torch.quantile(
            m_idx.flatten().float(),
            torch.linspace(0.0, 1.0, steps=quantile, device=m_idx.device),
        )[k]  # .unsqueeze(0).expand(X*Y).reshape(X,Y)
        # print(f'mean of: {mean_of.item()}')
        midx_diff = mean_of - max_idx_list
        # print(f'max idx list max: {max_idx_list.max()}')

        # tformed=max_idx_list+torch.clamp(k*midx_diff,max=0)

        tformed = torch.ones_like(max_idx_list) * mean_of
        tfo = tformed - max_idx_list
        tfo = tfo.reshape(X * Y, -1)  # .cuda()

        return (tfo, midx_diff)

    def prepare_field_for_transform(self, field_data):
        X, Y, Z = field_data.shape
        self.Z = Z

        fd_rs = field_data.reshape(X * Y, -1)
        m_idx = torch.argmax(fd_rs[:, : Z - 3], 1)  # 3 for boundary don't want to get Z=128 as argmax
        ss = fd_rs.shape[-1]

        return (fd_rs, m_idx, ss)

    def modify_rays(self, x, tfo, m_idx, ss, idx_to_shift, min_h_idx, max_h_idx):
        fd_rs = x.clone()

        lfmax = 0.0
        lfmin = 1.0

        rsc = min_h_idx
        nint = max_h_idx - rsc

        print(f"number to shift: {idx_to_shift.flatten().shape[0]}")
        for si in idx_to_shift:
            si = si.flatten()[0]

            if tfo[si] == 0:
                continue

            h_idx = m_idx[si].flatten()[0]

            if h_idx < rsc:
                continue

            if h_idx >= self.Z:
                continue

            tfo_o = tfo[si]
            a_split = h_idx + tfo_o

            ls, rs = torch.arange(ss).split((a_split.int(), ss - a_split.int()))

            ls = int(ls.shape[0])
            rs = ss - ls

            lh = torch.nn.functional.interpolate(fd_rs[si, :][:h_idx].view(1, 1, -1), size=ls)
            rh = torch.nn.functional.interpolate(fd_rs[si, :][h_idx:].view(1, 1, -1), size=rs)

            thenew = torch.cat((lh, rh), -1)

            lfactor = 1 - (nint - (h_idx - rsc)) / (nint)  # **2

            if lfactor > lfmax:
                lfmax = lfactor
                maxval = h_idx

                asplit_max = a_split
                lsmax = ls
                rsmax = rs

                tfomax = tfo_o
            if lfactor < lfmin:
                lfmin = lfactor
                minval = h_idx
                asplit_min = a_split

                lsmin = ls
                rsmin = rs

                tfomin = tfo_o

            fd_rs[si, :] = fd_rs[si, :] * (lfactor) + thenew * (1 - lfactor)  # *(0.5+lfactor) + (0.5-lfactor)* thenew)/0.5 #torch.cat((lh,rh),-1)

        print("\n")
        print(lfmin)
        print(lfmax)
        print("\n")

        print(maxval)
        print(minval)

        print("\n")
        print(asplit_max)
        print(asplit_min)

        print("\n")
        print(lsmax)
        print(rsmax)

        print("\n")
        print(lsmin)
        print(rsmin)

        print("\n")
        print(tfomax)
        print(tfomin)

        return fd_rs


# sigma field 256
class smooth_front_of_face_transform_bad(nn.Module):
    def __init__(self, k=4, p=1.0):
        super().__init__()

        self.k = k
        self.p_apply = p

    def forward(self, field_data, quantile=10, smooth_isolevel=False, k=-2, ps=None):
        X, Y, Z = field_data.shape

        p_cur = torch.rand((1), device=field_data.device)

        if p_cur > self.p_apply:
            return field_data

        original_isolevels = self.get_first_isolevel_crossing(field_data, isolevel=20)

        min_h_idx = max_h_idx = 1
        n_iter_for_quantile = 0
        while min_h_idx == max_h_idx:
            fd_rs, m_idx, ss = self.prepare_field_for_transform(field_data)  # preparation step

            if n_iter_for_quantile > 0:
                k = k - 1
            tfo, midx_diff = self.get_tfo_from_surface_idx(fd_rs, X, Y, Z, m_idx, quantile=quantile, k=k)  # get idx of max rays and transform

            idx_to_shift = torch.argwhere(tfo[:, 0] < 0)

            idx_qual = torch.where(tfo[:, 0].view(X, Y) < 0)

            if idx_qual[0].flatten().shape[0] <= 2 and idx_qual[1].flatten().shape[0] <= 2:
                return field_data

            xlist = idx_qual[0].flatten().clamp(20, X - 20)
            ylist = idx_qual[1].flatten().clamp(20, Y - 20)

            xmi = xlist.min().int().item()
            xma = xlist.max().int().item()
            ymi = ylist.min().int().item()
            yma = ylist.max().int().item()

            min_h_idx = m_idx[idx_to_shift].min()
            max_h_idx = m_idx[idx_to_shift].max()

            n_iter_for_quantile += 1

            if n_iter_for_quantile > 10:
                import sys

                print("error wcoudl not find quantile for this one...")
                sys.exit()

        x = field_data.clone()

        nose_start = x[:, :, :min_h_idx]
        nose_mid = x[:, :, min_h_idx:max_h_idx]
        nose_end = x[:, :, max_h_idx:]
        compress_range = int((max_h_idx - min_h_idx) / 1.1)
        compress_range = 6  # int((max_h_idx-min_h_idx)/1.1)
        nstart = nose_start
        nmid = torch.nn.functional.interpolate(nose_mid.unsqueeze(0).unsqueeze(0), size=(X, Y, compress_range)).squeeze(0).squeeze(0)
        end_range = Z - (nose_start.shape[-1]) - nmid.shape[-1]
        nend = torch.nn.functional.interpolate(nose_end.unsqueeze(0).unsqueeze(0), size=(X, Y, end_range)).squeeze(0).squeeze(0)
        fd_transformed = torch.cat((nstart, nmid, nend), -1)

        ddk = median_filter(fd_transformed.clone(), (5, 5, 5))

        if ps is not None:
            p_smooth = ps
        else:
            p_smooth = torch.rand((1), device=field_data.device).flatten()[0] * 0.8 + 0.1

        fd_transformed[xmi:xma, ymi:yma, min_h_idx:] = fd_transformed[xmi:xma, ymi:yma, min_h_idx:] * p_smooth + (1 - p_smooth) * ddk[xmi:xma, ymi:yma, min_h_idx:]

        x = fd_transformed.clone()

        nose_start = x[:, :, :min_h_idx]
        nose_mid = x[:, :, min_h_idx:max_h_idx]
        nose_end = x[:, :, max_h_idx:]
        compress_range = max(1, int((max_h_idx - min_h_idx) / 1.1))
        nstart = nose_start
        nmid = torch.nn.functional.interpolate(nose_mid.unsqueeze(0).unsqueeze(0), size=(X, Y, compress_range)).squeeze(0).squeeze(0)
        end_range = Z - (nose_start.shape[-1]) - nmid.shape[-1]
        nend = torch.nn.functional.interpolate(nose_end.unsqueeze(0).unsqueeze(0), size=(X, Y, end_range)).squeeze(0).squeeze(0)
        fd_transformed = torch.cat((nstart, nmid, nend), -1)

        ddk = median_filter(fd_transformed.clone(), (7, 7, 3))

        if ps is not None:
            p_smooth = ps
        else:
            p_smooth = torch.rand((1), device=field_data.device).flatten()[0] * 0.8 + 0.1

        # fd_transformed[xmi:xma,ymi:yma,min_h_idx-1:]=fd_transformed[xmi:xma,ymi:yma,min_h_idx-1:]*0.4+(1-0.4)*ddk[xmi:xma,ymi:yma,min_h_idx-1:]
        fd_transformed[xmi:xma, ymi:yma, min_h_idx - 1 :] = fd_transformed[xmi:xma, ymi:yma, min_h_idx - 1 :] * p_smooth + (1 - p_smooth) * ddk[xmi:xma, ymi:yma, min_h_idx - 1 :]

        if ps is not None:
            p_smooth = ps
        else:
            p_smooth = torch.rand((1), device=field_data.device).flatten()[0] * 0.4 + 0.3

        p_os = torch.randint(3, 7, [1], device=field_data.device).flatten()[0]

        ddk = median_filter(fd_transformed.clone(), (5, 5, 3))
        fd_transformed[xmi:xma, ymi:yma, min_h_idx - p_os :] = fd_transformed[xmi:xma, ymi:yma, min_h_idx - p_os :] * p_smooth + (1 - p_smooth) * ddk[xmi:xma, ymi:yma, min_h_idx - p_os :]
        fd_transformed = median_filter(fd_transformed.clone(), (1, 1, 1))

        if smooth_isolevel:
            fd_transformed = self.smooth_on_isolevel(fd_transformed, original_isolevels)

        return fd_transformed

    def get_first_isolevel_crossing(self, x, isolevel=20):
        # use this syntax to get index of first 20 iso level surf for each X*Y rays...
        np_field = x.cpu().numpy()
        shp = np_field.shape  # X,Y,Z
        X, Y, Z = shp
        isolev = np.argmax((np.flip(np_field, -1).reshape([X * Y, Z]) > isolevel).astype(float), -1)
        isolev = Z - isolev.reshape((X, Y, 1))
        return isolev

    def smooth_on_isolevel(self, sf_orig, isolevel_locations):
        # will apply median kernel filter (3dconv) on the coordinate which define the isosurface
        x_mp = 64
        y_mp = 70

        mcd_y = 60
        mcd_x = 30
        coords = []
        for xc in range(x_mp - mcd_x, x_mp - mcd_x):
            for yc in range(y_mp - mcd_y, y_mp + mcd_y):
                coords.append((xc, yc))

        coords = [coords[i] for i in torch.randperm(len(coords))]
        for c in coords:
            xc, yc = c
            zc = isolevel_locations[xc, yc][0]
            sf_orig[xc - 2 : xc + 3, yc - 2 : yc + 3, zc - 2 : zc + 3] = median_filter(sf_orig[xc - 2 : xc + 3, yc - 2 : yc + 3, zc - 2 : zc + 3], (5, 5, 3))

        return sf_orig

    def get_tfo_from_surface_idx(self, fd_rs, X, Y, Z, m_idx, quantile, k):
        if k == None:
            k = self.k
        max_idx_list = m_idx.reshape(X, Y)

        # print(f'quantile: {quantile}')

        # print('all such quantile')
        # print(torch.quantile(m_idx.flatten().float(),torch.linspace(0.0,1.0,steps=quantile,device=m_idx.device)))#[-1]#.unsqueeze(0).expand(X*Y).reshape(X,Y)

        mean_of = torch.quantile(
            m_idx.flatten().float(),
            torch.linspace(0.0, 1.0, steps=quantile, device=m_idx.device),
        )[k]  # .unsqueeze(0).expand(X*Y).reshape(X,Y)
        # print(f'mean of: {mean_of.item()}')
        midx_diff = mean_of - max_idx_list
        # print(f'max idx list max: {max_idx_list.max()}')

        # tformed=max_idx_list+torch.clamp(k*midx_diff,max=0)

        tformed = torch.ones_like(max_idx_list) * mean_of
        tfo = tformed - max_idx_list
        tfo = tfo.reshape(X * Y, -1)  # .cuda()

        return (tfo, midx_diff)

    def prepare_field_for_transform(self, field_data):
        X, Y, Z = field_data.shape
        self.Z = Z

        fd_rs = field_data.reshape(X * Y, -1)
        m_idx = torch.argmax(fd_rs[:, : Z - 3], 1)  # 3 for boundary don't want to get Z=128 as argmax
        ss = fd_rs.shape[-1]

        return (fd_rs, m_idx, ss)

    def modify_rays(self, x, tfo, m_idx, ss, idx_to_shift, min_h_idx, max_h_idx):
        fd_rs = x.clone()

        lfmax = 0.0
        lfmin = 1.0

        rsc = min_h_idx
        nint = max_h_idx - rsc

        print(f"number to shift: {idx_to_shift.flatten().shape[0]}")
        for si in idx_to_shift:
            si = si.flatten()[0]

            if tfo[si] == 0:
                continue

            h_idx = m_idx[si].flatten()[0]

            if h_idx < rsc:
                continue

            if h_idx >= self.Z:
                continue

            tfo_o = tfo[si]
            a_split = h_idx + tfo_o

            ls, rs = torch.arange(ss).split((a_split.int(), ss - a_split.int()))

            ls = int(ls.shape[0])
            rs = ss - ls

            lh = torch.nn.functional.interpolate(fd_rs[si, :][:h_idx].view(1, 1, -1), size=ls)
            rh = torch.nn.functional.interpolate(fd_rs[si, :][h_idx:].view(1, 1, -1), size=rs)

            thenew = torch.cat((lh, rh), -1)

            lfactor = 1 - (nint - (h_idx - rsc)) / (nint)  # **2

            if lfactor > lfmax:
                lfmax = lfactor
                maxval = h_idx

                asplit_max = a_split
                lsmax = ls
                rsmax = rs

                tfomax = tfo_o

            if lfactor < lfmin:
                lfmin = lfactor
                minval = h_idx
                asplit_min = a_split

                lsmin = ls
                rsmin = rs

                tfomin = tfo_o

            fd_rs[si, :] = fd_rs[si, :] * (lfactor) + thenew * (1 - lfactor)  # *(0.5+lfactor) + (0.5-lfactor)* thenew)/0.5 #torch.cat((lh,rh),-1)

        print("\n")
        print(lfmin)
        print(lfmax)
        print("\n")

        print(maxval)
        print(minval)

        print("\n")
        print(asplit_max)
        print(asplit_min)

        print("\n")
        print(lsmax)
        print(rsmax)

        print("\n")
        print(lsmin)
        print(rsmin)

        print("\n")
        print(tfomax)
        print(tfomin)

        return fd_rs


# @torch.jit.script
# on a type of sigma_field_256 data
class mouth_corner_smoothe(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, n_apps=40):
        # -------------------------------------------------------------------

        if n_apps == None:
            n_apps = 4

        # isolevel_locations=self.get_first_isolevel_crossing(sf_orig,isolevel=20)

        orig_shape = list(x.shape)

        if len(orig_shape) > 3:
            for o in orig_shape[: len(orig_shape) - 3]:
                assert o == 1, "error u can only do smooth side of face transform on a single item at time"
            for k in range(len(orig_shape) - 3):
                x = x.squeeze(0)

        newshape_nobatch = list(x.shape)
        inverse_permute_at_end = False
        if newshape_nobatch == [129, 141, 128]:
            x = x.permute(2, 1, 0)
            inverse_permute_at_end = True

        sf_orig = x.clone()

        x_mp = 65
        y_mp = 42

        for _ in range(n_apps):
            xo = 6
            xor = 10
            k_sizes = [7, 3, 9, 13, 17]
            xlo = [1, 0, -1, -2, -3, -4]
            xro = [-2, -1, 0, 1, 2, 3]

            xlo = xlo[torch.randperm(len(xlo))[0]]  # offset for it
            xro = xro[torch.randperm(len(xro))[0]]  # offset for it

            xo = xo + xlo
            xor = xor + xro

            mcd = k_sizes[torch.randperm(len(k_sizes))[0]]  # offset for it

            output_mouth_l = sf_orig[x_mp - 25 - mcd : x_mp - 25 + xo, y_mp - mcd : y_mp + mcd, :].clone()  # .shape
            output_mouth_r = sf_orig[x_mp + 25 - xor : x_mp + 25 + mcd, y_mp - mcd : y_mp + mcd, :].clone()  # .shape

            output_mouth_l = median_filter(output_mouth_l, (5, 5, 1))
            output_mouth_r = median_filter(output_mouth_r, (5, 5, 1))

            # --------------------------------------------------------------------------
            # do some smoothing on it otherwise we get some artefacts
            # interpolation factor in 0,1 is a function of the kernel size (mcd)
            # ie larger kernel will be more smooth
            # this reduces square artefacts in final output
            lp = 1 / mcd
            sf_orig[x_mp - 25 - mcd : x_mp - 25 + xo, y_mp - mcd : y_mp + mcd, :] = sf_orig[x_mp - 25 - mcd : x_mp - 25 + xo, y_mp - mcd : y_mp + mcd, :] * (1 - lp) + output_mouth_l * lp
            sf_orig[x_mp + 25 - xor : x_mp + 25 + mcd, y_mp - mcd : y_mp + mcd, :] = sf_orig[x_mp + 25 - xor : x_mp + 25 + mcd, y_mp - mcd : y_mp + mcd, :] * (1 - lp) + output_mouth_r * lp

            # this one is just smooothing some patches around the corners...
            # sf_orig=self.smooth_lr_corners_333(sf_orig)

            # this one is just smoothing on the isolevel patches
            # sf_orig=self.smooth_on_isolevel(sf_orig,isolevel_locations)

        x = sf_orig

        if inverse_permute_at_end:
            x = x.permute(2, 1, 0)

        x = x.reshape(orig_shape)

        return x

    def get_first_isolevel_crossing(self, x, isolevel=20):
        # use this syntax to get index of first 20 iso level surf for each X*Y rays...
        np_field = x.cpu().numpy()
        shp = np_field.shape  # X,Y,Z
        X, Y, Z = shp
        isolev = np.argmax((np.flip(np_field, -1).reshape([X * Y, Z]) > isolevel).astype(float), -1)
        isolev = Z - isolev.reshape((X, Y, 1))
        return isolevel_locations

    def smooth_on_isolevel(self, sf_orig, isolevel_locations):
        # will apply median kernel filter (3dconv) on the coordinate which define the isosurface
        x_mp = 65
        y_mp = 42

        mcd = 13
        coords = []
        for xc in range(x_mp - 25 - mcd, x_mp - 25 + mcd):
            for yc in range(y_mp - mcd, y_mp + mcd):
                coords.append((xc, yc))

        coords = [coords[i] for i in torch.randperm(len(coords))]
        for c in coords:
            xc, yc = c
            zc = isolevel_locations[xc, yc][0]
            sf_orig[xc - 2 : xc + 3, yc - 2 : yc + 3, zc - 2 : zc + 3] = median_filter(sf_orig[xc - 2 : xc + 3, yc - 2 : yc + 3, zc - 2 : zc + 3], (5, 5, 5))

        return sf_orig

    def smooth_lr_corners_333(self, sf_orig):
        k_sizes = [7, 3, 5]

        mcd = k_sizes[torch.randperm(len(k_sizes))[0]]

        sf_orig[x_mp - 25 - mcd : x_mp - 25 + mcd, y_mp - mcd : y_mp + mcd, :] = median_filter(
            sf_orig[x_mp - 25 - mcd : x_mp - 25 + mcd, y_mp - mcd : y_mp + mcd, :].clone(),
            (3, 3, 1),
        )
        sf_orig[x_mp + 25 - mcd : x_mp + 25 + mcd, y_mp - mcd : y_mp + mcd, :] = median_filter(
            sf_orig[x_mp + 25 - mcd : x_mp + 25 + mcd, y_mp - mcd : y_mp + mcd, :].clone(),
            (3, 3, 1),
        )

        return sf_orig


class flatten_all_face_and_smooth(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, field_data, p_smooth=0.8, KR=20):  # , quantile=10, smooth_isolevel=False, k=-2, ps=None):
        X, Y, Z = (128, 141, 129)
        x = field_data.clone().squeeze(0)

        min_h_idx = 10
        max_h_idx = 120

        # KR=15

        nose_start = x[:, :, :min_h_idx]
        nose_mid = x[:, :, min_h_idx:max_h_idx]
        nose_end = x[:, :, max_h_idx:]
        compress_range = int((max_h_idx - min_h_idx) / 1.05)
        # compress_range = max_h_idx-min_h_idx-KR  # int((max_h_idx-min_h_idx)/1.1)
        nstart = nose_start

        nmid = torch.nn.functional.interpolate(nose_mid.unsqueeze(0).unsqueeze(0), size=(X, Y, compress_range)).squeeze(0).squeeze(0)
        end_range = Z - (nose_start.shape[-1]) - nmid.shape[-1]
        nend = torch.nn.functional.interpolate(nose_end.unsqueeze(0).unsqueeze(0), size=(X, Y, end_range)).squeeze(0).squeeze(0)
        fd_transformed = torch.cat((nstart, nmid, nend), -1).unsqueeze(0)

        klist = [(3, 3, 7), (5, 5, 7), (5, 5, 3)]

        for k in klist:
            ddk = median_filter(fd_transformed.clone(), k)
            fd_transformed = fd_transformed * p_smooth + (1 - p_smooth) * ddk

        return fd_transformed.squeeze(0)


class kpts_3d_smooth(torch.nn.Module):
    def __init__(self, idx=54, dx=0, dy=0, kernel_size=(3, 3, 5), exponent=3):
        super().__init__()

        self.idx = idx
        self.dx = dx
        self.dy = dy
        self.kernel_size = kernel_size
        self.exponent = exponent

        import sys

        # Legacy note: old notebook-era code used ad hoc sys.path injection before the reward-model framework moved out of eg3d.
        from core_modules.utils import finetuning_utils

        self.MUDC = finetuning_utils.MeshUtilsDataClass()

    def forward(self, sigma_field, kpts_3d):
        samps, c_shape = self.MUDC.get_samples_coordinates_from_dtype("sigma_field_256")
        # -------------------------------------------------------------------

        x = sigma_field

        # isolevel_locations=self.get_first_isolevel_crossing(sf_orig,isolevel=20)

        orig_shape = list(x.shape)

        if len(orig_shape) > 3:
            for o in orig_shape[: len(orig_shape) - 3]:
                assert o == 1, "error u can only do smooth side of face transform on a single item at time"
            for k in range(len(orig_shape) - 3):
                x = x.squeeze(0)

        newshape_nobatch = list(x.shape)
        inverse_permute_at_end = False
        if newshape_nobatch == [129, 141, 128]:
            x = x.permute(2, 1, 0)
            inverse_permute_at_end = True

        sfc = x.clone()

        for iter in range(20):
            sel_kpt = kpts_3d[[self.idx]]  # pointy nose...

            sel_kpt[0, 0] += self.dx
            sel_kpt[0, 1] += self.dy

            dist_from = torch.linalg.norm(samps - sel_kpt.unsqueeze(1).expand_as(samps).cuda(), ord=2, dim=-1)
            df_rs = dist_from.reshape(c_shape[1:4])
            df_rs_sq = 1 / (1 + 1e-3 + df_rs) ** self.exponent

            # now norm....
            dmin = df_rs_sq.min()
            dmax = df_rs_sq.max()
            df_rs_sq = (df_rs_sq - dmin) / (dmax - dmin)

            sf_median = median_filter(sfc.clone(), self.kernel_size)
            lerp_factor_3d = torch.zeros_like(sf_median)
            lerp_factor_3d += df_rs_sq
            blended = lerp_factor_3d * sf_median + (1 - lerp_factor_3d) * sfc
            sfc = blended

        x = sfc

        if inverse_permute_at_end:
            x = x.permute(2, 1, 0)

        x = x.reshape(orig_shape)

        return x


class nose_tip_keypoint_smooth(kpts_3d_smooth):
    """Sigma-field smoothing around the nose tip keypoint."""

    def __init__(self, idx=54, dx=0, dy=0, kernel_size=(3, 3, 5), exponent=35):
        super().__init__(idx=idx, dx=dx, dy=dy, kernel_size=kernel_size, exponent=exponent)


class mcorner_lhs_smooth(kpts_3d_smooth):
    """Sigma-field smoothing around the left mouth corner keypoint."""

    def __init__(self, idx=76, dx=0, dy=0, kernel_size=(3, 3, 3), exponent=35):
        super().__init__(idx=idx, dx=dx, dy=dy, kernel_size=kernel_size, exponent=exponent)


class mcorner_rhs_smooth(kpts_3d_smooth):
    """Sigma-field smoothing around the right mouth corner keypoint."""

    def __init__(self, idx=82, dx=0, dy=0, kernel_size=(3, 3, 3), exponent=35):
        super().__init__(idx=idx, dx=dx, dy=dy, kernel_size=kernel_size, exponent=exponent)


class HighPassFilter(nn.Module):
    def __init__(self):
        super(HighPassFilter, self).__init__()
        # Define a high-pass filter kernel
        # This is a simple edge-detection kernel; more complex kernels can be used
        self.kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(1, 3, 3, 3)
        self.kernel.requires_grad_(False)  # Kernel does not require gradient

    def forward(self, x):
        # Assuming x is of shape (batch_size, channels, height, width)
        # Ensure the input tensor is on the same device as the model
        self.kernel = self.kernel.to(x.device)
        # Apply the high-pass filter
        return F.conv2d(x, self.kernel, padding=1)
