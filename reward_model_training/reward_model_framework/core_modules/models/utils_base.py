import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.batchnorm import _BatchNorm


# Activation helpers
def get_activation(name="silu", inplace=False):
    if name is None:
        return None
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "selu":
        module = nn.SELU(inplace=inplace)
    elif name == "softplus":
        module = nn.Softplus()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "hswish":
        module = HSwish()
    elif name == "gelu":
        module = nn.GELU()
    elif name == "cos":
        module = cosine_as_class()
    else:
        raise AttributeError(f"Unsupported activation function type: {name}")
    return module


def make_mlp(
    in_channels,
    mlp_channels,
    normalisation_type="batch_norm",
    activation_type="relu",
    use_residual_block_kaming=False,
):
    assert len(mlp_channels) >= 1
    layers = []
    using_norm = True
    if normalisation_type == "batch_norm":
        norm_layer = nn.BatchNorm1d
    elif normalisation_type == "layer_norm":
        norm_layer = nn.LayerNorm
    elif normalisation_type is None or normalisation_type.lower() == "none":
        using_norm = False
    act = get_activation(activation_type)
    for c in mlp_channels:
        layers += [nn.Linear(in_channels, c)]
        if using_norm:
            layers += [norm_layer(c)]
        layers += [act]
        in_channels = c
    if use_residual_block_kaming:
        layers += [nn.Linear(c, c)]
    return nn.Sequential(*layers)


ce_loss = nn.CrossEntropyLoss()
bce_w_logits_loss = nn.BCEWithLogitsLoss()


def get_ncomb2(k):
    if k == 2:
        return 1
    if k == 3:
        return 3
    if k == 4:
        return 6
    if k == 5:
        return 10
    if k == 6:
        return 15
    if k == 7:
        return 21
    return 0


def reorder_pair_by_idx(pair, idx):
    out_tuple = (pair[idx[0]], pair[idx[1]])
    return out_tuple


def get_rand_reordered_pair(p=0.5):
    sel_p = np.random.rand()
    in_tuple = (0, 1)
    if sel_p > p:
        out_tuple = (in_tuple[1], in_tuple[0])
    else:
        out_tuple = in_tuple
    return out_tuple


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(step=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, step):
        lr_factor = 0.5 * (1 + np.cos(np.pi * step / self.max_num_iters))
        if step <= self.warmup:
            lr_factor *= (step * 1.0 + 0.00001) / self.warmup
        return lr_factor


def reduce_x(x, agg_type, return_idx=False, sel_dim=2):
    if agg_type == "std":
        agg_features = torch.std(x, sel_dim)
    if agg_type == "max":
        agg_features = torch.max(x, sel_dim)
        if not return_idx:
            agg_features = agg_features[0]
    if agg_type in ["mean", "avg"]:
        agg_features = torch.mean(x, sel_dim)
    if agg_type == "sum":
        agg_features = torch.sum(x, sel_dim)
    if agg_type == "cumsum_std":
        agg_features_cumsum = torch.cumsum(x, sel_dim)
        agg_features = torch.std(agg_features_cumsum, sel_dim)
    if agg_type == "cumsum_norm":
        agg_features_cumsum = torch.cumsum(x, sel_dim)
        agg_features = torch.norm(agg_features_cumsum, p="fro", dim=sel_dim)
    if agg_type == "cumsum_mean":
        agg_features_cumsum = torch.cumsum(x, sel_dim)
        agg_features = torch.mean(agg_features_cumsum, sel_dim)
    if agg_type == "cummax_std":
        agg_features_cummax = torch.cummax(x, sel_dim)[0]
        agg_features = torch.std(agg_features_cummax, sel_dim)
    if agg_type == "cummax_norm":
        agg_features_cummax = torch.cummax(x, sel_dim)[0]
        agg_features = torch.norm(agg_features_cummax, p="fro", dim=sel_dim)
    if agg_type == "cummax_mean":
        agg_features_cummax = torch.cummax(x, sel_dim)[0]
        agg_features = torch.mean(agg_features_cummax, sel_dim)
    return agg_features


@torch.jit.script
def fourier_features(x: torch.Tensor, B: int, t: int, T: int) -> torch.Tensor:
    """Compute Fourier Features with frequency regularisation (FREENerf style)."""
    x = x.unsqueeze(-1)
    freqs = torch.linspace(1.0, B, B, device=x.device)
    features = torch.cat([torch.sin(np.pi * x * freqs), torch.cos(np.pi * x * freqs)], dim=-1)
    return features


class cosine_as_class(nn.Module):
    def forward(self, x):
        return torch.cos(x)


class HSwish(nn.Module):
    def forward(self, x):
        return x * torch.nn.functional.relu6(x + 3, inplace=True) / 6


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        dropout_rate=0.5,
        normalisation_type="batch_norm",
        activation_type="softplus",
        use_dropout=False,
        residual=True,
        norm_first_layer=False,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.norm_first_layer = norm_first_layer
        self.using_norm = True
        if normalisation_type in ["batch_norm", "batchnorm"]:
            norm_layer = nn.BatchNorm1d
        elif normalisation_type in ["layer_norm", "layernorm"]:
            norm_layer = nn.LayerNorm
        elif normalisation_type in ["instance_norm", "instancenorm"]:
            norm_layer = nn.InstanceNorm1d
        elif normalisation_type is None or normalisation_type == "none":
            self.using_norm = False

        self.residual = len(self.hidden_sizes) > 1 and residual
        act = get_activation(activation_type)
        self.use_dropout = use_dropout
        drop_layer = nn.Dropout(dropout_rate) if self.use_dropout else nn.Identity()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            if i == 0 and self.norm_first_layer:
                self.hidden_layers.append(nn.Sequential(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), norm_layer(hidden_sizes[i + 1]), act, drop_layer))
            elif i > 0 and self.using_norm:
                self.hidden_layers.append(nn.Sequential(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), norm_layer(hidden_sizes[i + 1]), act, drop_layer))
            else:
                self.hidden_layers.append(nn.Sequential(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), act, drop_layer))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        input_vec = x
        x = self.input_layer(input_vec)
        resid_input = x
        for hidden_layer in self.hidden_layers[:-1]:
            x = hidden_layer(x)
        if self.residual:
            x = x + resid_input
        x = self.hidden_layers[-1](x)
        x = self.output_layer(x)
        return x


import hydra
import torch


def build_optimizer_and_scheduler(hparams, params, log):
    """Construct optimizer and optional scheduler from Hydra hparams."""
    import omegaconf

    import sharpness_aware_optimiser

    opt_cfg = hparams.optimizer
    sched_cfg = hparams.scheduler

    if isinstance(opt_cfg, omegaconf.dictconfig.DictConfig):
        opt_cfg = hydra.utils.instantiate(opt_cfg)

    if sched_cfg is not None and isinstance(sched_cfg, omegaconf.dictconfig.DictConfig):
        sched_cfg = hydra.utils.instantiate(sched_cfg)

    using_sam = False

    if hasattr(opt_cfg, "func") and opt_cfg.func == sharpness_aware_optimiser.SAM:
        base_optimizer = torch.optim.SGD
        optimizer = sharpness_aware_optimiser.SAM(params, base_optimizer, lr=1e-3, momentum=0.9, adaptive=False, rho=0.05)
        using_sam = True
    else:
        optimizer = opt_cfg(params=params)

    scheduler = None
    if sched_cfg is not None:
        scheduler = sched_cfg(optimizer=optimizer)

    return optimizer, scheduler, using_sam


__all__ = [
    "MLP",
    "get_activation",
    "make_mlp",
    "cosine_as_class",
    "HSwish",
    "ce_loss",
    "bce_w_logits_loss",
    "get_ncomb2",
    "reorder_pair_by_idx",
    "get_rand_reordered_pair",
    "disable_running_stats",
    "enable_running_stats",
    "CosineWarmupScheduler",
    "reduce_x",
    "build_optimizer_and_scheduler",
]
