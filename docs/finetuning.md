# Fine-tuning EG3D against a reward model

This stage fine-tunes the pretrained EG3D generator so its geometry scores
better under a trained reward model. The reward enters as a differentiable
regulariser on the generator update, alongside the standard EG3D GAN loss and a
sigma-field consistency anchor to the frozen pretrained generator.

Driven by Hydra: select a `finetune` `experiment`, optionally point it at any
reward model with `rwd_model_id`.

## Prerequisites

- The CUDA environment from the [README](../README.md#setup).
- A pretrained EG3D FFHQ checkpoint (`ffhq512-128.pkl`, external).
- A trained reward-model bundle under
  `reward_model_training/reward_model_framework/core_modules/RWD_MODELS_FOR_TUNING/<id>/`
  (either trained yourself per [reward_models.md](reward_models.md), or the
  released `7wnzkgie` bundle — see the [README](../README.md#download-the-released-models)).

## Reported run

```sh
cd eg3d
bash reward_tune_analysis/scripts/sfield_reported_run.sh
```

This is the exact reported configuration: the `7wnzkgie` sigma-field reward
model, raw clipped reward `L_r = -λ·clip(r(x), -10, 10)` with `λ = 10`, fine-
tuning from the pretrained checkpoint (resume at 2048 kimg). Override the output
directory with `EG3D_TRAINING_OUTDIR=/path/to/runs`.

## Fine-tune from any reward model

The general entrypoint is `train_rlhf.py`. Pick a `finetune` config and pass the
reward-model id:

```sh
cd eg3d
python train_rlhf.py experiment=finetune_eg3d_sfield rwd_model_id=<your_id> \
  using_wandb=false click_legacy_args.outdir=/path/to/runs
```

| `experiment=` | Reward representation |
|---|---|
| `finetune_eg3d_sfield` | sigma field **(reported)** |
| `finetune_eg3d_sdmap` | single depth map |
| `finetune_eg3d_tdmap` | triple depth map |
| `finetune_eg3d_pn1` | point cloud (PointNet) |
| `finetune_eg3d_null` | none — `λ_r = 0` baseline (control for FID) |

The non-sigma configs reproduce the paper's comparison showing that image-
derived reward signals are unstable as a fine-tuning driver; `finetune_eg3d_null`
is the matched no-reward control.

## Fast smoke run

A cheap end-to-end pass that reaches `tick 0` and exits (~15 s on a 4090):

```sh
python train_rlhf.py experiment=finetune_eg3d_sfield +smoke=on \
  click_legacy_args.outdir=/tmp/eg3d_rlhf_smoke
```

The `+smoke=on` preset shrinks the batch, stops at `tick 0`, and disables the
expensive reporting paths (FID/KID, reward plots, the preference-study render,
W&B, end-of-run mesh render) while keeping the reward-loss path exercised. Any
inline override still wins over the preset.

## Verification across all configs

```sh
bash reward_tune_analysis/scripts/run_protected_finetune_one_tick.sh
```

Runs one tick of each maintained finetune config as a correctness check.

## Outputs

Each run writes a directory under `click_legacy_args.outdir` containing
`training_options.json`, `hydra_cfg.yaml`, network snapshots, preview images,
and (when enabled) reward histograms/CSVs and mesh exports.

## Before/after mesh export

To export the tuned-vs-untuned geometry comparison used for the paper figures:

```sh
python reward_tune_analysis/scripts/export_snapshot_mesh_bank.py \
  --baseline-pkl /path/to/ffhq512-128.pkl \
  --tuned-pkl /path/to/network-snapshot-002068_LAST.pkl
```

The paper-facing export uses the `legacy_sigma10` geometry path (marching cubes
at sigma level 10), matching the user-study extraction.

## Reproducing the paper

For the full map of paper results to scripts/configs, see
[reproducing_paper.md](reproducing_paper.md).
