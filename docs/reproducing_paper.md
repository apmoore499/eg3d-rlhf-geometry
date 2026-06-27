# Reproducing the paper

This maps the paper's results to the code that produces them: which script or
config to run, what external assets it needs, and what it outputs. For the
step-by-step command details see [data_generation.md](data_generation.md),
[reward_models.md](reward_models.md), and [finetuning.md](finetuning.md).

## External assets

The repo ships the code and the ranked-preference metadata, but not the large
binary inputs:

- a pretrained EG3D FFHQ checkpoint (`ffhq512-128.pkl`, external — NVIDIA);
- the rendered reward-model training data (regenerate via
  [data_generation.md](data_generation.md));
- trained reward-model bundles (train them, or download the released `7wnzkgie`).

What the public repo supports end-to-end: retraining reward models, fine-tuning
EG3D from an external generator checkpoint, and exporting before/after mesh
banks from snapshot `.pkl` files.

> Full EG3D fine-tuning additionally requires the original FFHQ images, which
> are not redistributed and must be re-synthesised following the upstream EG3D
> repository. See the [README](../README.md#what-is-released).

## Core pipeline

### 1. Reward-model training — representation/backbone comparison

- Entry: `python -m core_modules.train_rwd_model experiment=<name>`
  (or the full sweep `core_modules/scripts/train_all_reward_models.sh`).
- Configs: `sfield_256` (reported), `sdmap`, `tdmap`,
  `pcd_pnet_point_cloud_entire`, `pcd_pnet2_point_cloud_entire`,
  `pcd_cvnet_point_cloud_entire`.
- Paper role: the representation and backbone comparison, and the selection of
  the sigma-field model as the reported reward.

### 2. EG3D RLHF fine-tuning — the main method

- Entry: `eg3d/train_rlhf.py`; reported launcher
  `eg3d/reward_tune_analysis/scripts/sfield_reported_run.sh`.
- Configs: `finetune_eg3d_sfield` (reported), `finetune_eg3d_sdmap`,
  `finetune_eg3d_tdmap`, `finetune_eg3d_pn1`, `finetune_eg3d_null` (control).
- Paper role: the main fine-tuning result, the reward-vs-control FID comparison,
  and the reward-trajectory / histogram outputs.

### 3. Before/after mesh-bank export — the geometry figures

- Entry: `eg3d/reward_tune_analysis/scripts/export_snapshot_mesh_bank.py`
  (`--baseline-pkl`, `--tuned-pkl`).
- Paper role: the tuned-vs-untuned before/after geometry figures.
- The paper-facing export uses the `legacy_sigma10` path (marching cubes at
  sigma level 10), matching the user-study geometry. The `cummax` mode is
  exploratory only and distorts geometry relative to the user-study path.

## Verification (fast correctness checks)

These are not paper figures, but they confirm the public surface runs.

Reward-model smoke (from `reward_model_training/reward_model_framework/`):

```sh
python -m core_modules.train_rwd_model experiment=sfield_256 \
  logger=csv callbacks=public_local using_wandb=false test=false \
  dloader.num_workers=0 trainer.max_epochs=1 \
  trainer.limit_train_batches=2 trainer.limit_val_batches=2 \
  data.dset_dict.proportion_of_data_to_use=0.02
```

Public release verifier (loads the released reward model, runs a one-batch
forward pass through each reward config, and a tiny mesh-bank export):

```sh
python core_modules/scripts/verify_public_release.py \
  --baseline-pkl /path/to/untuned.pkl --tuned-pkl /path/to/tuned.pkl
```

RLHF smoke (from `eg3d/`):

```sh
python train_rlhf.py experiment=finetune_eg3d_sfield +smoke=on \
  click_legacy_args.outdir=/tmp/eg3d_rlhf_smoke
```

## Optional analyses (not the day-one reproduction path)

The embedding/UMAP, SHAP attribution, cross-generator transfer, and additional
mesh-tail diagnostics in the paper's appendices are exploratory analyses, kept
under `paper_result_analyses/`. They are not part of the minimal reproduction.

## Suggested reading order

1. [README](../README.md) — setup and repo map.
2. This guide.
3. Reward-model configs under `core_modules/configs/experiment/`.
4. RLHF configs under `eg3d/training/rlhf_tune_configs/experiment/`.
5. The launcher/export scripts referenced above.
