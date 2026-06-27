# Paper Results Guide

This guide maps the maintained code surface to the paper-facing results.

It is intentionally narrow:

- what script or config to run
- what external assets are required
- what outputs to expect
- what is core versus optional

Use this together with:

- [README.md](../README.md)
- [docs/run_quickstart.md](run_quickstart.md)
- [docs/paper_release_audit_2026-06-24.md](paper_release_audit_2026-06-24.md)

## External Assets

These are not fully bundled in the repo:

- reward-model training data under `RWD_DATA_DIR` / `paths.rwd_data_dir`
- pretrained EG3D generator checkpoint for finetuning
- trained reward-model checkpoints used by RLHF finetuning
- large run directories and paper export outputs
- preference data / user-study artifacts beyond the clean runtime surface

The public repo supports:

- retraining reward models from external geometry data
- finetuning EG3D from an external pretrained generator checkpoint
- exporting before/after mesh banks from external snapshot `.pkl` files

## Core Reported Pipeline

### 1. Reward-model training

Main entrypoint:

- [train_all_reward_models.sh](../reward_model_training/reward_model_framework/core_modules/scripts/train_all_reward_models.sh)

Maintained experiment configs:

- `sfield_256`
- `sdmap`
- `tdmap`
- `pcd_pnet_point_cloud_entire`
- `pcd_pnet2_point_cloud_entire`
- `pcd_cvnet_point_cloud_entire`

Paper role:

- representation comparison
- backbone comparison
- final reward-model selection

Required assets:

- reward-model training data in `RWD_DATA_DIR`
- the static sigma-cropping configs already tracked in the repo

Expected outputs:

- Lightning/Hydra run directory for each training run
- checkpoints
- local logs or WandB logs, depending on config
- reward-model ids written to `reward_model_ids.env` by the sweep script

Notes:

- the paper-winning representation/backbone is `sfield_256`
- the public smoke/default local path is
  `logger=csv callbacks=public_local using_wandb=false`

### 2. EG3D RLHF finetuning

Main entrypoint:

- [train_rlhf.py](../eg3d/train_rlhf.py)

Maintained finetune configs:

- `finetune_eg3d_null`
- `finetune_eg3d_sfield`
- `finetune_eg3d_sdmap`
- `finetune_eg3d_tdmap`
- `finetune_eg3d_pn1`

Paper role:

- main RLHF finetuning method
- reported comparison across reward-model choices
- reward-trajectory / histogram style outputs

Required assets:

- pretrained EG3D FFHQ checkpoint
- trained reward-model checkpoint ids / configs
- finetune output directory via `click_legacy_args.outdir`

Expected outputs per run directory:

- `training_options.json`
- `hydra_cfg.yaml`
- network snapshots
- preview images
- optional reward histograms / reward CSVs
- optional mesh visualisations and top/bottom ranking exports

Main reported launcher:

- [sfield_reported_run.sh](../eg3d/reward_tune_analysis/scripts/sfield_reported_run.sh)

Verification harness:

- [run_protected_finetune_one_tick.sh](../eg3d/reward_tune_analysis/scripts/run_protected_finetune_one_tick.sh)

Notes:

- the active reported loss surface is documented in
  [rlhf_reward_loss_surface_2026-06-24.md](rlhf_reward_loss_surface_2026-06-24.md)
- the public/local path should use `using_wandb=false`

### 3. Before/after mesh-bank export

Main entrypoint:

- [export_snapshot_mesh_bank.py](../eg3d/reward_tune_analysis/scripts/export_snapshot_mesh_bank.py)

Paper role:

- before/after geometry figure generation
- tuned-vs-untuned visual comparison

Required assets:

- tuned snapshot `.pkl`
- untuned baseline `.pkl`

Expected outputs:

- `metadata.json`
- per-method image banks under `cummax/` and `legacy_sigma10/`
- `compare/` side-by-side before/after images
- contact sheets
- CSV manifest of exported images

Notes:

- the exporter currently supports two surface extraction methods:
  `cummax` and `legacy_sigma10`
- this is the maintained path for paper-style tuned-vs-untuned mesh exports

## Fast Verification Paths

These are not paper figures, but they are the maintained correctness checks for
the public repo surface.

### Reward-model smoke

From `reward_model_training/reward_model_framework/`:

```sh
python -m core_modules.train_rwd_model \
  experiment=sfield_256 \
  logger=csv callbacks=public_local using_wandb=false test=false \
  dloader.num_workers=0 trainer.max_epochs=1 \
  trainer.limit_train_batches=2 trainer.limit_val_batches=2 \
  data.dset_dict.proportion_of_data_to_use=0.02
```

### RLHF smoke

From `eg3d/`:

```sh
python train_rlhf.py experiment=finetune_eg3d_sfield \
  +smoke=on click_legacy_args.outdir=/tmp/eg3d_rlhf_smoke
```

## Optional / Appendix-Style Analyses

These exist, but they are not the day-one public reproduction path:

- notebook-heavy embedding / UMAP analysis
- SHAP / attribution analysis
- cross-generator and extension experiments
- additional mesh-tail diagnostics

Public recommendation:

- keep these clearly marked as optional analyses
- do not make them part of the minimal reproduction contract

## Suggested Reading Order

For a new reader:

1. README quickstart and repo map
2. this results guide
3. reward-model configs under `core_modules/configs/experiment/`
4. RLHF configs under `eg3d/training/rlhf_tune_configs/experiment/`
5. the maintained launcher/export scripts above
