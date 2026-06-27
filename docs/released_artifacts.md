# Released Artifacts

This document defines the curated checkpoint artifacts intended for the public
GitHub release of this project.

## Release Channel

The primary release channel should be **GitHub Releases** for the public repo.
This project's released checkpoints are each below GitHub's per-asset `2 GiB`
limit, so GitHub Releases provides a simple public download path without
requiring a Hugging Face account.

## Curated Release Assets

### 1. Sigma reward-model bundle

Recommended asset filename:

- `reward-model-7wnzkgie-sfield256.zip`

Purpose:

- released sigma-volume reward model used by the reported EG3D RLHF fine-tuning
  path
- verified by the public release smoke through a real checkpoint load and real
  forward pass

Recommended zip layout:

```text
7wnzkgie/
  best_model.pt
  release_config.yaml
  datamodule_third.pt            # optional
  model_example_input.pt         # optional
```

Minimal contents required by the maintained public path:

- `best_model.pt`
- `release_config.yaml`

Optional helper files that can be bundled for inspection/debugging but are not
required for loading or forward inference:

- `datamodule_third.pt`
- `model_example_input.pt`

Deprecated file that should not be the public loading contract:

- `run_config.yaml`

Contents **not** required for the maintained public runtime and therefore not
necessary to release by default:

- `val_gv_epoch_*.csv`

Current local source bundle:

- `reward_model_training/reward_model_framework/core_modules/RWD_MODELS_FOR_TUNING/7wnzkgie/`

Approximate size of required files:

- `best_model.pt`: `49 MB`
- `release_config.yaml`: small text yaml
- `datamodule_third.pt`: `99 KB` (optional)
- `model_example_input.pt`: `27 MB` (optional)

### 2. Fine-tuned EG3D snapshot

Recommended asset filename:

- `eg3d-finetuned-sfield-run01446-network-snapshot-002068_LAST.pkl`

Purpose:

- reported fine-tuned EG3D snapshot used for tuned-vs-untuned geometry export
- consumed by the public mesh-bank export and public release verifier

Current local source file:

- `/media/krillman/240GB_DATA/training_runs_2/01446-ffhq-eg3d_w_mirrore-gpus1-batch16-gamma20/network-snapshot-002068_LAST.pkl`

Approximate size:

- `361 MB`

### 3. Untuned baseline EG3D snapshot

This is a required dependency, but it does **not** need to be mirrored as a
project-owned GitHub Release asset if you prefer to point users at the existing
upstream/public checkpoint.

Baseline expected by the maintained public path:

- `ffhq512-128.pkl`

Current local path used in verification:

- `pkl_pt/eg3d_1/ffhq512-128.pkl`

Upstream source:

- NVIDIA NGC EG3D checkpoints

## Expected Download Layout

Recommended layout after downloading the public release assets:

```text
external_assets/
  eg3d/
    ffhq512-128.pkl
    eg3d-finetuned-sfield-run01446-network-snapshot-002068_LAST.pkl
  reward_models/
    7wnzkgie/
      best_model.pt
      release_config.yaml
      datamodule_third.pt
      model_example_input.pt
```

## Script Expectations

### Public release verifier

Script:

- `reward_model_training/reward_model_framework/core_modules/scripts/run_public_release_verifier.sh`

Inputs:

- tuned EG3D snapshot `.pkl`
- untuned baseline EG3D snapshot `.pkl`
- released reward-model directory containing `7wnzkgie/`
- the maintained loader prefers `release_config.yaml` inside that bundle

Environment / arguments:

- first positional argument: tuned `.pkl`
- `PUBLIC_RELEASE_VERIFY_BASELINE_PKL`: optional override for untuned baseline
- `PUBLIC_RELEASE_VERIFY_RWD_MODELS_DIR` or `RWD_MODELS_DIR`: parent directory
  containing `7wnzkgie/`

### Mesh-bank export

Script:

- `eg3d/reward_tune_analysis/scripts/export_snapshot_mesh_bank.py`

Inputs:

- `--baseline-pkl`
- `--tuned-pkl`

Public-path note:

- the paper-facing export should use `legacy_sigma10`
- `cummax` is exploratory only and should not be treated as the public
  user-study geometry path

### RLHF fine-tuning

Main runtime uses:

- untuned baseline EG3D checkpoint as the generator starting point
- released reward-model bundle as the scoring model

The exact fine-tune config lineage for the reported sigma run is documented in:

- `docs/paper_results_guide_2026-06-24.md`
- `docs/rlhf_reward_loss_surface_2026-06-24.md`

## Public Release Notes

When creating the GitHub Release, include a short note that says:

- `reward-model-7wnzkgie-sfield256.zip` is the released sigma-volume reward
  model bundle, with `best_model.pt` + `release_config.yaml` as the maintained
  loading contract
- `eg3d-finetuned-sfield-run01446-network-snapshot-002068_LAST.pkl` is the
  released tuned EG3D snapshot used in the paper-aligned export path
- `ffhq512-128.pkl` is the paired untuned baseline expected by the maintained
  scripts
