# Public Release Scope

This repo is a research codebase with a curated public surface.

The goal of this document is to state, plainly, what the public repo is
promising and what it is not.

## Core Public Surface

These are the maintained, paper-facing parts of the repo:

- reward-model training in
  `reward_model_training/reward_model_framework/core_modules/`
- EG3D RLHF finetuning in `eg3d/`
- maintained experiment configs for:
  - reward-model training
  - RLHF finetuning
- paper-facing export and verification scripts in
  `eg3d/reward_tune_analysis/scripts/`
- curated project docs in `docs/`

## Public But Optional

These can be useful to readers, but they are not part of the minimal
reproduction contract:

- curated analysis scripts in `eg3d/reward_tune_analysis/`
- notebook-heavy post-hoc analyses in `paper_result_analyses/`
- small summary artifacts in `paper_artifacts/reward_embedding_analysis/`
- additional exploratory comparisons and diagnostics

These should be treated as optional analysis material, not the main entry path.
See:
[optional_analysis_surface_2026-06-24.md](optional_analysis_surface_2026-06-24.md)
[public_artifact_policy_2026-06-24.md](public_artifact_policy_2026-06-24.md)

## Outside The Public Contract

These are not fully provided or guaranteed by this repo alone:

- large reward-model training data
- pretrained EG3D checkpoints used as finetune starting points
- trained reward-model checkpoints used in the reported runs
- released fine-tuned EG3D snapshot bundles beyond the curated subset we choose
  to publish
- full run directories and paper export outputs
- preference-collection / user-study assets beyond the clean runtime surface
- the full local reward-model archive under `RWD_MODELS_FOR_TUNING/`

## Reproducibility Contract

What this public repo supports directly:

- retraining reward models from external geometry data
- finetuning EG3D from an external pretrained checkpoint
- exporting before/after mesh banks from external snapshot `.pkl` files
- running maintained smoke and verification commands locally without WandB

What it does not promise by itself:

- one-command reproduction of every reported result from raw assets
- redistribution of all paper data and checkpoints
- redistribution of every historical finetune snapshot or run directory
- a polished general-purpose library interface

## Reader Guidance

If you want to understand or verify the project, use this order:

1. [README.md](../README.md)
2. [paper_results_guide_2026-06-24.md](paper_results_guide_2026-06-24.md)
3. [run_quickstart.md](run_quickstart.md)
4. the maintained configs and launcher scripts

## Release Intent

This repo is intended to show:

- a complete end-to-end research system
- explicit method/config lineage
- practical refactoring and cleanup of a complex ML codebase
- local-first reproducibility for the maintained public paths
