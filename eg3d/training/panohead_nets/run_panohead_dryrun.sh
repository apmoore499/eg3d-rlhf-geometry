#!/usr/bin/env bash
# PanoHead reward-tune DRY-RUN launcher.
# Runs the real training loop for a few iterations to verify the PanoHead
# arch + BiSeNet head-mask seg channel + reward (G_depth) phase all run.
#
# Overrides vs panohead_dryrun.yaml:
#   - rlhf_tune_hpms.batch_size=4   (PanoHead is heavy; fits a 4090)
#   - rlhf_tune_hpms.lambda_rwd_model=0.1  (exercise the G_depth reward phase)
#   - using_wandb=False             (no online logging for a smoke run)
#
# Run from the eg3d/ tree root inside the conda env.
set -euo pipefail

python train_rlhf.py \
  click_legacy_args=panohead_dryrun \
  rlhf_tune_hpms.batch_size=4 \
  rlhf_tune_hpms.lambda_rwd_model=0.1 \
  using_wandb=False
