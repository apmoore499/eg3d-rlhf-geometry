#!/usr/bin/env bash
# Reported sigma-field tune: 01446-effective reward implemented intentionally
#   L_reward = -lambda * clip(r_phi(x), -c, c),  lambda=10, c=10, raw score (no standardisation)
# Fresh tune from the original EG3D FFHQ checkpoint (base resume, kimg 2048), 10 ticks.
#   bash reward_tune_analysis/scripts/sfield_reported_run.sh   (env hf_geom_eg3d_py39)
# Optional env override:
#   EG3D_TRAINING_OUTDIR=/path/to/training_runs_2
set -uo pipefail
cd "$(dirname "$0")/../.."      # -> eg3d/

ENV=hf_geom_eg3d_py39
OUT="${EG3D_TRAINING_OUTDIR:-$HOME/training_runs_2}"
LOGDIR=/tmp/sfield_reported
mkdir -p "$LOGDIR"

conda run --no-capture-output -n "$ENV" python train_rlhf.py \
  experiment=finetune_eg3d_sfield rwd_model_id=7wnzkgie \
  train_tick_stop=10 \
  using_wandb=false EVALUATE_METRICS=false \
  network_snapshot_ticks=10 export_first_images=true \
  +render_final_vis=false \
  'seedslist_visualisation=[2,3,4,5,6]' \
  click_legacy_args.outdir="$OUT" > "$LOGDIR/run.log" 2>&1
echo "exit=$? -> $LOGDIR/run.log"
