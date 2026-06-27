#!/usr/bin/env bash
# Run the five protected EG3D RLHF finetune configs for a single short verification
# tick, writing real run dirs under the standard training_runs_2 outdir.
#
# Usage:
#   bash reward_tune_analysis/scripts/run_protected_finetune_one_tick.sh
# Optional env override:
#   EG3D_TRAINING_OUTDIR=/path/to/training_runs_2
#
# Notes:
# - All protected configs currently resolve to tune_type=clamped.
# - This is a verification harness, not a reported experiment launcher.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$SCRIPT_DIR/../.."      # -> eg3d/

ENV=hf_geom_eg3d_py39
OUT="${EG3D_TRAINING_OUTDIR:-$HOME/training_runs_2}"
LOGDIR="${EG3D_TRAINING_LOGDIR:-$REPO_ROOT/release_verification_outputs/eg3d_finetune_logs/protected_one_tick}"
mkdir -p "$LOGDIR"

run_one () {                    # $1=experiment
  local exp="$1"
  local log="$LOGDIR/${exp}.log"
  echo "==================== ${exp} ===================="
  conda run --no-capture-output -n "$ENV" python train_rlhf.py \
    experiment="$exp" \
    train_tick_stop=1 \
    network_snapshot_ticks=9 \
    using_wandb=false \
    EVALUATE_METRICS=false \
    eval_metrics_only=false \
    export_first_images=false \
    plot_rwd_dist=true \
    +render_final_vis=false \
    click_legacy_args.outdir="$OUT" > "$log" 2>&1
  echo "exit=$? -> $log"
}

run_one finetune_eg3d_null
run_one finetune_eg3d_sfield
run_one finetune_eg3d_sdmap
run_one finetune_eg3d_tdmap
run_one finetune_eg3d_pn1

echo
echo "Protected one-tick finetune verification complete. Run dirs under $OUT"
