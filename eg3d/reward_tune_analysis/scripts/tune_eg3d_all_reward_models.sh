#!/usr/bin/env bash
# ============================================================================
# Reproducibly fine-tune eg3d against each of the 6 reward models, exercising the
# full RLHF harness end-to-end (one short tune per dtype, train_tick_stop=1).
#   sigma_field_256 -> finetune_eg3d_sfield
#   single_dmap     -> finetune_eg3d_sdmap
#   triple_dmap     -> finetune_eg3d_tdmap
#   point_cloud_entire (pnet1/pnet2/curvenet) -> finetune_eg3d_pn1  (dtype is read from the
#       reward-model id, so one config serves all three; CurveNet needs batch_size=2)
#
# Reward-model ids come from reward_model_ids.env (written by
# train_all_reward_models.sh). If that file is absent, the defaults below (the most
# recently trained ids) are used. Bump TICKS for a longer tune.
#
# Usage (env hf_geom_eg3d_py39):  bash reward_tune_analysis/scripts/tune_eg3d_all_reward_models.sh
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$(dirname "$0")/../.."      # -> eg3d/

ENV=hf_geom_eg3d_py39
OUT="${EG3D_RLHF_TRAINING_RUNS_DIR:-$REPO_ROOT/paper_artifacts/_external_training_runs}"
TICKS=1                         # train_tick_stop (0 = one tick; raise for a real tune)

# Defaults (overridden by reward_model_ids.env if present) --------------------
RWD_SIGMA=wkc6e2f7              # sigma_field_256
RWD_SDMAP=se9l433b             # single_dmap
RWD_TDMAP=zxmuq2m9             # triple_dmap
RWD_PNET1=541b13pt            # point_cloud_entire / PointNet1
RWD_PNET2=7pvkwpnz            # point_cloud_entire / PointNet2
RWD_CVNET=w4eberou            # point_cloud_entire / CurveNet

IDS_FILE="${IDS_FILE:-$REPO_ROOT/reward_model_ids.env}"
if [ -f "$IDS_FILE" ]; then
  echo "Using reward-model ids from $IDS_FILE"
  # shellcheck disable=SC1090
  source "$IDS_FILE"
fi

tune () {                      # $1=experiment  $2=rwd_id  $3=label  $4=extra overrides (optional)
  echo "==================== tuning: $3  (experiment=$1, rwd=$2) ===================="
  conda run --no-capture-output -n "$ENV" python train_rlhf.py \
    experiment="$1" rwd_model_id="$2" train_tick_stop="$TICKS" \
    click_legacy_args.outdir="$OUT" ${4:-}
}

tune finetune_eg3d_sfield "$RWD_SIGMA" "sigma_field_256"
tune finetune_eg3d_sdmap  "$RWD_SDMAP" "single_dmap"
tune finetune_eg3d_tdmap  "$RWD_TDMAP" "triple_dmap"
tune finetune_eg3d_pn1    "$RWD_PNET1" "PointNet1"
tune finetune_eg3d_pn1    "$RWD_PNET2" "PointNet2"
tune finetune_eg3d_pn1    "$RWD_CVNET" "CurveNet" "rlhf_tune_hpms.batch_size=2"

echo
echo "All 6 eg3d harness tunes done. Run dirs under $OUT"
