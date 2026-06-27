#!/usr/bin/env bash
# ============================================================================
# Reproducibly (re)train the 6 reward models used for eg3d RLHF tuning.
#   - sigma_field_256, single_dmap, triple_dmap
#   - point_cloud_entire x {PointNet1, PointNet2, CurveNet}
# 10 epochs, full dataset. Each saves to
#   core_modules/RWD_MODELS_FOR_TUNING/<wandb_id>/  (best_model.pt + run_config.yaml,
# with the tune-augmentation slot bundled in so eg3d can inherit it).
#
# The resulting reward-model ids are appended to reward_model_ids.env at the repo
# root; tune_eg3d_all_reward_models.sh (Stage 2) sources that file.
#
# Usage (env hf_geom_eg3d_py39):
#   bash reward_model_training/reward_model_framework/core_modules/scripts/train_all_reward_models.sh
# The experiment configs below default to smoke-scale trainer limits (1 batch);
# the overrides here restore full 10-epoch training. EarlyStopping (patience 3)
# may end a model sooner -- raise callbacks.early_stopping.patience to force 10.
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORE_MODULES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$CORE_MODULES_DIR"

ENV=hf_geom_eg3d_py39
IDS_FILE="${IDS_FILE:-$REPO_ROOT/reward_model_ids.env}"
RWD_MODELS_DIR="${RWD_MODELS_DIR:-$CORE_MODULES_DIR/RWD_MODELS_FOR_TUNING}"
OVERRIDES="prop_data=1.0 trainer.max_epochs=10 trainer.limit_train_batches=1.0 trainer.limit_val_batches=1.0 trainer.limit_test_batches=1.0"

: > "$IDS_FILE"                 # fresh ids file

train_one () {                 # $1=experiment  $2=ID_VAR_NAME  $3=label
  echo "==================== training: $3  (experiment=$1) ===================="
  conda run --no-capture-output -n "$ENV" python train_rwd_model.py experiment="$1" $OVERRIDES
  local id
  id=$(ls -1dt "$RWD_MODELS_DIR"/*/ | head -1 | xargs -n1 basename)
  echo "$2=$id" >> "$IDS_FILE"
  echo "==> $3 saved as reward-model id: $id"
}

train_one sfield_256                    RWD_SIGMA "sigma_field_256"
train_one sdmap                         RWD_SDMAP "single_dmap"
train_one tdmap                         RWD_TDMAP "triple_dmap"
train_one pcd_pnet_point_cloud_entire   RWD_PNET1 "point_cloud_entire / PointNet1"
train_one pcd_pnet2_point_cloud_entire  RWD_PNET2 "point_cloud_entire / PointNet2"
train_one pcd_cvnet_point_cloud_entire  RWD_CVNET "point_cloud_entire / CurveNet"

echo
echo "All 6 reward models trained. Ids written to $IDS_FILE :"
cat "$IDS_FILE"
