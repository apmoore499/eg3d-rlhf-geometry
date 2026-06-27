#!/usr/bin/env bash
# Train the 6 maintained reward models for a single epoch on the full data.
# Run from anywhere with env hf_geom_eg3d_py39 available.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORE_MODULES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CORE_MODULES_DIR"

python train_rwd_model.py experiment=sfield_256 prop_data=1.0 trainer.max_epochs=1 trainer.limit_train_batches=1.0 trainer.limit_val_batches=1.0 trainer.limit_test_batches=1.0
python train_rwd_model.py experiment=sdmap prop_data=1.0 trainer.max_epochs=1 trainer.limit_train_batches=1.0 trainer.limit_val_batches=1.0 trainer.limit_test_batches=1.0
python train_rwd_model.py experiment=tdmap prop_data=1.0 trainer.max_epochs=1 trainer.limit_train_batches=1.0 trainer.limit_val_batches=1.0 trainer.limit_test_batches=1.0
python train_rwd_model.py experiment=pcd_pnet_point_cloud_entire prop_data=1.0 trainer.max_epochs=1 trainer.limit_train_batches=1.0 trainer.limit_val_batches=1.0 trainer.limit_test_batches=1.0
python train_rwd_model.py experiment=pcd_pnet2_point_cloud_entire prop_data=1.0 trainer.max_epochs=1 trainer.limit_train_batches=1.0 trainer.limit_val_batches=1.0 trainer.limit_test_batches=1.0
python train_rwd_model.py experiment=pcd_cvnet_point_cloud_entire prop_data=1.0 trainer.max_epochs=1 trainer.limit_train_batches=1.0 trainer.limit_val_batches=1.0 trainer.limit_test_batches=1.0
