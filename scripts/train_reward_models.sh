#!/usr/bin/env bash
cd "$(dirname "$0")/../reward_model_training/reward_model_framework"

# Train the reported sigma-field reward model on the full dataset.
python -m core_modules.train_rwd_model experiment=sfield_256 prop_data=1.0 trainer.max_epochs=10 trainer.limit_train_batches=1.0 trainer.limit_val_batches=1.0 trainer.limit_test_batches=1.0

# Train the single-depth-map reward model on the full dataset.
python -m core_modules.train_rwd_model experiment=sdmap prop_data=1.0 trainer.max_epochs=10 trainer.limit_train_batches=1.0 trainer.limit_val_batches=1.0 trainer.limit_test_batches=1.0

# Train the triple-depth-map reward model on the full dataset.
python -m core_modules.train_rwd_model experiment=tdmap prop_data=1.0 trainer.max_epochs=10 trainer.limit_train_batches=1.0 trainer.limit_val_batches=1.0 trainer.limit_test_batches=1.0

# Train the PointNet reward model on the full dataset.
python -m core_modules.train_rwd_model experiment=pcd_pnet_point_cloud_entire prop_data=1.0 trainer.max_epochs=10 trainer.limit_train_batches=1.0 trainer.limit_val_batches=1.0 trainer.limit_test_batches=1.0

# Train the PointNet++ reward model on the full dataset.
python -m core_modules.train_rwd_model experiment=pcd_pnet2_point_cloud_entire prop_data=1.0 trainer.max_epochs=10 trainer.limit_train_batches=1.0 trainer.limit_val_batches=1.0 trainer.limit_test_batches=1.0

# Train the CurveNet reward model on the full dataset.
python -m core_modules.train_rwd_model experiment=pcd_cvnet_point_cloud_entire prop_data=1.0 trainer.max_epochs=10 trainer.limit_train_batches=1.0 trainer.limit_val_batches=1.0 trainer.limit_test_batches=1.0
