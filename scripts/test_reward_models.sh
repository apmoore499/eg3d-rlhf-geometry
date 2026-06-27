#!/usr/bin/env bash
cd "$(dirname "$0")/../reward_model_training/reward_model_framework"

# Test the reported sigma-field reward model on one batch per split with 5% of the data.
python -m core_modules.train_rwd_model experiment=sfield_256 using_wandb=false trainer.max_epochs=1 trainer.limit_train_batches=1 trainer.limit_val_batches=1 trainer.limit_test_batches=1 prop_data=0.05 model.act_type=softplus

# Test the single-depth-map reward model on one batch per split with 5% of the data.
python -m core_modules.train_rwd_model experiment=sdmap using_wandb=false trainer.max_epochs=1 trainer.limit_train_batches=1 trainer.limit_val_batches=1 trainer.limit_test_batches=1 prop_data=0.05 model.act_type=softplus

# Test the triple-depth-map reward model on one batch per split with 5% of the data.
python -m core_modules.train_rwd_model experiment=tdmap using_wandb=false trainer.max_epochs=1 trainer.limit_train_batches=1 trainer.limit_val_batches=1 trainer.limit_test_batches=1 prop_data=0.05 model.act_type=softplus

# Test the maintained point-cloud reward model on one batch per split with 5% of the data.
python -m core_modules.train_rwd_model experiment=pcd_pnet_point_cloud_entire using_wandb=false trainer.max_epochs=1 trainer.limit_train_batches=1 trainer.limit_val_batches=1 trainer.limit_test_batches=1 prop_data=0.05 model.act_type=softplus
