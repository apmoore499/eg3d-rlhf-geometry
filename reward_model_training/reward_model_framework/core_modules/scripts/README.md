Project-level shell launchers for the maintained reward-model framework live
here.

Current contents:

- `generate_reward_training_data.sh`: maintained launcher for synthesizing the
  EG3D-derived reward-model training inputs used by the public path.
- `train_all_reward_models.sh`: full retraining sweep for the six maintained
  reward-model experiment families.
- `train_all_reward_models_1epoch.sh`: quick one-epoch sweep across the same
  maintained experiments.
- other small experiment or smoke launchers kept for local verification.

The public mirror should prefer this folder over flat shell scripts placed
directly under `core_modules/`.
