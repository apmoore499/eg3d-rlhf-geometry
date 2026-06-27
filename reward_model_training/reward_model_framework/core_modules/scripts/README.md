Project-level shell launchers for the maintained reward-model framework live
here.

Current contents:

- `generate_reward_training_data.sh`: maintained launcher for synthesizing the
  EG3D-derived reward-model training inputs used by the public path.
- `run_public_release_verifier.sh`: single-command wrapper around the maintained
  public release verifier. By default it writes outputs under
  `release_verification_outputs/public_release_smoke/` in the public worktree.
- `verify_public_release.py`: maintained public release verifier for 2-seed
  data generation, loader smoke, maintained reward-model forward smokes,
  released reward-model checkpoint loading, and optional tuned-vs-untuned
  mesh-bank export.
- `train_all_reward_models.sh`: full retraining sweep for the six maintained
  reward-model experiment families.
- `train_all_reward_models_1epoch.sh`: quick one-epoch sweep across the same
  maintained experiments.
- other small experiment or smoke launchers kept for local verification.

The public mirror should prefer this folder over flat shell scripts placed
directly under `core_modules/`.
