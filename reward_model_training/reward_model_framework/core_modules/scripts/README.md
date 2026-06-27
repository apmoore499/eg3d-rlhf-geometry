Module-level maintenance scripts for the reward-model framework live here.

Current contents:

- `generate_reward_training_data.sh`: framework-local launcher for rebuilding
  the reward-model training inputs.
- `run_public_release_verifier.sh`: framework-local wrapper around the
  maintained public release verifier. By default it writes outputs under
  `release_verification_outputs/public_release_test/` in the public worktree.
- `verify_public_release.py`: maintained public release verifier for 2-seed
  data generation, loader tests, maintained reward-model forward tests,
  released reward-model checkpoint loading, and optional tuned-vs-untuned
  mesh-bank export.
- `train_all_reward_models.sh`: full retraining sweep for the six maintained
  reward-model experiment families.
- `train_all_reward_models_1epoch.sh`: quick one-epoch sweep across the same
  maintained experiments.

The public mirror should prefer the release-facing scripts under the repo-root
`scripts/` directory. Keep this folder for framework-local helpers only.
