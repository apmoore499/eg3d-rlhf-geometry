Release-facing shell entrypoints live here.

These scripts are intentionally simple:

- activate the `hf_geom_eg3d_py39` environment first;
- run them from the repo root with `bash scripts/<name>.sh`;
- edit the command-line overrides inline if you want a different batch size,
  output directory, or reward-model id.

Current entrypoints:

- `generate_reward_training_data.sh` rebuilds the reward-model inputs.
- `test_reward_models.sh` runs the small public release reward-model test sweep.
- `train_reward_models.sh` runs the maintained reward-model training sweep.
- `finetune_eg3d_reported_sfield.sh` launches the reported 20-tick sigma-field tune.
- `test_eg3d_finetuning.sh` runs the short 1-tick finetune test sweep.
- `finetune_eg3d_all_reward_models.sh` fine-tunes EG3D against each maintained reward model.
- `verify_public_release.sh` runs the maintained public release verifier.

Historical or cluster-specific launchers belong under `scripts/_archive/`, not in
the framework subdirectories.
