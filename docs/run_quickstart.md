Quick run notes

- From repo root, run:
  `cd reward_model_training/reward_model_framework && python -m core_modules.train_rwd_model experiment=sfield_256 logger=csv callbacks=public_local using_wandb=false test=false dloader.num_workers=0 trainer.max_epochs=1 trainer.limit_train_batches=2 trainer.limit_val_batches=2 data.dset_dict.proportion_of_data_to_use=0.02`
- Prereqs: install requirements. The local `autoroot.py` bootstrap sets repo-root paths and env defaults for the moved reward-model framework.
- Outputs: Hydra run dir under `logs/` by default (see `core_modules/configs/paths/default.yaml`). For local smoke use, keep `logger=csv callbacks=public_local using_wandb=false test=false`. Expect local CSV metrics plus checkpoints under that run dir; no WandB account is required.
- Common knobs: override batch/worker counts via `dloader.*`, toggle testing with `test=true/false`, point to a checkpoint with `ckpt_path=<path>`, or switch experiment YAMLs via `experiment=<name>.yaml`.
- Maintained reward-model experiment configs: `sfield_256`, `sdmap`, `tdmap`, `pcd_cvnet_point_cloud_entire`, `pcd_pnet_point_cloud_entire`, `pcd_pnet2_point_cloud_entire`.
- Cleanup: temporary datamodule cache lives in `tmp_datamodules/`; delete if you need a fresh build.

## EG3D RLHF tuning

- From `eg3d/`, run: `python train_rlhf.py experiment=<exp>` (e.g. `finetune_eg3d_sfield`).
- Outputs: Hydra run dir under `click_legacy_args.outdir`. For local smoke and verification runs, keep `using_wandb=false`. A normal local run writes `training_options.json`, `hydra_cfg.yaml`, snapshots, preview images, and any enabled reward-histogram / mesh-export artifacts directly into that run directory.
- Maintained RLHF experiment configs: `finetune_eg3d_null`, `finetune_eg3d_sfield`, `finetune_eg3d_sdmap`, `finetune_eg3d_tdmap`, `finetune_eg3d_pn1`.

### Fast smoke runs

Apply the reusable smoke preset on top of any experiment to get a cheap, quiet
end-to-end pass that reaches `tick 0` and exits cleanly (~15s on a 4090):

```
python train_rlhf.py experiment=<exp> +smoke=on \
  click_legacy_args.outdir=/tmp/eg3d_rlhf_smoke
```

The preset (`training/rlhf_tune_configs/smoke/on.yaml`) shrinks the batch
(`batch_size=1`, `accum_steps=1`), stops at `tick 0` (`train_tick_stop=0`), and
disables the expensive reporting/eval paths (FID/KID, reward-distribution
plotting + combined epoch analysis, the preference-study render, W&B, and the
end-of-training mesh render). It deliberately keeps `init_seeds_first=true` so
the reward-loss path is still exercised. Any inline CLI override still wins over
the preset.

### Paper-facing entrypoints

- Reward-model full retrain sweep:
  `reward_model_training/reward_model_framework/core_modules/scripts/train_all_reward_models.sh`
- RLHF verification across the five maintained finetune configs:
  `eg3d/reward_tune_analysis/scripts/run_protected_finetune_one_tick.sh`
- Reported sigma-field finetune run:
  `eg3d/reward_tune_analysis/scripts/sfield_reported_run.sh`
- Before/after tuned-vs-untuned mesh-bank export:
  `eg3d/reward_tune_analysis/scripts/export_snapshot_mesh_bank.py`
