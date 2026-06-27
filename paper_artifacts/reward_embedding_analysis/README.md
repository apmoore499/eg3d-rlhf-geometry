This directory holds the small tracked subset of paper-facing reward-analysis
artifacts that were worth keeping in the repo.

It is intentionally narrow:

- summary-level CSV / JSON outputs
- no heavy image banks
- no full run directories
- no bulky intermediate tensors

Current kept surface:

- `cross_generator_trainorient/hyplanehead/`
- `cross_generator_trainorient/panohead/`
- `cross_generator_trainorient/spherehead/`
- `eg3d_baseline/trunc_train/`
- `hyplanehead_reward_transfer/trunc0.70/`

Each subfolder contains:

- `per_seed_rewards.csv`
  Per-seed reward scores on the shared canonical seed bank.
- `summary.json`
  Aggregate statistics for the same run.

These files support the cross-generator reward-transfer analysis discussed in
the paper without dragging large local run-output trees into the repo.

Public-mirror rule:

- keep small, text-based summary artifacts here
- do not mirror bulky local render outputs or private run directories
- if a result needs large external assets, document the generating script
  instead of committing the assets
