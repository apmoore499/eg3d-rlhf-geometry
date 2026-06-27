This directory holds the remaining paper-linked and exploratory analysis
scripts.

This is now the canonical home for the repo's paper-linked analysis layer.
Older notes referred to this area as `jupyter_notebooks` and later
`analysis_scripts`.

This is no longer a dump of every historical debugging helper. Single-run
mesh-export one-offs, seed-200050 watch/probe scripts, and other resolved
floater-debugging residue have been pruned from the public-facing tree.

The kept surface is selective:

- paper-backed post-hoc analyses such as truncation studies, reward embedding
  / explainability, mesh tails, and cross-generator transfer
- a few exploratory scripts that still have clear explanatory value

The remaining subfolder is:

- `sigma_representation_checks/`
  Optional sigma-volume inspection helpers for slab crops and MRC exports.

If a script here is tied to a specific run directory, check whether there is a
maintained equivalent under `eg3d/reward_tune_analysis/` before treating it as
part of the supported public path.

Maintained scripts in this directory now share path defaults from
`paper_result_analyses/path_defaults.py`. The main environment variables are:

- `EG3D_RLHF_TRAINING_RUNS_DIR`
  Root containing external EG3D fine-tune run folders such as `01446-...`.
- `EG3D_RLHF_REPORTED_RUN_DIR`
  Explicit override for the main reported run directory when a script needs
  that run specifically.
- `EG3D_RLHF_PAPER_FIG_DIR`
  Output folder for generated paper figures and comparison JPGs.
- `EG3D_RLHF_DATASET_ZIP`
  Override for the EG3D dataset zip used by reward-embedding analyses.
- `EG3D_EXTERNAL_PROJECTS_ROOT`
  Parent directory containing sibling repos such as `PanoHead`,
  `HyPlaneHead`, and `SphereHead`.
- `PANOHEAD_ROOT`, `HYPLANEHEAD_ROOT`, `SPHEREHEAD_ROOT`
  Per-project overrides if those generator repos live somewhere else.
