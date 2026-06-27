# Optional Analysis Surface

This document classifies the exploratory analysis layer of the repo.

It exists so a public reader can distinguish:

- core reported pipeline code
- curated optional analysis
- legacy or one-off exploratory work

See also:

- [public_release_scope_2026-06-24.md](public_release_scope_2026-06-24.md)
- [paper_results_guide_2026-06-24.md](paper_results_guide_2026-06-24.md)

## Curated Optional Analysis

The clearest optional analysis surface is:

- `eg3d/reward_tune_analysis/`

This contains project-local scripts that are still useful, but are not part of
the minimal reproduction contract.

Main categories there:

- snapshot / representation comparison:
  - `compare_snapshot_*`
  - `analyze_snapshot_svd.py`
  - `plot_snapshot_spectrum_profiles.py`
- reward-conditioned structure analysis:
  - `analyze_reward_conditioned_*`
  - `analyze_sigma_field_regions.py`
  - `analyze_sigma_mesh_diffs.py`
- hygiene / utility:
  - `find_corrupted_images.py`
- maintained export / verification scripts:
  - `scripts/export_snapshot_mesh_bank.py`
  - `scripts/run_protected_finetune_one_tick.sh`
  - `scripts/sfield_reported_run.sh`

These are public-optional: useful for deeper inspection, not required for the
main paper-backed runtime path.

## Notebook-Heavy Exploratory Layer

`paper_result_analyses/` is now the canonical home for the smaller selective
subset of the old exploratory tree. The local `jupyter_notebooks/` symlink is
kept only as a compatibility alias:

- truncation / baseline comparison scripts
- reward embedding / explainability analysis
- PanoHead / extension experiments
- selected mesh-tail / cross-generator diagnostics
- a few explicitly archived optional subfolders for non-paper side analyses

It is still exploratory and should not be treated as the main public
entrypoint, but the most obvious single-run debugging residue has already been
pruned.

Representative categories:

- embedding / explainability:
  - `reward_embedding_tsne_analysis.py`
  - `reward_embedding_tail_signal_analysis.py`
  - `reward_geometry_explainability.py`
  - `panohead_reward_attribution.py`
- truncation and baseline studies:
  - `eg3d_orig_reward_vs_truncation.py`
  - `eg3d_tuned_reward_vs_truncation.py`
- sigma-representation inspection:
  - `sigma_representation_checks/`
- archived non-paper side analyses:
  - `legacy_optional/demographic_checks/`
- paper asset helpers:
  - `paper_asset_utils/`

## Legacy / Non-Core Material

Readers should treat these as historical or exploratory, not maintained public
API:

- temporary objects like `tmp.obj`
- scripts written for a single run directory or a single snapshot id
- archived demographic / accessory-confound checks that did not become part of
  the main paper result surface

## Public Recommendation

For portfolio / release presentation:

- keep `reward_tune_analysis/` visible as optional analysis
- keep `paper_result_analyses/` clearly labelled exploratory
- keep `jupyter_notebooks/` only as a compatibility alias
- do not imply that every notebook/script is part of the supported public path

## Reader Guidance

If you are evaluating the project:

1. start with the core runtime and paper-results guide
2. use `reward_tune_analysis/` for deeper but still curated follow-up analysis
3. only then browse `paper_result_analyses/` as exploratory research residue
