# Paper-Backed Release Audit

This audit works backward from the current arXiv manuscript and asks a narrow
question: what must be clean, documented, and runnable for a public
paper-aligned release?

## Readiness Summary

- `arXiv`: close
- public `GitHub`: not yet

The core method code now exists in a much cleaner state than at the start of
the refactor, and the main EG3D fine-tune configs were verified again during
this cleanup pass. The main remaining gap is not "does the method run?" but
"does the public repo present a coherent, local-first, paper-backed release
surface?"

## Core Release Matrix

| Paper area | Current code anchor | Status | Main release risk | Action |
|---|---|---|---|---|
| Reward-model training method and main representation comparison (`sec:method:reward`, `tab:repaccuracy`) | `reward_model_training/reward_model_framework/core_modules/train_rwd_model.py`, `core_modules/configs/experiment/sfield_256.yaml`, `sdmap.yaml`, `tdmap.yaml`, `pcd_*_point_cloud_entire.yaml`, `core_modules/scripts/train_all_reward_models.sh` | Core training path exists and is still runnable | Public docs are stale; data/checkpoint contract is not explicit | Update README/quickstart to current experiment names; document required external data and expected outputs |
| EG3D RLHF fine-tuning method (`sec:method:finetune`, `tab:fid`, `fig:reward_hist`, `fig:reward_traj`, `fig:reward_convergence`) | `eg3d/train_rlhf.py`, `eg3d/training/training_loop.py`, `eg3d/training/loss.py`, protected finetune configs under `eg3d/training/rlhf_tune_configs/experiment/` | Active path is much cleaner and was re-smoked | Public quickstart still points to stale experiment names; logging is still WandB-shaped | Replace stale commands in README/docs; make local/no-WandB operation the documented default |
| Before/after mesh visuals (`fig:before_after`, `fig:beforeafter`, `fig:rgb_stability`) | `eg3d/reward_tune_analysis/scripts/export_snapshot_mesh_bank.py`, `eg3d/training/render_final_snapshot_vis.py` | Now has a coherent exporter path | Not yet documented as a paper-facing asset pipeline | Add one short documented command for producing the paper-style before/after mesh bank |
| Reported finetune lineage | `finetune_eg3d_null.yaml`, `finetune_eg3d_sfield.yaml`, `finetune_eg3d_sdmap.yaml`, `finetune_eg3d_tdmap.yaml`, `finetune_eg3d_pn1.yaml` | Good | README/docs still reference removed legacy config names | Replace stale references with the protected configs only |
| User study (`tab:userstudy`) | Ranking / preference data lives outside the clean runtime surface | Research artifact exists, but not yet curated for release | Reproducibility is unclear; likely includes non-repo assets and manual workflow | Decide whether this is public, partially documented, or stated as non-redistributable |
| Post-hoc embedding / attribution analysis (`fig:embed_umap`, `fig:embed_umap_reward`, `fig:shap`, `fig:region_shapley`) | `paper_result_analyses/*umap*`, `*shap*`, `*reward_attribution*`, various analysis outputs | Exists, but mainly as research notebooks/scripts | Notebook-heavy, machine-specific, not curated as public entrypoints | Either curate as optional analysis notebooks or clearly mark as non-core research extras |
| Truncation baseline / cross-generator transfer / mesh tails (`tab:truncation_baseline`, `tab:trunc_sweep`, `fig:pointnet_topbottom`, `fig:geom_reward_compare`, `fig:panohead_mesh`, `fig:sigma_histogram`, `fig:unstratified_mesh_tails`) | Mix of `paper_result_analyses/`, `reward_tune_analysis/`, and saved outputs | Technical work exists | Not assembled into a clean public reproduction path | Create a dedicated `paper_analysis` guide or keep these as optional appendix material |
| SphereHead / PanoHead extensions (`fig:spherehead_inversion`, `tab:spherehead_inversion`) | Notebook / analysis layer, not main EG3D path | Exists, but not core-release clean | Additional external assets and ad hoc scripts | Treat as optional extension, not day-one public release requirement |

## Immediate Public-Release Blockers

### 1. Stale public quickstarts

The current public-facing docs still point at old config names:

- [README.md](../README.md) uses `experiment=3dconv_net_sigma_256_unet3d`
- [README.md](../README.md) uses `experiment=eg3d_sf256_7wnzkgie_winner_singleiter.yaml`
- [docs/run_quickstart.md](run_quickstart.md) and [docs/run_quickstart.md](run_quickstart.md) still describe the same outdated surface

These are now wrong enough to confuse a public reader about what the maintained
entrypoints actually are.

### 2. Machine-specific paths still exist in the active surface

The repo still has live `/home/krillman/...` and `/media/krillman/...`
assumptions in core or semi-core files, for example:

- [eg3d/training/loss.py](../eg3d/training/loss.py)
- [eg3d/reward_tune_analysis/scripts/sfield_reported_run.sh](../eg3d/reward_tune_analysis/scripts/sfield_reported_run.sh)
- [eg3d/reward_tune_analysis/scripts/run_protected_finetune_one_tick.sh](../eg3d/reward_tune_analysis/scripts/run_protected_finetune_one_tick.sh)
- [reward_model_training/reward_model_framework/core_modules/configs/paths/default.yaml](../reward_model_training/reward_model_framework/core_modules/configs/paths/default.yaml)
- [reward_model_training/reward_model_framework/core_modules/data/io_geometry_utils.py](../reward_model_training/reward_model_framework/core_modules/data/io_geometry_utils.py)
- [reward_model_training/reward_model_framework/core_modules/scripts/train_all_reward_models.sh](../reward_model_training/reward_model_framework/core_modules/scripts/train_all_reward_models.sh)

For a public release, these should become:

- env vars
- Hydra `paths.*` config
- CLI arguments
- or clearly private/internal scripts outside the public path

### 3. WandB is still structurally important

This is the most important release-ops issue.

#### Reward-model side

The reward-model framework already has logger abstraction:

- [core_modules/configs/train.yaml](../reward_model_training/reward_model_framework/core_modules/configs/train.yaml) defaults to `logger: wandb`
- [core_modules/configs/logger/mlflow.yaml](../reward_model_training/reward_model_framework/core_modules/configs/logger/mlflow.yaml) exists already
- `csv` and `tensorboard` logger configs also exist

But `train_rwd_model.py` still contains raw WandB lifecycle and artifact logic:

- [train_rwd_model.py](../reward_model_training/reward_model_framework/core_modules/train_rwd_model.py)
- [train_rwd_model.py](../reward_model_training/reward_model_framework/core_modules/train_rwd_model.py)
- [train_rwd_model.py](../reward_model_training/reward_model_framework/core_modules/train_rwd_model.py)

So this side is "logger-pluggable, but still WandB-shaped."

#### EG3D RLHF side

The EG3D fine-tune path is more tightly coupled:

- [eg3d/train_rlhf.py](../eg3d/train_rlhf.py) imports `wandb`
- [eg3d/train_rlhf.py](../eg3d/train_rlhf.py) directly calls `wandb.init(...)`

The code can run with `using_wandb=false`, and that path was re-verified during
the refactor, but the public release surface is still conceptually built around
WandB being the primary run tracker.

### 4. The paper-to-code map is not yet exposed to readers

The repo has the method code, configs, and many figure-analysis scripts, but it
does not yet tell a public reader:

- which scripts reproduce the main paper results
- which analyses are appendix / optional
- which data and checkpoints are external
- which results are not redistributable end-to-end

### 5. The repo still mixes core code and research residue

There is much less clutter than before, but the public surface still includes:

- notebook-driven analysis
- historical handover docs
- internal run scripts
- private-path assumptions
- old output dumps and run artifacts in local trees

For a resume/portfolio release, this should be intentionally curated rather
than merely "mostly working."

## Recommendation On WandB vs MLflow

Do **not** start by migrating the whole project from WandB to MLflow.

That is a larger project than is necessary for a strong paper-aligned public
release, and it risks turning a presentational cleanup into an infrastructure
rewrite.

### Better sequence

1. Make WandB **non-essential** in the public path.
2. Keep local run metadata and outputs first-class.
3. Only then decide whether MLflow is worth adding as the preferred local UI.

### Practical recommendation

#### Reward-model framework

Keep the existing logger abstraction and switch the public/default release path
to a local logger:

- `logger=mlflow`, or
- `logger=csv`, or
- `logger=tensorboard`

Then remove the remaining assumption that the persistent run folder must be
keyed by `wandb.run.id`.

#### EG3D RLHF fine-tuning

Do not add full MLflow first. Add a small local-first run manifest/output layer
instead:

- local run ID
- saved Hydra config
- summary JSON / CSV
- local image and mesh artifacts

After that, WandB can become optional backend logging rather than a structural
dependency.

### Bottom line

`MLflow` is a reasonable later improvement, especially for the reward-model
side, because the config surface already anticipates it. But the immediate
problem is **mandatory WandB shape**, not absence of MLflow.

## Recommended Next Tasks

1. Update [README.md](../README.md) and [docs/run_quickstart.md](run_quickstart.md) to current config names and current smoke commands only.
2. Add one release-facing doc that maps paper claims to code entrypoints and states what is external.
3. Remove or parameterize machine-specific paths from the active public surface.
4. Make the public/default run path local-first and non-WandB for both frameworks.
5. Curate a small `paper_results/` or `paper_analysis/` guide for the main figures and tables.
6. Decide what stays private or appendix-only:
   - user study assets
   - notebook-heavy attribution work
   - cross-generator extras
7. Publish from a curated release branch or public mirror, not from the raw research working branch.

## Publishability Judgment

If the goal is:

- `arXiv` submission: the project is close
- strong public `GitHub` portfolio release: one more focused release-hardening pass is needed

That pass is now much smaller than the original refactor. It is mainly:

- doc correction
- path sanitization
- logger decoupling
- and intentional presentation

not another large runtime cleanup.
