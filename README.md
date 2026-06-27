# RLHF for 3D Face Geometry (EG3D + learned reward model)

Improving the **3D geometry** of an EG3D face generator with reinforcement
learning from human feedback (RLHF). EG3D produces gorgeous multi-view images,
but the underlying density field often contains geometric artefacts ("floaters",
distorted surfaces) that the 2D renders hide. This project learns a **reward
model from human preference rankings over generated meshes**, then **fine-tunes
the EG3D generator** to maximise that reward — yielding cleaner geometry while
preserving image quality.

This is research code from my PhD (Generative AI, mathematics). It accompanies
<!-- TODO: paper/thesis title + link (CVPR 2025 submission) -->.

> **Status — research code, honestly labelled.** This is a fork of
> [EG3D](https://github.com/NVlabs/eg3d) plus a reward-model training framework.
> It was built to run the experiments and produce the paper results, not as a
> polished library. The best-performing reward model and the fine-tuned EG3D
> generator are **not** included, and the (large) training data lives outside
> the repo. What you *can* do quickly is read the method and run the smoke
> pipelines below on synthetic/sample data.
>
> Public-scope summary:
> [public_release_scope_2026-06-24.md](docs/public_release_scope_2026-06-24.md)

---

## What's the idea (in three steps)

1. **Generate & rank.** Sample meshes from EG3D and collect human preference
   rankings of their geometry (best→worst within small groups).
2. **Learn a reward model.** Train a Bradley-Terry / scalar reward model on a 3D
   representation of each mesh. Several representations + backbones were explored
   (sigma/density field → 3D-conv/UNet3D, point cloud → PointNet/PointNet++/
   CurveNet, depth map → ResNet, landmarks). **The winner: the sigma-density
   field at res 256 through a 3D-conv backbone.**
3. **RLHF fine-tune.** Optimise the EG3D generator against the frozen reward
   model (plus regularisers) so its extracted geometry scores higher.

The reward model consumes a cropped "slab" of the EG3D **sigma density volume**
(see `reward_model_training/static_configs/pads_vals_*.yaml`) and outputs a
scalar reward / preference logit.

---

## Repo map (where to look)

| Path | What |
|------|------|
| `eg3d/` | The EG3D fork: generator, training, and the **RLHF fine-tuning loop**. Entry point `eg3d/train_rlhf.py`; the RLHF loss is in `eg3d/training/loss.py`; the training loop in `eg3d/training/training_loop.py`. |
| `reward_model_training/reward_model_framework/core_modules/` | The **reward-model framework** (Hydra + PyTorch Lightning). Entry point `train_rwd_model.py`. |
| `core_modules/data/` | Data pipeline. The core class is `dset_loaders.dset_single_stream_ordered_minimal`; representations are catalogued in `configs/data/data_defaults.yaml`. |
| `core_modules/models/` | Reward-model backbones (`modules_conv3d`, `modules_pointnet`, `modules_curvenet`, `modules_depthmap`, …) on a shared `UniversalRWDModel` base. |
| `core_modules/configs/` | Hydra configs: `experiment/`, `model/`, `data/`. |
| `core_modules/data/create_train_data/` | Data synthesis from a trained EG3D checkpoint (`generation_utils.py`, `synthesize_*.py`). |
| `eg3d/reward_tune_analysis/` | Curated project-local analysis, export, and verification scripts for reward tuning. |
| `paper_result_analyses/` | Optional paper-linked post-hoc analyses and figure-generation work. The local `jupyter_notebooks/` symlink remains only as a compatibility alias. |
| `paper_artifacts/` | Small tracked paper-supporting result artifacts kept separate from source code. |
| `docs/` | Handoff notes, run guides, and refactor history. |

---

## Quickstart — run a smoke

Two independent pipelines, each with a fast end-to-end smoke (built so the code
is verifiable without the full data/compute). Environment: `hf_geom_eg3d_py39`
(see Setup).

**Reward-model training** (winner config on a tiny data fraction, GPU):
```sh
cd reward_model_training/reward_model_framework
python -m core_modules.train_rwd_model \
  experiment=sfield_256 \
  logger=csv callbacks=public_local using_wandb=false test=false dloader.num_workers=0 trainer.max_epochs=1 \
  trainer.limit_train_batches=2 trainer.limit_val_batches=2 \
  data.dset_dict.proportion_of_data_to_use=0.02
```
This local-first path writes checkpoints under the Hydra run dir (`logs/...` by
default) and keeps WandB fully optional.

**EG3D RLHF fine-tuning** (reusable smoke preset; reaches `tick 0` and exits):
```sh
cd eg3d
python train_rlhf.py experiment=finetune_eg3d_sfield \
  +smoke=on click_legacy_args.outdir=/tmp/eg3d_rlhf_smoke
```
With `using_wandb=false`, the RLHF run stays fully local: each run directory
stores `training_options.json`, `hydra_cfg.yaml`, network snapshots, preview
images, and any enabled reward-histogram / mesh export artifacts.

(Startup prints some benign TensorFlow/TensorBoard import warnings; the real run
begins at the `Training options:` line.)

### Maintained experiment surface

The current maintained experiment configs are:

- reward-model training:
  `sfield_256`, `sdmap`, `tdmap`,
  `pcd_cvnet_point_cloud_entire`,
  `pcd_pnet_point_cloud_entire`,
  `pcd_pnet2_point_cloud_entire`
- EG3D RLHF fine-tuning:
  `finetune_eg3d_null`,
  `finetune_eg3d_sfield`,
  `finetune_eg3d_sdmap`,
  `finetune_eg3d_tdmap`,
  `finetune_eg3d_pn1`

### Paper-facing entrypoints

- Reward-model retraining sweep:
  [train_all_reward_models.sh](reward_model_training/reward_model_framework/core_modules/scripts/train_all_reward_models.sh)
- Protected finetune verification across the five maintained RLHF configs:
  [run_protected_finetune_one_tick.sh](eg3d/reward_tune_analysis/scripts/run_protected_finetune_one_tick.sh)
- Reported sigma-field tuning launcher:
  [sfield_reported_run.sh](eg3d/reward_tune_analysis/scripts/sfield_reported_run.sh)
- Before/after tuned-vs-untuned mesh-bank export:
  [export_snapshot_mesh_bank.py](eg3d/reward_tune_analysis/scripts/export_snapshot_mesh_bank.py)

See also:
[paper_results_guide_2026-06-24.md](docs/paper_results_guide_2026-06-24.md)

## Tests

Pytest smokes that exercise the **real** pipeline (no mocks):
```sh
cd reward_model_training/reward_model_framework
python -m pytest core_modules/tests/ -v
```
- `test_data_types_loadable.py` — every live representation loads from the data dir.
- `test_backbone_train_smoke.py` — each backbone (Conv3D/UNet3D, PointNet,
  PointNet++, CurveNet) trains+validates+tests for a couple of batches through
  the real trainer. Known issues are marked `xfail` with their cause.

---

## Setup

EG3D's stack (CUDA extensions, PyTorch3D) makes the environment fiddly. The env
used for the commands above is `hf_geom_eg3d_py39` (Python 3.9). The original
paper environment was `hf_geom_eg3d` (Python 3.8); install steps:

```sh
conda config --add channels conda-forge
conda create -n hf_geom_eg3d python=3.8
conda activate hf_geom_eg3d
conda install -y pytorch=2.0.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# uv for the rest (https://github.com/astral-sh/uv)
python -m pip install uv
python -m uv pip install fvcore iopath
python -m pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt201/pytorch3d-0.7.4-cp38-cp38-linux_x86_64.whl
python -m uv pip install -r requirements.txt
```
A few analysis scripts also expect
`external/dlib/shape_predictor_5_face_landmarks.dat` (dlib 5-point landmarks,
tracked in the repo) and PyGeM (`git clone
https://github.com/mathLab/PyGeM`).

## Data

The training data is large and **external** to the repo. The reward-model data
(per-seed sigma fields / depth maps) defaults to `RWD_DATA_DIR`
(`~/Documents/eg3dredo_data`). Override via the env var or Hydra
`paths.rwd_data_dir`. Other path overrides: `STATIC_CONFIGS_DIR`,
`RWD_MODELS_DIR`, `RUNS_SUMMARY_CSV` (mirrored by `paths.*` in Hydra). To
regenerate data from an EG3D checkpoint, see
`core_modules/data/create_train_data/synthesize_*.py`.

The main pretrained/tuned checkpoints used in the paper are also external to
the repo. The public code supports:

- training reward models from external preference-derived geometry data
- fine-tuning EG3D from an external pretrained generator checkpoint
- exporting paper-style before/after mesh banks from external snapshot `.pkl`
  files

---

## Notes

- Built originally during a from-scratch learning curve; recently refactored for
  reproducibility (reusable smoke presets, a real-trainer test suite, an audited
  config catalog). See `docs/` for the refactor history.
- Upstream EG3D: https://github.com/NVlabs/eg3d (see its license).

<!-- TODO (author to fill): paper/thesis title + link; author name + contact;
     a before/after geometry result figure; license. -->
