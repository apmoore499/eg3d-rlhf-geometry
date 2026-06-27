# Released artifacts

Two checkpoints are distributed as GitHub Release assets; the untuned EG3D
baseline is external (NVIDIA). Each released asset is below GitHub's 2 GiB
per-asset limit. See the [README](../README.md#download-the-released-models) for
the download commands.

## 1. Sigma-field reward model

- **Asset:** `reward-model-7wnzkgie-sfield256.zip` (~49 MB)
- The reported sigma-field reward model used by the EG3D fine-tuning path.
- **Loading contract:** the loader reads a bundle directory containing
  `best_model.pt` + `release_config.yaml`, which keeps the tune-time data
  transforms coupled to the weights. Unzip to
  `reward_model_training/reward_model_framework/core_modules/RWD_MODELS_FOR_TUNING/7wnzkgie/`.
- Optional inspection files may be bundled (`datamodule_third.pt`,
  `model_example_input.pt`); they are not required for loading or inference.

## 2. Fine-tuned EG3D snapshot

- **Asset:** `eg3d-finetuned-sfield-run01446-network-snapshot-002068_LAST.pkl`
  (~361 MB)
- The reported fine-tuned EG3D generator, used for tuned-vs-untuned geometry
  export and consumed directly by the analysis scripts (no special placement).

## 3. Untuned baseline (external)

- `ffhq512-128.pkl` — the pretrained EG3D FFHQ checkpoint, the starting point for
  fine-tuning and the paired baseline for mesh export. Obtain it from the
  upstream [EG3D repository](https://github.com/NVlabs/eg3d) / NVIDIA NGC.

## Scripts that consume these

- **Mesh-bank export:** `eg3d/reward_tune_analysis/scripts/export_snapshot_mesh_bank.py`
  (`--baseline-pkl`, `--tuned-pkl`); use the `legacy_sigma10` path for the
  paper-facing geometry.
- **Release verifier:** `reward_model_training/reward_model_framework/core_modules/scripts/verify_public_release.py`
  — loads the released reward model and runs forward/export test checks.
- **Fine-tuning:** see [finetuning.md](finetuning.md).
