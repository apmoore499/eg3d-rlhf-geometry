This directory stores legacy project-local EG3D reward-tuning scripts that
are not part of the current live fine-tune runtime.

Current contents:

- `aux_data_tools.py`: obsolete EG3D-side depth-map/point-cloud helper module.
  The live reward-model and fine-tune paths now use
  `core_modules.utils.depth_to_pcd` plus `core_modules.utils.camera_utils`
  instead, and no active runtime imports this file.
- `batch_rlhf_15.slurm`: older Spartan SLURM launcher tied to a historical
  project environment and path layout.
- `mesh_export_preview_legacy.py`: old mixed mesh-export, ChimeraX preview, and
  image-stacking helper file that was previously named
  `mesh_synthesis_boiler_07_2023.py`.
- `ranking_reward_tools/`: archived binary-ranking and old reward-model helper
  chain formerly kept under `eg3d/training/`.
- `run_sigma_harness.sh`: local hardcoded sigma-field smoke launcher kept only
  as a historical reference script, not as a maintained entrypoint.
- `train_rlhf_get_meshes.py`: stale standalone RLHF entrypoint that points to a
  missing Hydra config.
- `synth_views.py`: one-off mesh preview script with hardcoded local paths and
  duplicated render logic.

These files are kept for reference, not as active entrypoints.
