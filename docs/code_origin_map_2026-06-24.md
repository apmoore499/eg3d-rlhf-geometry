# Code Origin Map - 2026-06-24

This note records a practical provenance split for the main `eg3d/` top-level
files after cleanup. It is not a legal attribution document; it is a maintenance
aid so future cleanup does not mix upstream EG3D files with project-local code.

## NVIDIA-origin or NVIDIA-derived EG3D surface

These are upstream EG3D files or close local adaptations of the original EG3D
runtime surface. Treat them as the core model/training code, not as disposable
project scratch code.

- `eg3d/calc_metrics.py`
- `eg3d/camera_utils.py`
- `eg3d/dataset_tool.py`
- `eg3d/dataset_tool_noempty.py`
- `eg3d/environment.yml`
- `eg3d/gen_meshes.py`
- `eg3d/gen_samples.py`
- `eg3d/gen_videos.py`
- `eg3d/legacy.py`
- `eg3d/shape_utils.py`
- `eg3d/train.py`
- `eg3d/visualizer.py`

## NVIDIA-derived but heavily project-adapted RLHF runtime

These files sit on the active EG3D fine-tuning path and have substantial local
reward-tuning modifications.

- `eg3d/train_rlhf.py`
- `eg3d/training/training_loop.py`
- `eg3d/training/loss.py`
- `eg3d/training/render_final_snapshot_vis.py`
- `eg3d/training/mesh_preview_utils.py`
- `eg3d/visualise_sdf_chimerax.py`

## Project-local reward-tuning / analysis code

These are local scripts, notes, and archived tools created for this project.

- `eg3d/reward_tune_analysis/`
- `docs/rlhf_reward_loss_surface_2026-06-24.md`

## Historical note

`eg3d/reward_tune_analysis/legacy/` contains archived local tooling that is kept
for reference only and is not part of the maintained active runtime surface.
