This directory stores the old binary-ranking and reward-model helper chain that
used to live under `eg3d/training/` but is no longer imported by the active
fine-tune runtime.

Contents:

- `ranking_utils.py`: image-pair composition and binary-ranking prep helpers.
- `create_binary_dset_torch_10_08_2023.py`: dataset builder for the historical
  binary-ranking workflow.
- `rwd_model_definitions.py`: older local reward-model definitions.
- `rwd_model_imports.py`: historical catch-all import surface for the above.

These files are preserved for reference only. They keep their internal
historical structure and hardcoded paths, but they are not part of the current
Hydra -> `train_rlhf.py` -> `training_loop.py` runtime.
