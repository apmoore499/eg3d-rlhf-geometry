#!/usr/bin/env bash
cd "$(dirname "$0")/../reward_model_training/reward_model_framework/core_modules/data/create_train_data"

# Render the triple RGB views used by the reward-model pipeline.
python synthesize_triple_rgb.py

# Render the triple depth maps used by the depth-map and point-cloud paths.
python synthesize_triple_dmap.py

# Render the cropped 256^3 sigma fields used by the reported reward model.
python synthesize_sigma_field_256_combined.py

# Render the AW98 landmark tensors derived from the RGB views.
python synthesize_landmarks.py --views 0 1 2
