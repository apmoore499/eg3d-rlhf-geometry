#!/bin/bash

# Initialize conda for this script
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hf_geom_eg3d_py39

# python synthesize_sigma_field_256_combined.py
# python synthesize_single_dmap.py <-not needed anymore can just use triple_dmap aand take the middle tensor as it will have canonical view ie same as canon dmap
python synthesize_triple_rgb.py
python synthesize_triple_dmap.py
# python save_mesh_visualisations.py