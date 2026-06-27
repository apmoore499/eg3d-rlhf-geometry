#!/usr/bin/env bash
# Sigma-field reward harness check (smoke) against the new reward model wkc6e2f7.
# Light visualisation (vis_shape_res=64, 2 seeds) so the mesh synth is fast.
# Writes a run dir under training_runs_2.
set -euo pipefail

cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/eg3d

conda run --no-capture-output -n hf_geom_eg3d_py39 \
  python train_rlhf.py \
    experiment=finetune_eg3d_sfield \
    +smoke=on \
    rwd_model_id=wkc6e2f7 \
    train_tick_stop=3 \
    network_snapshot_ticks=1 \
    vis_shape_res=64 \
    'seedslist_visualisation=[2,3]' \
    click_legacy_args.outdir=/media/krillman/240GB_DATA/training_runs_2
