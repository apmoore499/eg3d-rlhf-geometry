#!/usr/bin/env bash
cd "$(dirname "$0")/.."

OUTDIR="${EG3D_TRAINING_OUTDIR:-$HOME/training_runs_2}"

# Run the reported 20-tick sigma-field fine-tune from the released reward-model id.
python eg3d/train_rlhf.py experiment=finetune_eg3d_sfield rwd_model_id=7wnzkgie train_tick_stop=20 network_snapshot_ticks=20 using_wandb=false EVALUATE_METRICS=false export_first_images=true plot_rwd_dist=true +render_final_vis=false seedslist_visualisation=[2,3,4,5,6] click_legacy_args.outdir="$OUTDIR"
