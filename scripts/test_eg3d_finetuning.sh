#!/usr/bin/env bash
cd "$(dirname "$0")/.."

OUTDIR="${EG3D_TRAINING_OUTDIR:-$HOME/training_runs_2}"

# Test the no-reward control config for one tick.
python eg3d/train_rlhf.py experiment=finetune_eg3d_null train_tick_stop=1 network_snapshot_ticks=9 using_wandb=false EVALUATE_METRICS=false eval_metrics_only=false export_first_images=false plot_rwd_dist=true +render_final_vis=false click_legacy_args.outdir="$OUTDIR"

# Test the sigma-field reward config for one tick.
python eg3d/train_rlhf.py experiment=finetune_eg3d_sfield rwd_model_id=7wnzkgie train_tick_stop=1 network_snapshot_ticks=9 using_wandb=false EVALUATE_METRICS=false eval_metrics_only=false export_first_images=false plot_rwd_dist=true +render_final_vis=false click_legacy_args.outdir="$OUTDIR"

# Test the single-depth-map reward config for one tick.
python eg3d/train_rlhf.py experiment=finetune_eg3d_sdmap rwd_model_id=se9l433b train_tick_stop=1 network_snapshot_ticks=9 using_wandb=false EVALUATE_METRICS=false eval_metrics_only=false export_first_images=false plot_rwd_dist=true +render_final_vis=false click_legacy_args.outdir="$OUTDIR"

# Test the triple-depth-map reward config for one tick.
python eg3d/train_rlhf.py experiment=finetune_eg3d_tdmap rwd_model_id=zxmuq2m9 train_tick_stop=1 network_snapshot_ticks=9 using_wandb=false EVALUATE_METRICS=false eval_metrics_only=false export_first_images=false plot_rwd_dist=true +render_final_vis=false click_legacy_args.outdir="$OUTDIR"

# Test the point-cloud reward config for one tick.
python eg3d/train_rlhf.py experiment=finetune_eg3d_pn1 rwd_model_id=541b13pt train_tick_stop=1 network_snapshot_ticks=9 using_wandb=false EVALUATE_METRICS=false eval_metrics_only=false export_first_images=false plot_rwd_dist=true +render_final_vis=false click_legacy_args.outdir="$OUTDIR"
