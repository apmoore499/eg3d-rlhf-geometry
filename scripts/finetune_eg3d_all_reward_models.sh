#!/usr/bin/env bash
cd "$(dirname "$0")/.."

OUTDIR="${EG3D_TRAINING_OUTDIR:-$HOME/training_runs_2}"

# Fine-tune EG3D against the reported sigma-field reward model.
python eg3d/train_rlhf.py experiment=finetune_eg3d_sfield rwd_model_id=7wnzkgie click_legacy_args.outdir="$OUTDIR"

# Fine-tune EG3D against the single-depth-map reward model.
python eg3d/train_rlhf.py experiment=finetune_eg3d_sdmap rwd_model_id=se9l433b click_legacy_args.outdir="$OUTDIR"

# Fine-tune EG3D against the triple-depth-map reward model.
python eg3d/train_rlhf.py experiment=finetune_eg3d_tdmap rwd_model_id=zxmuq2m9 click_legacy_args.outdir="$OUTDIR"

# Fine-tune EG3D against the PointNet reward model.
python eg3d/train_rlhf.py experiment=finetune_eg3d_pn1 rwd_model_id=541b13pt click_legacy_args.outdir="$OUTDIR"

# Fine-tune EG3D against the PointNet++ reward model.
python eg3d/train_rlhf.py experiment=finetune_eg3d_pn1 rwd_model_id=7pvkwpnz click_legacy_args.outdir="$OUTDIR"

# Fine-tune EG3D against the CurveNet reward model.
python eg3d/train_rlhf.py experiment=finetune_eg3d_pn1 rwd_model_id=w4eberou rlhf_tune_hpms.batch_size=2 click_legacy_args.outdir="$OUTDIR"
