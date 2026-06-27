# Public Artifact Policy

This document defines how paper-adjacent artifacts should be split between the
working research repo and the future public mirror.

See also:

- [public_release_scope_2026-06-24.md](public_release_scope_2026-06-24.md)
- [paper_results_guide_2026-06-24.md](paper_results_guide_2026-06-24.md)

## Goal

The public mirror should expose:

- the code required to understand and rerun the maintained paper-backed paths
- small summary artifacts that help validate reported analyses

It should not expose:

- large local run archives
- redundant generated outputs
- every historical checkpoint kept during research

## `paper_artifacts/reward_embedding_analysis/`

Policy:

- keep this in the public mirror
- keep it summary-level only

Why:

- the currently tracked contents are small CSV / JSON artifacts
- they support the paper's cross-generator analysis surface
- they do not carry the burden of large render banks or private local outputs

Allowed contents:

- per-seed reward CSVs
- small summary JSON files
- concise README / provenance notes

Disallowed contents for the public mirror:

- large image banks
- bulky intermediate tensors
- copied run directories from `/media/...`

## `RWD_MODELS_FOR_TUNING/`

Policy:

- keep in the private/working repo
- exclude from the default public mirror

Why:

- it is a large mixed archive of reward-model runs
- the directory is not a curated release surface
- publishing it wholesale adds noise and repo weight without improving the
  paper-backed story

### Public release rule

If reward-model weights are released, do not publish the full archive. Publish
only a curated released subset.

Minimum mandatory released checkpoint:

- `7wnzkgie`
  The sigma-field reward model used by the reported EG3D RLHF fine-tuning path.

Optional additional released checkpoints:

- representatives for the other maintained experiment families:
  - `sdmap`
  - `tdmap`
  - `pcd_pnet_point_cloud_entire`
  - `pcd_pnet2_point_cloud_entire`
  - `pcd_cvnet_point_cloud_entire`

These should only be released once there is an explicit curated manifest
mapping experiment names to chosen run ids.

## Reported Fine-Tuned EG3D Snapshots

Policy:

- do not mirror bulky run directories wholesale
- if reported EG3D fine-tuned snapshots are released, publish them as a small
  curated bundle outside normal git contents

Why:

- the tuned EG3D snapshot is part of the paper story, but the surrounding run
  directories are not a clean public artifact surface
- a small released snapshot bundle is useful; wholesale run mirrors are not

Recommended released contents:

- the reported finetuned EG3D snapshot `.pkl`
- a short manifest giving:
  - run id
  - finetune config
  - reward-model checkpoint used
  - expected paired untuned baseline checkpoint

## Recommended Release Shape

For the public mirror:

- keep code/configs in git
- keep small summary artifacts in git
- put released checkpoint files in:
  - GitHub release assets, or
  - a separate lightweight public artifact bundle

Recommended manifest fields:

- experiment name
- released run id
- representation / backbone
- file names included
- short note on paper role
- for finetuned EG3D snapshots, the finetune config and paired untuned
  baseline checkpoint

## Current Judgment

As of 2026-06-24:

- `paper_artifacts/reward_embedding_analysis/` is suitable to keep in the public mirror
- `RWD_MODELS_FOR_TUNING/` is not suitable to mirror wholesale
- `7wnzkgie` is the only unambiguously mandatory reward-model checkpoint for a
  paper-aligned public release
- the reported finetuned EG3D snapshot is a reasonable additional curated
  release artifact, but should be published as a small explicit bundle rather
  than by exposing full run directories
