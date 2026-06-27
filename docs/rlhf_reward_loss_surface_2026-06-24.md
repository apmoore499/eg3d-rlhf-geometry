# RLHF Reward-Loss Surface - 2026-06-24

This note records the **current active RLHF reward-loss surface** after the
2026-06-24 cleanup. Older archived configs, wandb dumps, and handover docs still
mention removed branches; treat this file as the authoritative post-cleanup state.

## Active `tune_type` values

- `clamped`
  - fixed clamp bounds from `rwd_clamp_min` / `rwd_clamp_max`
- `clamped_iqr`
  - same clamped loss shape, but bounds come from the initial reward distribution
    after IQR filtering
  - kept as a supported experimental variant, but not part of the reported run
    lineage
- `neg_softplus`
  - smooth reward-maximization surrogate using `neg_softplus_scale < 0`
- `pairs_refset`
  - maintained pairwise reward path using sampled reference-set embeddings

## Active pairwise backend

- `pairs_refset` is the maintained pairwise backend name.
- Archived configs that still use `pairs` are accepted as a **legacy alias** and
  resolve to `pairs_refset`.

## Removed legacy branches

These names are no longer part of the active runtime:

- `PPO`
  - removed because the implementation was not a proper PPO objective
- `median`
  - removed as an abandoned experimental objective
- `pairs_old`
  - removed as an obsolete pairwise branch tied to `old_G_ema`
- `pairs_patches`
  - removed as an archived architecture-specific pairwise branch

If an archived config or old wandb run still mentions one of these names, that is
historical evidence only, not a supported current runtime path.

## Notes

- The five protected finetune configs all still resolve to `tune_type=clamped`.
- `neg_softplus` remains supported because it is implemented coherently and was
  used experimentally.
- `clamped` remains the reported / validated path for the paper-facing sigma-field
  run lineage.
