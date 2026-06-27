# Design notes

Choices made during the project that the paper ([arXiv:2606.27305](https://arxiv.org/abs/2606.27305))
does not detail. The paper reports the method and the sigma-field result; the repo holds a
wider set of experiments. The committed config defaults are the settings used for the
reported models.

## Reward-model loss (three heads)

The reward model trains three preference heads together
(`reward_model_training/reward_model_framework/core_modules/configs/loss/all_losses.yaml`,
assembled in `core_modules/models/base.py`):

- `lambda_pairs` (1.0): binary cross-entropy over the ranked pair, computed in both orders for order-invariance (paired head outputs logits).
- `lambda_scalar_rwd` (1e-3): the Ouyang et al. (InstructGPT/2022) scalar-reward objective — the loss
  written as `L_w` in the paper.
- `lambda_BT` (1e-3): a Bradley-Terry term (learns the BT lambda parameter).

All three express the same ranking/transitivity signal. They were run together to impose
order-invariance and to confirm no single formulation dominated. Throughout the course of project, ablations showed no meaningful difference, so the three-head default was kept. A reconstruction loss (`lambda_recon`) applies to the sigma-field model only.

## Representations

The sigma-field (256³ density volume) preserved the most geometric signal and is the reported reward model and fine-tuning input. The others were explored but not pursued:

- Depth-map filters (Laplacian, high-pass, low-pass, side-crop): tried, no improvement. The reported depth-map models use no filtering (`core_modules/configs/data/augmentations/depth_map_maps_transforms_none.yaml`).
- Point-cloud transforms (subsample, jitter, random scale, normalisation): varied without improvement. The point-cloud backbones (PointNet/PointNet++/CurveNet) do not learn the ranking and collapse toward chance.

## Fine-tuning hyperparameters

Only the sigma-field fine-tune uses a single, settled hyperparameter set — the one reported
(`eg3d/training/rlhf_tune_configs/experiment/finetune_eg3d_sfield.yaml` + `base_config.yaml`).
The other fine-tune configs (`finetune_eg3d_{null,pn1,sdmap,tdmap}`) are examples whose
hyperparameters were left varying between configs: tuning the non-sigma representations to
one consistent set gave no benefit, so they reflect abandoned experimentation. The
hard-nose-depth penalty (`lambda_nose_hard: 1.0`) is used only for depth-map fine-tuning
and is off for the reported sigma-field run.

## Where the configs are

- Reward-model experiments: `reward_model_training/reward_model_framework/core_modules/configs/experiment/`
- Reward-model losses / transforms: `.../configs/loss/`, `.../configs/data/augmentations/`
- EG3D fine-tuning: `eg3d/training/rlhf_tune_configs/` (`base_config.yaml` + `experiment/`)
