# Training reward models

A reward model maps a geometry representation of an EG3D sample to a scalar
quality score, trained from the ranked preference pairs. Training is driven by
[Hydra](https://hydra.cc/) + PyTorch Lightning: you select a representation/
backbone by picking an `experiment` config.

## Quick start

From `reward_model_training/reward_model_framework/`:

```sh
python -m core_modules.train_rwd_model experiment=sfield_256
```

`sfield_256` is the **reported** model: a cropped `256³` sigma-density field
scored by a squeeze-and-excitation residual 3D U-Net. It is the representation
that worked both as a standalone ranker and as a stable fine-tuning signal.

## Experiment configs

All configs live in
`core_modules/configs/experiment/`. Each selects a representation and backbone:

| `experiment=` | Representation | Backbone |
|---|---|---|
| `sfield_256` | cropped `256³` sigma field | ResNet-SE 3D U-Net **(reported)** |
| `sdmap` | single depth map | ResNet-50 |
| `tdmap` | triple depth map | ResNet-50 |
| `pcd_pnet_point_cloud_entire` | point cloud | PointNet |
| `pcd_pnet2_point_cloud_entire` | point cloud | PointNet++ |
| `pcd_cvnet_point_cloud_entire` | point cloud | CurveNet |

The depth-map and point-cloud configs are **intentionally retained**: they are
the alternative representations evaluated in the paper. They are accurate
standalone rankers but unstable when used to drive fine-tuning, which is the
core finding that motivated the sigma-field choice. Keeping them runnable lets a
reader reproduce that comparison.

## Local / smoke runs

For a quiet, CPU-friendly end-to-end pass without Weights & Biases:

```sh
python -m core_modules.train_rwd_model \
  experiment=sfield_256 \
  logger=csv callbacks=public_local using_wandb=false test=false \
  dloader.num_workers=0 trainer.max_epochs=1 \
  trainer.limit_train_batches=2 trainer.limit_val_batches=2 \
  data.dset_dict.proportion_of_data_to_use=0.02
```

## Useful overrides

- `ckpt_path=<path>` — resume from a checkpoint.
- `dloader.num_workers=<n>` / `batch_size_all=<n>` — throughput knobs.
- `test=true/false` — toggle the held-out test pass.
- `using_wandb=false logger=csv` — local logging, no W&B account needed.

## Train all reported models

To reproduce the full representation/backbone sweep in one go:

```sh
bash core_modules/scripts/train_all_reward_models.sh
```

This writes the resulting reward-model ids to `reward_model_ids.env`.

## Outputs

Each run creates a Hydra/Lightning run directory (under `logs/` by default; see
`core_modules/configs/paths/default.yaml`) containing checkpoints and metrics.
The trained checkpoint is what the fine-tuning stage consumes — see
[finetuning.md](finetuning.md).

## Data

Training reads the geometry representations produced in
[data_generation.md](data_generation.md). The ranked preference metadata
(`rankedseedsall.csv`) is already in the repo.
