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

## Quick test

For the public release test sweep, run:

```sh
bash scripts/test_reward_models.sh
```

## Useful overrides

- `ckpt_path=<path>` — resume from a checkpoint.
- `dloader.num_workers=<n>` / `batch_size_all=<n>` — throughput knobs.
- `test=true/false` — toggle the held-out test pass.
- `using_wandb=false logger=csv` — local logging, no W&B account needed.

## Train all reported models

To reproduce the full representation/backbone sweep in one go:

```sh
bash scripts/train_reward_models.sh
```

The simpler repo-root script is the release-facing path. The framework-local
helper under `core_modules/scripts/` remains available if you want the older
auto-generated `reward_model_ids.env` behavior.

## Outputs

Each run creates a Hydra/Lightning run directory (under `logs/` by default; see
`core_modules/configs/paths/default.yaml`) containing checkpoints and metrics.
The trained checkpoint is what the fine-tuning stage consumes — see
[finetuning.md](finetuning.md).

## Data

Training reads the geometry representations produced in
[data_generation.md](data_generation.md). The ranked preference metadata
(`rankedseedsall.csv`) is already in the repo.
