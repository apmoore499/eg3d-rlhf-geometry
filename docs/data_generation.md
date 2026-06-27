# Reward-model data generation

This stage synthesises the geometry representations that the reward models are
trained on, by sampling the pretrained EG3D generator over a fixed set of latent
seeds. It produces four representations per seed: triple RGB views, triple depth
maps, a cropped `256³` sigma-density field, and AW98 facial landmarks.

You only need this stage if you want to **rebuild the reward-model training
inputs from scratch**. The ranked-preference metadata
(`rankedseedsall.csv`) is already in the repo, so the seeds and their
winner/loser labels are fixed; this stage just renders the geometry for them.

## Prerequisites

- The CUDA environment from the [README](../README.md#setup)
  (`hf_geom_eg3d_py39`).
- A pretrained EG3D FFHQ checkpoint, `ffhq512-128.pkl` (external; see the
  [README](../README.md#what-is-released)).
- Substantial disk and time — the sigma fields in particular are large.

## Configuration (environment variables)

| Variable | Purpose | Default |
| --- | --- | --- |
| `E3D_RLHF_GENERATOR_PKL` | path to the EG3D `ffhq512-128.pkl` checkpoint | `pkl_pt/eg3d_1/ffhq512-128.pkl` |
| `E3D_RLHF_SAVE_DIR` | base output directory for generated data | `generated_data/` |
| `E3D_RLHF_SIGMA_DATA_DIR` | override for the sigma-field output directory | `<save dir>/entire_sigma_field_256_...` |

The seed list and preference labels are read from
`reward_model_training/reward_model_framework/core_modules/data/create_train_data/rankedseedsall.csv`
(4,346 winner/loser pairs). High-quality anchor samples `x_HQ` are drawn from
the `100000–101000` seed range at truncation `ψ = 0.25`.

## Run it

```sh
bash scripts/generate_reward_training_data.sh
```

The launcher runs four synthesis steps in sequence. Each can be toggled with an
environment variable (`1` = run, `0` = skip), which is useful for regenerating a
single representation:

```sh
cd reward_model_training/reward_model_framework/core_modules/data/create_train_data
python synthesize_sigma_field_256_combined.py
```

| Step | Script | Output |
| --- | --- | --- |
| Triple RGB | `synthesize_triple_rgb.py` | three RGB views per seed |
| Triple depth | `synthesize_triple_dmap.py` | three depth maps per seed |
| Sigma field `256³` | `synthesize_sigma_field_256_combined.py` | cropped sigma slab per seed (the reported representation) |
| Landmarks | `synthesize_landmarks.py --views 0 1 2` | AW98 facial landmarks per view |

Outputs land under `E3D_RLHF_SAVE_DIR`, one file per seed per representation.
Existing outputs are reused where the underlying scripts support skipping, so
the launcher can be re-run to resume an interrupted generation.

## Next

Once the data exists, train a reward model on it — see
[reward_models.md](reward_models.md).
