This directory contains fixed runtime assets shared across reward-model
training, EG3D fine-tuning, and some data-generation utilities.

Kept files:

- `pads_vals_entire.yaml`
  Maintained sigma-volume crop used by the `sigma_field_256` dtype, the
  `sfield_256` reward model, and the active EG3D RLHF reward path.
- `pads_vals_nose.yaml`
  Nose-region crop used by the `nose_512` dtype only. This is an old
  experimental path, not part of the maintained public reward-model surface.
- `pads_vals_64c.yaml`
  Older centered sigma crop kept only for legacy compatibility.
- `single_dmap_cameras.pt`, `single_dmap_conditioning.pt`
  Cached single-view depth-map camera / conditioning tensors.
- `triple_dmap_cameras.pt`, `triple_dmap_conditioning.pt`
  Cached triple-view depth-map camera / conditioning tensors.
- `single_img_cameras.pt`, `single_img_conditioning.pt`
  Cached single-view RGB render camera / conditioning tensors.
- `triple_img_cameras.pt`, `triple_img_conditioning.pt`
  Cached triple-view RGB render camera / conditioning tensors.
- `rankings_template.csv`
  Legacy ranking-app template. Kept as an example schema only; not part of the
  maintained runtime path.
- `logging.ini`
  Legacy logging defaults used by some reward-model utilities.

Notes:

- These files are runtime inputs, not generated outputs.
- If a future public mirror moves bundled assets into a dedicated `weights/` or
  `assets/` tree, this directory is one of the main candidates to relocate as a
  unit.
