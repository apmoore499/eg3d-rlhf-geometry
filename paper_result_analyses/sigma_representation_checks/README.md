This folder holds optional sigma-volume inspection helpers.

They are useful for checking what the `sfield_256` reward model actually sees,
but they are not part of the minimal paper reproduction path.

Contents:

- `crop_pads_slab_to_mrc.py`
  Apply the reward-model slab crop to full sigma cubes and export MRC files.
- `pads_vals_box_viz.py`
  Visualise the retained slab region relative to the full sigma volume.
- `export_sigma_mrc_untuned_vs_tuned_2068.py`
  Export untuned vs tuned EG3D sigma cubes as MRC files for direct inspection.
