# Camera conventions

The camera handling follows EG3D unchanged; this note records the conventions
the data-generation and analysis code assumes.

- **Pose format:** camera-to-world (`cam2world`) extrinsics in OpenCV convention
  (`+x` right, `+y` down, `+z` forward / into the scene).
- **Intrinsics:** normalised (focal length and principal point expressed as
  fractions of the image dimensions), so they are resolution-independent.
- **Canonical view:** the frontal pose used for the sigma-field crop, depth
  maps, and landmark back-projection. Sigma fields are sampled in this canonical
  scene space.
- **Field of view:** the shared EG3D default (~18.84°), identical across the
  EG3D-family generators compared in the paper; per-generator differences live
  in the rendering parameters (camera radius, pivot, box warp), not the camera
  convention itself.

The static camera/conditioning tensors used by the synthesis scripts are loaded
from the `STATIC_CONFIGS_DIR` (e.g. `single_dmap_cameras.pt`,
`triple_img_cameras.pt`).
