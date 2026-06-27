import autoroot  # noqa: F401

from training.render_final_snapshot_vis import render_final_snapshot_vis


def render_img_for_vis(pkl_fn):
    """Compatibility shim for the old module/function name."""
    return render_final_snapshot_vis(pkl_fn)
