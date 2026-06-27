"""Introspect the PanoHead checkpoint to read G/D init_kwargs and
rendering_kwargs. Run from the eg3d/ tree root in the conda env:

    python training/panohead_nets/introspect_pkl.py

It only PRINTS metadata; it does not build the vendored networks (so it
works even before the synthesis shim is finalised, by loading the pickle
with the ORIGINAL PanoHead package on sys.path is NOT required -- legacy
load just needs the persistence-stored init_args, which it can read
without re-instantiating if we trap construction). To be safe we add the
PanoHead repo to sys.path so the original classes resolve.
"""
import os
import sys

import autoroot  # noqa: F401  sets eg3d tree root on sys.path

# Allow original PanoHead network classes to resolve during unpickling.
PANOHEAD = "/home/krillman/Documents/eg3dredo/PanoHead"
if PANOHEAD not in sys.path:
    sys.path.insert(0, PANOHEAD)

import dnnlib  # noqa: E402
import legacy  # noqa: E402

PKL = "/home/krillman/Documents/eg3dredo/PanoHead/models/easy-khair-180-gpc0.8-trans10-025000.pkl"


def main():
    with dnnlib.util.open_url(PKL) as f:
        data = legacy.load_network_pkl(f)
    for key in ("G", "G_ema", "D"):
        if key not in data or data[key] is None:
            print(f"== {key}: MISSING ==")
            continue
        net = data[key]
        print(f"== {key}: {type(net).__module__}.{type(net).__name__} ==")
        ik = getattr(net, "init_kwargs", {})
        for k in sorted(ik.keys()):
            if k == "rendering_kwargs":
                continue
            print(f"   {k} = {ik[k]!r}")
        if "rendering_kwargs" in ik:
            print("   rendering_kwargs:")
            rk = ik["rendering_kwargs"]
            for rkk in sorted(rk.keys()):
                print(f"      {rkk} = {rk[rkk]!r}")
        print(f"   #params = {sum(p.numel() for p in net.parameters())}")
        print()


if __name__ == "__main__":
    main()
