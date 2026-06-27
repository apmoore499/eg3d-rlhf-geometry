import ast
import shutil
import sys
from pathlib import Path

from chimerax.core.commands import run


def parse_args(argv):
    if len(argv) != 8:
        raise SystemExit(
            "Usage: visualise_sdf_chimerax.py <mrc_fn> <out_png> <pixel_size> "
            "<yinit> <supersample> <ysteps> <meshpath> <level>"
        )

    mrc_fn, out_png, pixel_size, yinit, supersample, ysteps, meshpath, level = argv
    ysteps_num = ast.literal_eval(ysteps)
    if not isinstance(ysteps_num, list):
        raise SystemExit("ysteps must be a Python list literal, e.g. '[-30, -30]'")

    return {
        "mrc_fn": Path(mrc_fn),
        "out_png": Path(out_png),
        "pixel_size": str(pixel_size),
        "yinit": float(yinit),
        "supersample": str(supersample),
        "ysteps": [float(x) for x in ysteps_num],
        "meshpath": Path(meshpath),
        "level": str(level),
    }


def main(argv):
    args = parse_args(argv)
    meshpath = args["meshpath"].resolve()
    mrc_path = meshpath / args["mrc_fn"]
    tmp_dir = meshpath / "tmp"
    out_fn = args["out_png"].with_suffix("").name + "_mesh.png"

    run(session, "close session")
    run(session, "log clear")
    run(session, f'cd "{meshpath.as_posix()}"')
    run(session, f'open "{mrc_path.as_posix()}" format mrc')
    run(session, "volume #1 style surface")
    run(session, f"volume #1 level {args['level']}")
    run(session, "volume #1 step 1")
    run(session, "lighting full")
    run(session, f"zoom pixelsize {args['pixel_size']}")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    total_angle = 0.0
    for ydeg in [args["yinit"]] + args["ysteps"]:
        run(session, f"turn y {ydeg}")
        total_angle += ydeg
        save_fn = tmp_dir / out_fn.replace("_mesh.png", f"_mesh_y{int(total_angle) if total_angle.is_integer() else total_angle}.png")
        run(
            session,
            f'save "{save_fn.as_posix()}" supersample {args["supersample"]} width 1024 height 1024',
        )

    run(session, "exit")


main(sys.argv[1:])
