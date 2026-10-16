"""Command line interface.

    rocketcfd check                      verify OpenFOAM is reachable
    rocketcfd inspect model.stl          normalise geometry, print sizes, save a preview
    rocketcfd init model.stl             write a starter YAML config next to the model
    rocketcfd run config.yaml            run the full pipeline (or pass a model file directly)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Config, config_for_model, load_config

CONFIG_TEMPLATE = """\
# rocketcfd configuration. Every setting except 'model' is optional.
model: {model}
# name: my_rocket            # output folder name (default: model file name)
regime: incompressible       # subsonic, Mach < ~0.3. ('compressible' is planned)

geometry:
  units: {units}                # m | cm | mm | in | auto (ignored for STEP: always metres)
  nose_direction: auto       # auto | +x | -x | +y | -y | +z | -z  (direction the nose points in the file)
  roll_deg: 0                # roll the rocket about its axis, e.g. 45 for an 'x' fin layout
  # ref_diameter: 0.0416     # metres; auto-detected from the body tube
  # cg_from_nose: 0.38       # metres; enables the stability margin in the report

air:
  rho: 1.225
  nu: 1.5e-5

mesh:
  preset: medium             # coarse (quick look) | medium | fine (mesh independence check)

solver:
  iterations: 1500
  average_last: 300
  nprocs: {nprocs}

sweeps:
  speed:                     # Cd vs speed at a fixed angle of attack
    aoa: 0
    speeds: [10, 20, 35, 50, 75, 100]
  aoa:                       # stability sweep at a fixed speed
    speed: 50
    angles: [0, 2, 4, 6, 8, 10]

visualize: true

openfoam:
  runner: auto               # auto | wsl | native
  wsl_distro: Ubuntu
  bashrc: /usr/lib/openfoam/openfoam2412/etc/bashrc
  work_dir: ~/rocketcfd_runs # Linux-side folder for the OpenFOAM cases

output_dir: {output_dir}
"""


def _load(target: str, args) -> Config:
    p = Path(target)
    cfg = load_config(p) if p.suffix.lower() in (".yaml", ".yml") else config_for_model(p)
    if getattr(args, "preset", None):
        cfg.mesh.preset = args.preset
    if getattr(args, "nprocs", None):
        cfg.solver.nprocs = args.nprocs
    if getattr(args, "iterations", None):
        cfg.solver.iterations = args.iterations
    return cfg


def cmd_check(args) -> int:
    from .config import OpenFOAMConfig
    from .runner import Runner

    r = Runner(OpenFOAMConfig())
    print(f"OpenFOAM {r.check()} found via {r.mode} runner; case folder {r.work_dir} -> {r.local(r.work_dir)}")
    return 0


def cmd_inspect(args) -> int:
    from .config import GeometryConfig
    from .geometry import plot_geometry, prepare_geometry
    from .mesh import plan_mesh

    model = Path(args.model)
    geo = prepare_geometry(model, GeometryConfig(units=args.units, nose_direction=args.nose))
    print(json.dumps(geo.summary(), indent=2))
    cfg = config_for_model(model)
    plan = plan_mesh(geo, cfg)
    print(f"medium mesh: surface cells {plan.surface_size * 1000:.2f} mm, "
          f"{plan.summary()['background_cells']:,} background cells")
    preview = model.with_name(model.stem + "_preview.png")
    plot_geometry(geo, preview)
    print(f"Preview: {preview}")
    return 0


def cmd_init(args) -> int:
    import os

    model = Path(args.model).resolve()
    out = Path(args.output) if args.output else model.with_suffix(".yaml")
    if out.exists() and not args.overwrite:
        print(f"{out} already exists (use --overwrite)", file=sys.stderr)
        return 1
    rel = os.path.relpath(model, out.parent.resolve()).replace("\\", "/")
    out_dir = os.path.relpath(Path("runs").resolve(), out.parent.resolve()).replace("\\", "/")
    nprocs = max(1, min(8, (os.cpu_count() or 2) // 2))
    out.write_text(CONFIG_TEMPLATE.format(model=rel, units="m" if model.suffix.lower() in (".step", ".stp")
                                          else "auto", nprocs=nprocs, output_dir=out_dir), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


def cmd_run(args) -> int:
    from .pipeline import run_pipeline

    cfg = _load(args.target, args)
    out = run_pipeline(cfg, mesh_only=args.mesh_only, force=args.force, verbose=args.verbose)
    print(f"\nResults in {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rocketcfd", description="Model rocket CFD pipeline (OpenFOAM)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="verify OpenFOAM is reachable").set_defaults(func=cmd_check)

    p = sub.add_parser("inspect", help="normalise a model and report its dimensions")
    p.add_argument("model")
    p.add_argument("--units", default="auto")
    p.add_argument("--nose", default="auto", help="direction the nose points: auto, +x, -z, ...")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("init", help="write a starter config for a model")
    p.add_argument("model")
    p.add_argument("-o", "--output")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("run", help="run the pipeline from a YAML config or a model file")
    p.add_argument("target", help="config .yaml, or an .stl/.step model to run with defaults")
    p.add_argument("--mesh-only", action="store_true", help="stop after meshing")
    p.add_argument("--force", action="store_true", help="redo mesh and cases even if up to date")
    p.add_argument("--preset", choices=["coarse", "medium", "fine"])
    p.add_argument("--nprocs", type=int)
    p.add_argument("--iterations", type=int)
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_run)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("matplotlib", "trimesh", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume; finished cases are kept.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        if getattr(args, "verbose", False):
            raise
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
