"""End-to-end orchestration: geometry -> mesh -> sweep of solver cases -> report.

Work is resumable: the mesh and each finished case leave a fingerprint file, and are
skipped on the next run if nothing that affects them has changed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import sys
import time
from pathlib import Path, PurePosixPath

from . import post, render3d
from .config import Config
from .geometry import RocketGeometry, plot_geometry, prepare_geometry
from .mesh import mesh_commands, parse_check_mesh, plan_mesh, write_mesh_case
from .regimes import get_regime
from .runner import Runner, write_text

log = logging.getLogger(__name__)
DONE_FILE = ".rocketcfd_done"


def case_name(speed: float, aoa: float) -> str:
    return f"V{speed:06.2f}_A{aoa:+06.2f}".replace(".", "p")


def _fingerprint(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def _is_done(runner: Runner, case: PurePosixPath, fp: str) -> bool:
    marker = runner.local(case / DONE_FILE)
    return marker.exists() and marker.read_text().strip() == fp


def _fresh_dir(runner: Runner, case: PurePosixPath) -> None:
    runner.shell(f"rm -rf '{case}' && mkdir -p '{case}'")


def _setup_logging(out: Path, verbose: bool) -> logging.Handler:
    fh = logging.FileHandler(out / "pipeline.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger("rocketcfd").addHandler(fh)
    logging.getLogger("rocketcfd").setLevel(logging.DEBUG if verbose else logging.INFO)
    return fh


@contextlib.contextmanager
def _keep_awake():
    """Stop Windows from sleeping while a run is in progress (a sleeping PC also pauses WSL).

    The request belongs to this process and ends when the run finishes or the process exits;
    it does not change any power settings. Display sleep and closing the lid still apply.
    """
    if sys.platform != "win32":
        yield
        return
    import ctypes
    es_continuous, es_system_required = 0x80000000, 0x00000001
    kernel32 = ctypes.windll.kernel32
    if kernel32.SetThreadExecutionState(es_continuous | es_system_required):
        log.info("Keeping the PC awake until the run finishes")
    else:
        log.warning("Could not stop the PC from sleeping; a sleep will pause the run")
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(es_continuous)


def run_pipeline(cfg: Config, mesh_only: bool = False, force: bool = False, verbose: bool = False) -> Path:
    out = cfg.output_dir / cfg.name
    (out / "flow").mkdir(parents=True, exist_ok=True)
    handler = _setup_logging(out, verbose)
    try:
        with _keep_awake():
            return _run(cfg, out, mesh_only, force)
    finally:
        logging.getLogger("rocketcfd").removeHandler(handler)
        handler.close()


def _run(cfg: Config, out: Path, mesh_only: bool, force: bool) -> Path:
    t_start = time.time()
    regime = get_regime(cfg.regime)
    runner = Runner(cfg.openfoam)
    log.info("OpenFOAM %s via %s runner", runner.check(), runner.mode)

    # Geometry
    log.info("Preparing geometry from %s", cfg.model)
    geo = prepare_geometry(cfg.model, cfg.geometry)
    warnings = list(geo.warnings) + regime.check_conditions(cfg, cfg.air.speed_of_sound)
    plot_geometry(geo, out / "geometry.png")
    geo.mesh.export(out / "rocket_normalized.stl")
    s = geo.summary()
    log.info("Rocket: L = %.3f m, d_ref = %.1f mm, fineness %.1f", geo.length, geo.ref_diameter * 1000,
             s["fineness_ratio"])

    # Mesh
    plan = plan_mesh(geo, cfg)
    root = runner.work_dir / cfg.name
    mesh_case = root / "mesh"
    mesh_fp = _fingerprint((out / "rocket_normalized.stl").read_bytes(), plan.summary(), regime.name)
    if not force and _is_done(runner, mesh_case, mesh_fp):
        log.info("Reusing existing mesh in %s", mesh_case)
    else:
        log.info("Meshing (surface cells %.2f mm, ~%d background cells) ...", plan.surface_size * 1000,
                 plan.summary()["background_cells"])
        t = time.time()
        _fresh_dir(runner, mesh_case)
        write_mesh_case(runner.local(mesh_case), geo, plan, cfg, regime)
        runner.run(mesh_case, mesh_commands(cfg.solver.nprocs))
        write_text(runner.local(mesh_case / DONE_FILE), mesh_fp)
        log.info("Mesh finished in %.0f s", time.time() - t)
    mesh_stats = parse_check_mesh(runner.local(mesh_case / "log.checkMesh").read_text(errors="replace"))
    log.info("Mesh: %s cells, checkMesh %s", f"{mesh_stats.get('cells', 0):,}",
             "OK" if mesh_stats["ok"] else "reported problems")
    if not mesh_stats["ok"]:
        warnings.append(f"checkMesh reported {mesh_stats['failed_checks']} failed check(s); see "
                        f"{mesh_case}/log.checkMesh")
    (out / "mesh.json").write_text(json.dumps({"plan": plan.summary(), "checkMesh": mesh_stats}, indent=2))

    results: list[dict] = []
    if not mesh_only:
        results = _run_cases(cfg, geo, regime, runner, root, mesh_case, mesh_fp, out, force)

    # Sweep plots and report
    images: dict[str, str] = {}
    if cfg.sweeps.speed and post.plot_speed_sweep(results, cfg.sweeps.speed.aoa, cfg, out / "drag_vs_speed.png"):
        images["Drag vs speed"] = "drag_vs_speed.png"
    if cfg.sweeps.aoa and post.plot_aoa_sweep(results, cfg.sweeps.aoa.speed, geo, cfg, out / "aoa_sweep.png"):
        images["Angle-of-attack sweep"] = "aoa_sweep.png"
    hero = _hero_case(results, cfg)
    if hero and (out / "render" / f"{hero['case']}.png").exists():
        images[f"3D view: {hero['speed_mps']:g} m/s, α = {hero['aoa_deg']:g}°"] = f"render/{hero['case']}.png"
    for r in results:
        img = out / "flow" / f"{r['case']}.png"
        if img.exists():
            images[f"Flow: {r['speed_mps']:g} m/s, α = {r['aoa_deg']:g}°"] = f"flow/{img.name}"
    cp = post.cp_from_sweep(results, cfg.sweeps.aoa.speed, geo, cfg) if cfg.sweeps.aoa else None
    if cp:
        log.info("Centre of pressure: %.0f mm from the nose (%.2f cal)%s", cp["xcp_m"] * 1000, cp["xcp_cal"],
                 f", stability {cp['stability_cal']:.2f} cal" if "stability_cal" in cp else "")
        (out / "centre_of_pressure.json").write_text(json.dumps(cp, indent=2))
    post.write_report(out / "report.md", cfg, geo, mesh_stats, plan.summary(), results, warnings, str(root),
                      images)
    (out / "config_used.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))
    log.info("Done in %.1f min. Report: %s", (time.time() - t_start) / 60, out / "report.md")
    return out


def _run_cases(cfg: Config, geo: RocketGeometry, regime, runner: Runner, root: PurePosixPath,
               mesh_case: PurePosixPath, mesh_fp: str, out: Path, force: bool) -> list[dict]:
    conditions = cfg.sweeps.conditions()
    solver_settings = {k: v for k, v in dataclasses.asdict(cfg.solver).items() if k != "nprocs"}
    air = dataclasses.asdict(cfg.air)
    if regime.name == "incompressible":  # keep existing cases valid: p and T do not affect them
        air = {k: air[k] for k in ("rho", "nu", "turbulence_intensity", "viscosity_ratio")}
    results = []
    for i, (speed, aoa) in enumerate(conditions, 1):
        name = case_name(speed, aoa)
        case = root / "cases" / name
        fp = _fingerprint(mesh_fp, speed, aoa, air, solver_settings, cfg.visualize)
        tag = f"[{i}/{len(conditions)}] {name} (V = {speed:g} m/s, AoA = {aoa:g} deg)"
        if not force and _is_done(runner, case, fp):
            log.info("%s: already done, reusing", tag)
        else:
            log.info("%s: solving ...", tag)
            t = time.time()
            _fresh_dir(runner, case)
            regime.write_case(runner.local(case), geo, cfg, speed, aoa)
            copy = ("copyMesh", f"mkdir -p constant && cp -r '{mesh_case}/constant/polyMesh' constant/")
            runner.run(case, [copy] + regime.solve_commands(cfg))
            write_text(runner.local(case / DONE_FILE), fp)
            log.info("%s: solved in %.0f s", tag, time.time() - t)

        local = runner.local(case)
        r = post.case_result(local, geo, cfg, regime, speed, aoa)
        r["case"] = name
        results.append(r)
        xcp = "" if math.isnan(r["Xcp_cal"]) else f", Xcp = {r['Xcp_cal']:.2f} cal"
        log.info("%s: CD = %.4f, CN = %.4f%s%s", tag, r["CD"], r["CN"], xcp,
                 "" if r["converged"] else f" (not converged, CD swing {r['cd_fluctuation']:.1%})")
        if cfg.visualize:
            post.render_flow(local, geo, cfg, regime, speed, aoa, out / "flow" / f"{name}.png")
        if cfg.render_3d:
            _render_3d(local, geo, cfg, regime, speed, aoa, out / "render" / f"{name}.png")
        post.write_csv(results, out / "results.csv")
    return results


_warned_no_pyvista = False


def _render_3d(case: Path, geo: RocketGeometry, cfg: Config, regime, speed: float, aoa: float,
               png: Path) -> bool:
    """3D presentation image of one case. Never fails the run: a bad render is only logged."""
    global _warned_no_pyvista
    if not render3d.available():
        if not _warned_no_pyvista:
            log.info("Skipping 3D renders: PyVista is not installed (pip install -e \".[render]\")")
            _warned_no_pyvista = True
        return False
    try:
        return render3d.render_case(case, geo, cfg, regime, speed, aoa, png)
    except Exception as exc:  # noqa: BLE001
        log.warning("3D render of %s failed: %s", case.name, exc)
        return False


def _hero_case(results: list[dict], cfg: Config) -> dict | None:
    """The case shown as the report's 3D view: the largest angle of the AoA sweep, else the first case."""
    if cfg.sweeps.aoa:
        sweep = [r for r in results if r["speed_mps"] == cfg.sweeps.aoa.speed]
        if sweep:
            return max(sweep, key=lambda r: abs(r["aoa_deg"]))
    return results[0] if results else None


def render_cases(cfg: Config, speed: float | None = None, aoa: float | None = None) -> list[Path]:
    """Re-render the 3D images of already-solved cases, without meshing or solving anything."""
    if not render3d.available():
        raise RuntimeError("3D rendering needs PyVista: pip install -e \".[render]\"")
    out = cfg.output_dir / cfg.name
    regime = get_regime(cfg.regime)
    runner = Runner(cfg.openfoam)
    geo = prepare_geometry(cfg.model, cfg.geometry)
    written = []
    for v, a in cfg.sweeps.conditions():
        if (speed is not None and v != speed) or (aoa is not None and a != aoa):
            continue
        name = case_name(v, a)
        case = runner.local(runner.work_dir / cfg.name / "cases" / name)
        if not (case / DONE_FILE).exists():
            log.info("%s: not solved yet, skipping", name)
            continue
        png = out / "render" / f"{name}.png"
        if _render_3d(case, geo, cfg, regime, v, a, png):
            log.info("%s: wrote %s", name, png)
            written.append(png)
    return written
