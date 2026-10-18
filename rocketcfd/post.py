"""Turn OpenFOAM output into aerodynamic coefficients, plots and flow images.

Axes: x is the rocket axis (nose tip at 0, pointing aft), y is the pitch-up direction.
Body-axis coefficients (CA, CN) and wind-axis coefficients (CD, CL) are both reported.
Moments are taken about the nose tip.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import Config  # noqa: E402
from .geometry import RocketGeometry  # noqa: E402

# -- reading OpenFOAM output -------------------------------------------------------------


def read_vector_dat(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read forces/force.dat or moment.dat (v2012+ column format).

    Returns time and a dict with 'total', 'pressure', 'viscous' arrays of shape (n, 3).
    """
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        rows.append([float(x) for x in line.replace("(", " ").replace(")", " ").split()])
    data = np.array(rows)
    if data.ndim != 2 or data.shape[1] < 10:
        raise ValueError(f"Unexpected format in {path}")
    return data[:, 0], {"total": data[:, 1:4], "pressure": data[:, 4:7], "viscous": data[:, 7:10]}


def _latest_dat(func_dir: Path, name: str) -> Path:
    files = sorted(func_dir.glob(f"*/{name}"), key=lambda p: float(p.parent.name.split("_")[0]))
    if not files:
        raise FileNotFoundError(f"No {name} under {func_dir}")
    return files[-1]


def read_vtk_polydata(path: Path) -> tuple[np.ndarray, list[np.ndarray], dict[str, np.ndarray]]:
    """Minimal reader for OpenFOAM's legacy ASCII VTK surface output.

    Returns points (n, 3), a list of polygon index arrays, and point-data fields.
    """
    tok = path.read_text().split()
    i, n_tok = 0, len(tok)
    points = np.empty((0, 3))
    polys: list[np.ndarray] = []
    fields: dict[str, np.ndarray] = {}
    location = None
    while i < n_tok:
        t = tok[i]
        if t == "POINTS":
            n = int(tok[i + 1])
            points = np.array(tok[i + 3:i + 3 + 3 * n], dtype=float).reshape(n, 3)
            i += 3 + 3 * n
        elif t == "POLYGONS":
            total = int(tok[i + 2])
            arr = np.array(tok[i + 3:i + 3 + total], dtype=int)
            j = 0
            while j < total:
                k = arr[j]
                polys.append(arr[j + 1:j + 1 + k])
                j += k + 1
            i += 3 + total
        elif t in ("POINT_DATA", "CELL_DATA"):
            location = t
            i += 2
        elif t == "FIELD":
            n_arrays = int(tok[i + 2])
            i += 3
            for _ in range(n_arrays):
                name, nc, nt = tok[i], int(tok[i + 1]), int(tok[i + 2])
                vals = np.array(tok[i + 4:i + 4 + nc * nt], dtype=float)
                if location == "POINT_DATA":
                    fields[name] = vals.reshape(nt, nc) if nc > 1 else vals
                i += 4 + nc * nt
        else:
            i += 1
    return points, polys, fields


# -- coefficients ------------------------------------------------------------------------


def case_result(case: Path, geo: RocketGeometry, cfg: Config, regime, speed: float, aoa: float) -> dict:
    pp = case / "postProcessing"
    t, F = read_vector_dat(_latest_dat(pp / "forces", "force.dat"))
    _, M = read_vector_dat(_latest_dat(pp / "forces", "moment.dat"))
    n = min(cfg.solver.average_last, max(1, len(t) // 4))

    q = 0.5 * cfg.air.rho * speed**2
    qA = q * geo.ref_area
    d = geo.ref_diameter
    a = math.radians(aoa)

    f = F["total"][-n:].mean(axis=0)
    fp = F["pressure"][-n:].mean(axis=0)
    fv = F["viscous"][-n:].mean(axis=0)
    m = M["total"][-n:].mean(axis=0)

    ca, cn, cy = f / qA
    cd = ca * math.cos(a) + cn * math.sin(a)
    cl = cn * math.cos(a) - ca * math.sin(a)

    # Fluctuation of CD over the averaging window, as a convergence indicator.
    cd_hist = (F["total"][-n:] @ np.array([math.cos(a), math.sin(a), 0.0])) / qA
    unsteadiness = float(np.ptp(cd_hist) / max(abs(cd), 1e-9))

    xcp = m[2] / f[1] if abs(cn) > 0.02 else float("nan")
    solver_log = case / f"log.{regime.application}"
    converged = solver_log.exists() and "solution converged" in solver_log.read_text(errors="replace")

    res = {
        "case": case.name,
        "speed_mps": speed,
        "aoa_deg": aoa,
        "mach": speed / cfg.air.speed_of_sound,
        "reynolds_L": speed * geo.length / cfg.air.nu,
        "CD": cd,
        "CL": cl,
        "CA": ca,
        "CA_pressure": fp[0] / qA,
        "CA_viscous": fv[0] / qA,
        "CN": cn,
        "CY": cy,
        "Cm_nose": m[2] / (qA * d),
        "Xcp_m": xcp,
        "Xcp_cal": xcp / d,
        "drag_N": cd * qA,
        "iterations": int(t[-1]),
        "converged": converged,
        "cd_fluctuation": unsteadiness,
    }
    if cfg.geometry.cg_from_nose is not None and not math.isnan(xcp):
        res["stability_cal"] = (xcp - cfg.geometry.cg_from_nose) / d
    return res


RESULT_COLUMNS = ["case", "speed_mps", "aoa_deg", "mach", "reynolds_L", "CD", "CL", "CA", "CA_pressure",
                  "CA_viscous", "CN", "CY", "Cm_nose", "Xcp_m", "Xcp_cal", "stability_cal", "drag_N",
                  "iterations", "converged", "cd_fluctuation"]


def write_csv(results: list[dict], path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def cp_from_sweep(results: list[dict], speed: float, geo: RocketGeometry, cfg: Config,
                  max_aoa: float = 6.0) -> dict | None:
    """Centre of pressure from the nose, from the slope of moment vs normal force over angle of attack.

    Unlike the per-case X_cp = M/F, the slope cancels the force and moment present at zero angle
    of attack (asymmetric fins, launch lugs, small mesh asymmetry), so it is the robust number
    to use for stability. Needs at least two distinct angles |α| <= max_aoa at this speed.
    """
    pts = sorted({(r["aoa_deg"], r["CN"], r["Cm_nose"]) for r in results
                  if r["speed_mps"] == speed and abs(r["aoa_deg"]) <= max_aoa})
    if len({p[0] for p in pts}) < 2:
        return None
    _, cn, cm = np.array(pts).T
    if np.ptp(cn) < 0.05:  # force barely changes with angle: slope would be noise
        return None
    xcp = float(np.polyfit(cn, cm, 1)[0]) * geo.ref_diameter
    out = {"speed_mps": speed, "xcp_m": xcp, "xcp_cal": xcp / geo.ref_diameter,
           "xcp_frac_length": xcp / geo.length}
    if cfg.geometry.cg_from_nose is not None:
        out["stability_cal"] = (xcp - cfg.geometry.cg_from_nose) / geo.ref_diameter
    return out


def cn_alpha(results: list[dict], speed: float, max_aoa: float = 6.0) -> float | None:
    """Normal-force slope per radian from a linear fit through the small-angle points."""
    pts = [(math.radians(r["aoa_deg"]), r["CN"]) for r in results
           if r["speed_mps"] == speed and abs(r["aoa_deg"]) <= max_aoa]
    if len({a for a, _ in pts}) < 2:
        return None
    a, cn = np.array(pts).T
    return float(np.polyfit(a, cn, 1)[0])


# -- plots ---------------------------------------------------------------------------------


def plot_speed_sweep(results: list[dict], aoa: float, cfg: Config, out: Path) -> bool:
    rs = sorted((r for r in results if r["aoa_deg"] == aoa), key=lambda r: r["speed_mps"])
    if len(rs) < 2:
        return False
    v = [r["speed_mps"] for r in rs]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(v, [r["CD"] for r in rs], "o-", color="#1f4e8c", lw=2, label="C_D total")
    ax.plot(v, [r["CA_pressure"] for r in rs], "s--", color="#c0504d", label="pressure (form + base)")
    ax.plot(v, [r["CA_viscous"] for r in rs], "^--", color="#4f9a55", label="viscous (skin friction)")
    ax.set_xlabel("Speed [m/s]")
    ax.set_ylabel("Drag coefficient (ref. body cross-section)")
    ax.set_title(f"Drag vs speed at α = {aoa:g}°")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    a = cfg.air.speed_of_sound
    sec = ax.secondary_xaxis("top", functions=(lambda x: x / a, lambda m: m * a))
    sec.set_xlabel("Mach")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def plot_aoa_sweep(results: list[dict], speed: float, geo: RocketGeometry, cfg: Config, out: Path) -> bool:
    rs = sorted((r for r in results if r["speed_mps"] == speed), key=lambda r: r["aoa_deg"])
    if len(rs) < 2:
        return False
    a = [r["aoa_deg"] for r in rs]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(a, [r["CN"] for r in rs], "o-", color="#1f4e8c", lw=2)
    slope = cn_alpha(results, speed)
    if slope is not None:
        axes[0].plot(a, [slope * math.radians(x) for x in a], ":", color="grey",
                     label=f"C_Nα = {slope:.2f} /rad")
        axes[0].legend()
    axes[0].set_ylabel("Normal force coefficient C_N")

    xcp = [(x, r["Xcp_cal"]) for x, r in zip(a, rs) if not math.isnan(r["Xcp_cal"])]
    if xcp:
        axes[1].plot(*zip(*xcp), "o-", color="#c0504d", lw=2, label="CP (per case)")
    sweep_cp = cp_from_sweep(results, speed, geo, cfg)
    if sweep_cp:
        axes[1].axhline(sweep_cp["xcp_cal"], color="#c0504d", ls=":", label="CP (slope)")
    if cfg.geometry.cg_from_nose is not None:
        axes[1].axhline(cfg.geometry.cg_from_nose / geo.ref_diameter, color="k", ls="--", label="CG")
    axes[1].axhline(geo.length / geo.ref_diameter, color="grey", lw=0.8, label="tail")
    axes[1].invert_yaxis()
    axes[1].set_ylabel("Centre of pressure from nose [calibers]")
    axes[1].legend(loc="center right")

    axes[2].plot(a, [r["CD"] for r in rs], "o-", color="#4f9a55", lw=2)
    axes[2].set_ylabel("Drag coefficient C_D")

    for ax in axes:
        ax.set_xlabel("Angle of attack [deg]")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Angle-of-attack sweep at {speed:g} m/s (Mach {speed / cfg.air.speed_of_sound:.2f})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def render_flow(case: Path, geo: RocketGeometry, cfg: Config, regime, speed: float, aoa: float,
                out: Path) -> bool:
    """Cp and velocity magnitude on the pitch plane (z = 0)."""
    slices = case / "postProcessing" / "slices"
    times = sorted((p for p in slices.glob("*") if p.is_dir()), key=lambda p: float(p.name)) if slices.exists() else []
    if not times or not (times[-1] / "pitchPlane.vtk").exists():
        return False
    pts, polys, fields = read_vtk_polydata(times[-1] / "pitchPlane.vtk")
    tris = np.array([(p[0], p[k], p[k + 1]) for p in polys for k in range(1, len(p) - 1)])
    cp = regime.pressure_coefficient(fields["p"], speed, cfg)
    umag = np.linalg.norm(fields["U"], axis=1) / speed

    L, R = geo.length, geo.max_radius
    half_h = max(2 * R, 0.1 * L)
    xlim, ylim = (-0.08 * L, 1.35 * L), (-half_h, half_h)
    in_view = (pts[:, 0] > xlim[0]) & (pts[:, 0] < xlim[1]) & (np.abs(pts[:, 1]) < half_h)
    cp_lim = float(np.clip(np.percentile(np.abs(cp[in_view]), 99), 0.2, 1.0)) if in_view.any() else 1.0

    panel_w = 11.0
    panel_h = panel_w * (2 * half_h) / (xlim[1] - xlim[0])
    fig, axes = plt.subplots(2, 1, figsize=(panel_w + 1.6, 2 * panel_h + 1.3), sharex=True,
                             layout="constrained")
    for ax, values, label, cmap, lim in [
        (axes[0], cp, "Pressure coefficient C_p", "RdBu_r", (-cp_lim, cp_lim)),
        (axes[1], umag, "Velocity magnitude |U| / V∞", "viridis", (0.0, 1.3)),
    ]:
        ax.set_facecolor("#5a5a5a")  # the rocket is a hole in the slice
        tpc = ax.tripcolor(pts[:, 0], pts[:, 1], tris, np.clip(values, *lim), shading="gouraud",
                           cmap=cmap, vmin=lim[0], vmax=lim[1])
        fig.colorbar(tpc, ax=ax, label=label, fraction=0.03, pad=0.01, aspect=12)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_ylabel("y [m]")
    axes[1].set_xlabel("x [m]")
    axes[0].set_title(f"Pitch plane, V = {speed:g} m/s, α = {aoa:g}° (flow from the left)")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return True


# -- report --------------------------------------------------------------------------------


def write_report(path: Path, cfg: Config, geo: RocketGeometry, mesh_stats: dict, plan: dict,
                 results: list[dict], warnings: list[str], case_root: str, images: dict[str, str]) -> None:
    g = geo.summary()
    lines = [
        f"# CFD report: {cfg.name}",
        "",
        f"Model: `{cfg.model}`  ",
        f"Regime: {cfg.regime}, mesh preset: {cfg.mesh.preset}  ",
        f"OpenFOAM cases: `{case_root}`",
        "",
        "## Geometry",
        "",
        f"- Length: {g['length_m']} m, reference diameter: {g['ref_diameter_m'] * 1000:.2f} mm "
        f"(reference area {g['ref_area_m2']:.4g} m²), fineness ratio {g['fineness_ratio']}",
        f"- Moments about the nose tip; reference length = reference diameter",
        f"- Watertight: {g['watertight']}, source units: {g['source_units']}",
        "",
        "![geometry](geometry.png)",
        "",
        "## Mesh",
        "",
        f"- Cells: {mesh_stats.get('cells', '?'):,}, max non-orthogonality: "
        f"{mesh_stats.get('max_non_orthogonality', '?')}, max skewness: {mesh_stats.get('max_skewness', '?')}",
        f"- Surface cell size: {plan['surface_size'] * 1000:.2f} mm (level {plan['surface_level']}), "
        f"boundary layers: {plan['layers']}",
        f"- checkMesh: {'OK' if mesh_stats.get('ok') else 'FAILED ' + str(mesh_stats.get('failed_checks'))}",
        "",
    ]
    if warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in warnings] + [""]

    if results:
        lines += ["## Results", "",
                  "| case | V [m/s] | α [°] | Mach | C_D | C_A press. | C_A visc. | C_N | X_cp [cal] | drag [N] | converged |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in results:
            xcp = "–" if math.isnan(r["Xcp_cal"]) else f"{r['Xcp_cal']:.2f}"
            conv = "yes" if r["converged"] else f"no ({r['cd_fluctuation']:.1%} CD swing)"
            lines.append(f"| {r['case']} | {r['speed_mps']:g} | {r['aoa_deg']:g} | {r['mach']:.3f} | "
                         f"{r['CD']:.4f} | {r['CA_pressure']:.4f} | {r['CA_viscous']:.4f} | {r['CN']:.4f} | "
                         f"{xcp} | {r['drag_N']:.3f} | {conv} |")
        lines.append("")
        if cfg.sweeps.aoa:
            slope = cn_alpha(results, cfg.sweeps.aoa.speed)
            if slope is not None:
                lines += [f"C_Nα (α ≤ 6°, {cfg.sweeps.aoa.speed:g} m/s): **{slope:.3f} per radian**", ""]
            cp = cp_from_sweep(results, cfg.sweeps.aoa.speed, geo, cfg)
            if cp is not None:
                lines += ["### Centre of pressure", "",
                          f"- **{cp['xcp_m'] * 1000:.0f} mm from the nose tip** "
                          f"({cp['xcp_cal']:.2f} calibers, {cp['xcp_frac_length']:.0%} of the length), "
                          f"from the moment/force slope over the angle sweep at {cp['speed_mps']:g} m/s"]
                if "stability_cal" in cp:
                    lines.append(f"- Stability margin: **{cp['stability_cal']:.2f} calibers** with the CG at "
                                 f"{cfg.geometry.cg_from_nose * 1000:.0f} mm (above 1 is the usual rule of thumb)")
                lines.append("")
        for title, img in images.items():
            lines += [f"### {title}", "", f"![{title}]({img})", ""]

    lines += ["## Notes", "",
              "- Steady RANS (k-ω SST, wall functions), "
              + ("compressible perfect gas (rhoSimpleFoam)." if cfg.regime == "compressible"
                 else "incompressible (simpleFoam).")
              + " Fully turbulent: no laminar run on the nose, so drag at low Reynolds number is "
              "somewhat overestimated.",
              "- Coefficients are averaged over the last iterations; check `cd_fluctuation` and the "
              "`converged` column before trusting a number.",
              "- Run a finer mesh preset to check the numbers do not change much (mesh independence).",
              ""]
    path.write_text("\n".join(lines), encoding="utf-8")
