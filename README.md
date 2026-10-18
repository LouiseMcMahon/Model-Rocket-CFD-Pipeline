# Rocket CFD Pipeline

Pass in a model rocket (STL or STEP) and get its aerodynamics from OpenFOAM:

- **Drag vs speed**: C_D split into pressure (form + base) and skin-friction parts
- **Angle-of-attack sweep**: normal force C_N, C_Nα, centre of pressure, and stability margin if you give a CG
- **Flow pictures**: pressure coefficient and velocity on the pitch plane for each case
- A Markdown report, `results.csv`, and the full OpenFOAM cases for inspection in ParaView

```
model.stl / .step
   │  geometry: load → units → find axis & nose → align nose at origin, axis +x → measure diameter
   ▼
snappyHexMesh (once per model)          ← mesh preset: coarse / medium / fine
   ▼
simpleFoam, k-ω SST, one case per (speed, AoA); AoA is applied by turning the free-stream,
so every case reuses the same mesh
   ▼
forces → C_D, C_L, C_A, C_N, C_m, X_cp   +   plots, flow images, report.md
```

## Requirements

- OpenFOAM (ESI/openfoam.com, tested with **v2412**). On Windows it runs inside **WSL**; on Linux
  it runs natively. The pipeline sources `openfoam.bashrc` itself, so nothing is needed in your shell.
- Python 3.10+ on the machine you run the pipeline from.

## Setup (Windows)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[step]"
.\.venv\Scripts\rocketcfd check
```

`check` should print `OpenFOAM v2412 found via wsl runner`. (Use `py -3.12` rather than `python`:
on this machine `python` on PATH is Inkscape's bundled interpreter.)

## Usage

```powershell
# 1. Look at the model: prints length/diameter and writes <model>_preview.png
rocketcfd inspect models\my_rocket.stl

# 2. Write a config next to the model, then edit speeds, angles, CG, etc.
rocketcfd init models\my_rocket.stl

# 3. Run
rocketcfd run models\my_rocket.yaml
```

Results land in `runs/<name>/`:

| file | what |
|---|---|
| `report.md` | geometry, mesh quality, results table, all plots |
| `results.csv` | every coefficient for every case |
| `drag_vs_speed.png`, `aoa_sweep.png` | sweep plots |
| `flow/*.png` | C_p and velocity on the pitch plane |
| `geometry.png`, `rocket_normalized.stl` | the model as the pipeline sees it, to check orientation and scale |
| `pipeline.log` | full log |

The OpenFOAM cases live on the Linux side in `~/rocketcfd_runs/<name>/` (from Windows:
`\\wsl.localhost\Ubuntu\home\<you>\rocketcfd_runs`). Open a case in ParaView by creating an empty
`case.foam` file in it.

**Resuming:** each finished mesh and case is fingerprinted. Re-running the same config skips anything
already done, so you can stop with Ctrl+C, add speeds or angles, and run again. `--force` redoes everything.

Handy flags: `--preset coarse` for a quick look, `--mesh-only` to check the mesh before committing
to a sweep, `--nprocs`, `--iterations`, `-v`.

Try it on the bundled test rocket:

```powershell
python examples\make_test_rocket.py models\test_rocket.stl
rocketcfd run examples\test_rocket_quick.yaml
```

## Preparing your model

- **Closed outer surface.** Export the outer shape of the assembled rocket as a solid. Hollow body tubes
  with an open aft end let air flow inside the tube. Fill them, or cap the base. Separate closed bodies
  (nose, tube, fins) are merged automatically.
- **Units.** STL has no units; `units: auto` guesses from size (a 600-long model is taken as mm). Set it
  explicitly if `inspect` shows the wrong length. STEP files carry units and are read in metres.
- **Orientation.** The axis is taken as the longest dimension, and the nose as the slimmer end.
  Override with `nose_direction: +z` (the direction the nose points in your file) if needed.
- **Fins vs angle of attack.** The pitch plane is x-y. With 4 fins, `roll_deg: 0` gives a "+" layout
  (two fins in the pitch plane) and `roll_deg: 45` gives an "x" layout.
- **Thin fins** need a fine enough mesh: the surface cell size (printed by `inspect`) should be well under
  the fin thickness for good fin-edge drag. Use `preset: fine` or set `mesh.cells_per_diameter`.

## Conventions

- Reference area = body tube cross-section; reference length = body diameter.
- Axes: x along the rocket from the nose tip, y = pitch-up. Moments about the nose tip.
- C_A / C_N are body-axis forces; C_D / C_L are wind-axis forces. At α = 0 they coincide.
- **Centre of pressure from the nose tip.** Each case reports X_cp = M_z / F_y, but that is unreliable at
  small angles and includes any force present at 0° (asymmetric fins, launch lugs). The report's
  *Centre of pressure* section instead takes the slope of moment against normal force over the angle
  sweep (|α| ≤ 6°), which cancels the 0° offset. It is given in mm, calibers and % of length, and is also
  written to `centre_of_pressure.json`. It needs at least two angles in the sweep, and a normal force that
  actually changes between them.
- Stability margin = (X_cp − X_cg) / d, shown when `geometry.cg_from_nose` is set.

## Accuracy notes

- Steady RANS with k-ω SST and wall functions (Spalding, valid at any y+), fully turbulent.
  Real model rockets often have laminar flow over the nose, so drag at low speed is somewhat high.
- Incompressible: fine below Mach ~0.3, usable to ~0.5 with a warning, refused above 0.6. Use the
  `compressible` regime above that.
- Always check mesh independence (re-run with `preset: fine` and compare C_D) before trusting
  absolute numbers. Trends between designs on the same mesh settings are more robust.
- Check the `converged` and `cd_fluctuation` columns. Bluff-base wakes can oscillate in steady solvers.

## Flow regimes

Set `regime:` in the config. Both regimes share the geometry, mesh, sweeps and force post-processing.

| regime | solver | use for |
|---|---|---|
| `incompressible` (default) | simpleFoam, k-ω SST | Mach < 0.3 (accepted up to 0.6 with a warning) |
| `compressible` | rhoSimpleFoam, k-ω SST, perfect gas, Sutherland viscosity | Mach 0.3 to 1.5 |

For `compressible`, the free stream comes from `air.pressure` (101325 Pa) and `air.temperature` (288.15 K);
the Mach number in the report uses the speed of sound at that temperature. The wall is adiabatic.
Warnings are written to the report for transonic (Mach 0.85-1) and supersonic (Mach >= 1) cases.

- **Supersonic cases** use first-order upwind convection. Steady second-order schemes oscillate against
  the bow shock and never converge. Shocks are therefore smeared over several cells.
- **Not yet done:** meshing that refines around shocks, and a transient density-based solver
  (rhoCentralFoam) for sharper supersonic shocks. Compressible cases are slower than incompressible ones.
- Compressible and incompressible drag differ a little even at low Mach (different numerics), so compare
  designs within one regime.

Flow regimes live in `rocketcfd/regimes/`. A regime supplies the fields, numerics and solver commands on
top of the shared machinery (see `base.py`).

## Layout

```
rocketcfd/
  config.py        YAML config (dataclasses with defaults)
  geometry.py      load STL/STEP, clean, orient, measure
  mesh.py          blockMesh + snappyHexMesh dictionaries and sizing
  regimes/         incompressible (simpleFoam); compressible placeholder
  runner.py        runs OpenFOAM natively or in WSL
  post.py          coefficients, plots, flow images, report
  pipeline.py      orchestration and resume logic
  cli.py           rocketcfd command
examples/          test rocket generator and a quick config
tests/             unit tests (no OpenFOAM needed): pytest
```
