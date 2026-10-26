# Rocket CFD Pipeline

Pass in a model rocket (STL or STEP) and get its aerodynamics from OpenFOAM:

- **Drag vs speed**: C_D split into pressure (form + base) and skin-friction parts
- **Angle-of-attack sweep**: normal force C_N, C_Nα, centre of pressure, and stability margin if you give a CG
- **Flow pictures**: pressure coefficient and velocity on the pitch plane for each case
- **3D renders** for presentations: the rocket coloured by surface pressure, with streamlines and the wake
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
.\.venv\Scripts\python -m pip install -e ".[step,render]"
.\.venv\Scripts\rocketcfd check
```

`check` should print `OpenFOAM v2412 found via wsl runner`. (Use `py -3.12` rather than `python`:
on this machine `python` on PATH is Inkscape's bundled interpreter.)

## Quick start

Run the example on a generated test rocket (coarse mesh, 6 cases, roughly 15 minutes on 8 cores):

```powershell
.\.venv\Scripts\python examples\make_test_rocket.py models\test_rocket.stl
.\.venv\Scripts\rocketcfd run examples\example.yaml
```

Then open `runs\example\report.md`. The commands below assume the venv is active
(`.\.venv\Scripts\Activate.ps1`), so `rocketcfd` is on PATH.

## Running your own rocket

Put the model in `models\`. Git ignores everything in that folder, so your models and their run
configs stay local.

1. **Check the geometry.** `rocketcfd inspect models\my_rocket.stl` prints length and diameter and
   writes `models\my_rocket_preview.png`. Make sure the scale is right and the nose is where you expect.
2. **Write a config.** Either `rocketcfd init models\my_rocket.stl`, which writes `models\my_rocket.yaml`
   with the main settings, or copy [`examples/example.yaml`](examples/example.yaml), which lists every
   setting with a comment. Set the speeds, angles and CG you care about.
3. **Check the mesh.** `rocketcfd run models\my_rocket.yaml --mesh-only` builds the mesh and writes
   `runs\<name>\geometry.png` (the model as the solver sees it) and `mesh.json` (cell count and quality).
4. **Run the sweep.** `rocketcfd run models\my_rocket.yaml`. Each case takes from a few minutes (coarse)
   to around an hour (fine, with tail refinement), so start with `--preset coarse`.
5. **Read the results** in `runs\<name>\report.md`.

### The config file

[`examples/example.yaml`](examples/example.yaml) is the reference: every setting, its default and what it
does. The ones you will change most:

| setting | what it does |
|---|---|
| `model` | STL or STEP file, relative to the config |
| `geometry.nose_direction` | set it if `inspect` puts the nose at the wrong end (common with boat-tails and thin booms) |
| `geometry.cg_from_nose` | CG in metres from the nose tip; adds the stability margin to the report |
| `geometry.ref_diameter` | reference diameter in metres, if the auto-detected body diameter is not the one you want |
| `mesh.preset` | `coarse` for a quick look, `medium` by default, `fine` to check mesh independence |
| `mesh.upstream` / `downstream` / `lateral`, `mesh.tail_refinement` | domain size and extra refinement over the fins and base wake |
| `solver.iterations`, `solver.average_last` | run length, and how much of the end is averaged into the results |
| `sweeps.speed`, `sweeps.aoa` | the drag-vs-speed and angle-of-attack sweeps; `null` skips one |
| `regime` | `incompressible` below Mach 0.3, `compressible` from 0.3 to 1.5 |

Results land in `runs/<name>/`:

| file | what |
|---|---|
| `report.md` | geometry, mesh quality, results table, all plots |
| `results.csv` | every coefficient for every case |
| `drag_vs_speed.png`, `aoa_sweep.png` | sweep plots |
| `flow/*.png` | C_p and velocity on the pitch plane |
| `render/*.png` | 3D presentation image per case; the largest angle is also in the report |
| `geometry.png`, `rocket_normalized.stl` | the model as the pipeline sees it, to check orientation and scale |
| `pipeline.log` | full log |

The OpenFOAM cases live on the Linux side in `~/rocketcfd_runs/<name>/` (from Windows:
`\\wsl.localhost\Ubuntu\home\<you>\rocketcfd_runs`). Open a case in ParaView by creating an empty
`case.foam` file in it.

**Sleep:** on Windows the run asks the PC not to sleep until it finishes (a sleeping PC pauses WSL and
the solver with it). Closing a laptop lid or choosing Sleep yourself still suspends it.

**Resuming:** each finished mesh and case is fingerprinted. Re-running the same config skips anything
already done, so you can stop with Ctrl+C, add speeds or angles, and run again. `--force` redoes everything.

**Progress:** `runs\<name>\pipeline.log` shows each case as it starts and finishes.

Handy flags: `--preset coarse` for a quick look, `--mesh-only` to check the mesh before committing
to a sweep, `--nprocs`, `--iterations`, `-v`.

## 3D renders

With the `render` extra installed (`pip install -e ".[render]"`, which adds PyVista), every case also gets
a 3D image in `runs\<name>\render\`, made for slides and sharing:

- the rocket coloured by surface pressure coefficient
- a slice through the axis shaded by flow speed, which shows the wake behind the base
- streamlines arriving at the angle of attack, and the recirculation just behind the base

The report shows the largest angle of the angle-of-attack sweep. The camera and streamlines scale with the
rocket, so any model is framed the same way: nose to the left, seen from slightly above.

To redo the images of cases that are already solved, without running OpenFOAM again:

```powershell
rocketcfd render models\my_rocket.yaml            # every solved case
rocketcfd render models\my_rocket.yaml --aoa 6    # one angle (--speed works the same way)
```

Each image takes a few seconds. Without PyVista the run still works and just skips these images.
Set `render_3d: false` in the config to turn them off.

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
  regimes/         incompressible (simpleFoam) and compressible (rhoSimpleFoam)
  runner.py        runs OpenFOAM natively or in WSL
  post.py          coefficients, plots, flow images, report
  render3d.py      3D presentation renders (PyVista, optional)
  pipeline.py      orchestration and resume logic
  cli.py           rocketcfd command
examples/          example.yaml (every setting, commented) and a test rocket generator
models/            your models and run configs (ignored by git)
tests/             unit tests (no OpenFOAM needed): pytest
```
