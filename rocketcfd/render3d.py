"""3D presentation render of a finished case: surface C_p, streamlines and a pitch-plane speed slice.

Needs PyVista (``pip install -e ".[render]"``). Without it, rendering is skipped with a note.
Sizes, seeds and the camera all scale with the rocket's length and radius, so any model frames
the same way: nose to the left, flow from the left, seen from slightly above and in front.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .config import Config
from .geometry import RocketGeometry

log = logging.getLogger(__name__)

BACKGROUND = "#0b0f17"
SPEED_LIM = (0.0, 1.2)


def available() -> bool:
    try:
        import pyvista  # noqa: F401
    except ImportError:
        return False
    return True


def render_case(case: Path, geo: RocketGeometry, cfg: Config, regime, speed: float, aoa: float,
                out: Path, title: str | None = None) -> bool:
    """Render the latest time of an OpenFOAM case to ``out`` (PNG). Returns False if it could not."""
    import pyvista as pv

    data = _read_case(pv, case)
    if data is None:
        return False
    internal, wall = data
    L, R = geo.length, geo.max_radius

    wall = wall.cell_data_to_point_data()
    wall["Cp"] = regime.pressure_coefficient(np.asarray(wall["p"]), speed, cfg)
    cp_lim = float(np.clip(np.ceil(np.percentile(np.abs(wall["Cp"]), 95) * 10) / 10, 0.2, 1.0))

    # Work in a box around the rocket: much faster streamlines and slices.
    box = _crop_cells(internal, (-0.33 * L, -3 * R, -3 * R), (1.7 * L, 3 * R, 3 * R))
    box = box.cell_data_to_point_data()
    box["Umag"] = np.linalg.norm(box["U"], axis=1) / speed

    # Free-stream lines: a sparse ring just outside the fin tips, upstream of the nose.
    ring = [(-0.16 * L, 1.5 * R * np.cos(t), 1.5 * R * np.sin(t))
            for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
    outer = box.streamlines_from_source(pv.PolyData(np.array(ring)), vectors="U",
                                        integration_direction="forward", max_steps=6000,
                                        initial_step_length=0.2, max_length=3 * L, terminal_speed=1e-3)
    # Wake lines: seeded just behind the base, short enough to stay in the recirculation.
    disc = [(1.016 * L, f * R * np.cos(t), f * R * np.sin(t))
            for f in (0.14, 0.3, 0.47, 0.68) for t in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    wake = box.streamlines_from_source(pv.PolyData(np.array(disc)), vectors="U",
                                       integration_direction="forward", max_steps=6000,
                                       initial_step_length=0.2, max_length=0.4 * L, terminal_speed=1e-3)
    pitch = box.slice(normal="z", origin=(0, 0, 0)).clip_box(
        (-0.11 * L, 1.7 * L, -1.5 * R, 1.5 * R, -1, 1), invert=False)

    p = pv.Plotter(off_screen=True, window_size=(2400, 1000))
    p.set_background(BACKGROUND)
    p.add_mesh(pitch, scalars="Umag", cmap="inferno_r", clim=SPEED_LIM, opacity=0.95, lighting=False,
               show_scalar_bar=False)
    if outer.n_points:
        p.add_mesh(outer.tube(radius=0.012 * R), color="#cfd9e8", opacity=0.35, smooth_shading=True)
    if wake.n_points:
        p.add_mesh(wake.tube(radius=0.014 * R), scalars="Umag", cmap="cool", clim=(0, 1),
                   smooth_shading=True, show_scalar_bar=False)
    p.add_mesh(wall, scalars="Cp", cmap="RdBu_r", clim=(-cp_lim, cp_lim), smooth_shading=True,
               specular=0.6, specular_power=40, show_scalar_bar=False)
    p.camera.focal_point = (0.64 * L, 0.0, 0.0)
    p.camera.position = (0.46 * L, 0.41 * L, 1.8 * L)
    p.camera.up = (0, 1, 0)
    p.camera.view_angle = 26
    p.enable_anti_aliasing("ssaa")
    scene = p.screenshot(None, return_img=True)
    p.close()

    if title is None:
        title = f"{cfg.name}  ·  {speed:g} m/s at {aoa:g}° angle of attack"
    _compose(autocrop(scene), out, title, cp_lim)
    return True


def _read_case(pv, case: Path):
    foam = case / "case.foam"
    if not foam.exists():
        foam.write_text("")
    reader = pv.OpenFOAMReader(str(foam))
    if not reader.time_values:
        log.warning("No time directories in %s; nothing to render", case)
        return None
    reader.set_active_time_value(reader.time_values[-1])
    reader.enable_all_patch_arrays()
    data = reader.read()
    boundary = data["boundary"]
    if "rocket" not in boundary.keys():
        log.warning("No 'rocket' patch in %s; nothing to render", case)
        return None
    return data["internalMesh"], boundary["rocket"]


def _crop_cells(mesh, lo, hi):
    """Whole cells whose centres lie in the box, split into tetrahedra.

    Clipping cuts snappyHexMesh's polyhedral cells (at refinement-level changes) badly in VTK 9.5,
    leaving holes in slices and a mesh streamlines cannot find points in. Keeping cells whole and
    then tetrahedralising avoids both, and the ragged box edge is outside the framed view.
    """
    c = mesh.cell_centers().points
    inside = np.all((c >= np.asarray(lo)) & (c <= np.asarray(hi)), axis=1)
    return mesh.extract_cells(np.flatnonzero(inside)).triangulate()


def autocrop(img: np.ndarray, pad: int = 40) -> np.ndarray:
    """Trim rows that are plain background from the top and bottom of a rendered image."""
    bg = np.array([int(BACKGROUND[i:i + 2], 16) for i in (1, 3, 5)])
    busy = np.abs(img[..., :3].astype(int) - bg).max(axis=2) > 12
    rows = np.flatnonzero(busy.any(axis=1))
    if not len(rows):
        return img
    return img[max(0, rows[0] - pad):min(len(img), rows[-1] + 1 + pad)]


def _compose(scene: np.ndarray, out: Path, title: str, cp_lim: float) -> None:
    """Add the title, caption and colour bars with matplotlib, which sets type better than VTK."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    w_in = 24.0
    scene_h = w_in * scene.shape[0] / scene.shape[1]
    top, bottom = 1.5, 1.9  # inches for the title block and the colour bars
    fig_h = scene_h + top + bottom
    fig = plt.figure(figsize=(w_in, fig_h), dpi=100, facecolor=BACKGROUND)
    ax = fig.add_axes((0, bottom / fig_h, 1, scene_h / fig_h))
    ax.imshow(scene)
    ax.axis("off")
    fig.text(0.025, 1 - 0.45 / fig_h, title, color="white", fontsize=30, va="top", weight="bold")
    fig.text(0.025, 1 - 1.1 / fig_h, "Rocket coloured by surface pressure; the slice through the axis "
             "shows flow speed, with the slow recirculating wake behind the base",
             color="#9aa7b8", fontsize=18, va="top")
    cp_ticks = np.round(np.linspace(-cp_lim, cp_lim, 5), 2)
    for x0, cmap, lim, label, ticks in (
            (0.14, "RdBu_r", (-cp_lim, cp_lim), "Surface pressure coefficient  Cp", cp_ticks),
            (0.58, "inferno_r", SPEED_LIM, "Flow speed / free-stream speed", [0, 0.3, 0.6, 0.9, 1.2])):
        cax = fig.add_axes((x0, 0.6 / fig_h, 0.28, 0.28 / fig_h))
        cb = fig.colorbar(ScalarMappable(Normalize(*lim), cmap), cax=cax, orientation="horizontal",
                          ticks=ticks)
        cb.outline.set_visible(False)
        cax.tick_params(colors="white", labelsize=16, length=0, pad=8)
        cax.set_title(label, color="white", fontsize=18, pad=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
