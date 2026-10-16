"""Load rocket geometry and put it in the pipeline's standard frame.

Standard frame (metres): the nose tip is at the origin, the rocket axis runs along +x
toward the tail, and +y is "up" for angle of attack (pitch plane is x-y).
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from .config import GeometryConfig

log = logging.getLogger(__name__)

UNIT_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254}
AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


@dataclass
class RocketGeometry:
    mesh: trimesh.Trimesh
    length: float          # m, nose tip to aftmost point
    ref_diameter: float    # m, body diameter used for reference area/length
    max_radius: float      # m, largest radial extent (usually fin tips)
    watertight: bool
    source_units: str
    warnings: list[str]

    @property
    def ref_area(self) -> float:
        return np.pi * self.ref_diameter**2 / 4

    def summary(self) -> dict:
        return {
            "length_m": round(self.length, 5),
            "ref_diameter_m": round(self.ref_diameter, 5),
            "ref_area_m2": float(f"{self.ref_area:.6g}"),
            "max_radius_m": round(self.max_radius, 5),
            "fineness_ratio": round(self.length / self.ref_diameter, 2),
            "triangles": int(len(self.mesh.faces)),
            "watertight": self.watertight,
            "source_units": self.source_units,
            "warnings": self.warnings,
        }


def load_surface(path: Path) -> tuple[trimesh.Trimesh, bool]:
    """Read an STL or STEP file. Returns the mesh and whether it is already in metres."""
    ext = path.suffix.lower()
    if ext == ".stl":
        return trimesh.load(path, force="mesh"), False
    if ext in (".step", ".stp"):
        return _tessellate_step(path), True
    raise ValueError(f"Unsupported model format '{ext}' (use .stl, .step or .stp)")


def _tessellate_step(path: Path) -> trimesh.Trimesh:
    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError("STEP input needs gmsh: pip install gmsh") from exc

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(-1, -1)
        size = max(x1 - x0, y1 - y0, z1 - z0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 36)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size / 150)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size / 1500)
        gmsh.model.mesh.generate(2)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "surface.stl"
            gmsh.write(str(out))
            return trimesh.load(out, force="mesh")
    finally:
        gmsh.finalize()


def _guess_units(extent: float) -> str:
    # Model rockets are roughly 0.2 - 4 m long.
    if extent > 100:
        return "mm"
    if extent > 8:
        return "in" if extent < 60 else "cm"
    return "m"


def _slice_radius(mesh: trimesh.Trimesh, x: float, center=(0.0, 0.0), bins: int = 36) -> float | None:
    """Body radius at station x, robust to fins and launch lugs.

    The section is split into angular bins; each bin keeps its outermost point, and the
    median over bins is returned. Fins and lugs only occupy a few bins so they drop out.
    """
    segs = trimesh.intersections.mesh_plane(mesh, plane_normal=[1, 0, 0], plane_origin=[x, 0, 0])
    if len(segs) == 0:
        return None
    pts = segs.reshape(-1, 3)[:, 1:] - np.asarray(center)
    r = np.hypot(pts[:, 0], pts[:, 1])
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    idx = np.floor((theta + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    rmax = np.zeros(bins)
    np.maximum.at(rmax, idx, r)
    filled = rmax[rmax > 0]
    if len(filled) < bins // 2:
        return None
    return float(np.median(filled))


def _profile(mesh: trimesh.Trimesh, fractions: np.ndarray, center=(0.0, 0.0)) -> np.ndarray:
    x0, x1 = mesh.bounds[:, 0]
    out = [_slice_radius(mesh, x0 + f * (x1 - x0), center) for f in fractions]
    return np.array([np.nan if r is None else r for r in out])


def _clean(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Merge near-duplicate vertices, drop degenerate triangles and make normals point outward."""
    scale = float(mesh.extents.max())
    mesh.merge_vertices(digits_vertex=max(0, int(-np.log10(scale * 1e-6))))
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def _drop_fragments(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Remove slivers left by tessellation (e.g. at a cone apex); they are far below mesh resolution."""
    bodies = mesh.split(only_watertight=False)
    if len(bodies) < 2:
        return mesh
    size = float(mesh.extents.max())
    keep = [b for b in bodies if b.extents.max() > 0.005 * size]
    if len(keep) < len(bodies):
        log.info("Dropped %d tiny fragment(s) from the surface", len(bodies) - len(keep))
    return trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]


def prepare_geometry(path: Path, cfg: GeometryConfig) -> RocketGeometry:
    """Load, scale, orient and measure a rocket model."""
    mesh, in_metres = load_surface(path)
    warnings: list[str] = []
    mesh = _drop_fragments(_clean(mesh))

    if cfg.union_bodies:
        bodies = mesh.split(only_watertight=False)
        if len(bodies) > 1:
            if all(b.is_watertight for b in bodies):
                log.info("Merging %d separate bodies into one solid", len(bodies))
                mesh = trimesh.boolean.union(bodies, engine="manifold")
            else:
                warnings.append(f"Model has {len(bodies)} separate bodies, not all closed; "
                                "could not merge them. Overlapping shells may mesh badly.")

    # Units
    extent = float(mesh.extents.max())
    units = "m" if in_metres else cfg.units
    if units == "auto":
        units = _guess_units(extent)
        log.info("Units not given; longest extent %.4g looks like %s", extent, units)
    if units not in UNIT_SCALE:
        raise ValueError(f"Unknown units '{units}', choose from {list(UNIT_SCALE)}")
    mesh.apply_scale(UNIT_SCALE[units])

    # Axis and nose direction
    if cfg.nose_direction == "auto":
        axis = int(np.argmax(mesh.extents))
        nose = np.eye(3)[axis]
        # Put the candidate axis along x to measure both ends.
        trial = mesh.copy()
        trial.apply_transform(trimesh.geometry.align_vectors(nose, [1, 0, 0]))
        frac = np.linspace(0.02, 0.15, 6)
        lo = np.nanmean(_profile(trial, frac))
        hi = np.nanmean(_profile(trial, 1 - frac))
        # Nose is the slimmer end.
        if hi > lo:
            nose = -nose
    else:
        if cfg.nose_direction not in AXES:
            raise ValueError(f"nose_direction must be 'auto' or one of {list(AXES)}")
        nose = np.array(AXES[cfg.nose_direction], dtype=float)

    mesh.apply_transform(trimesh.geometry.align_vectors(nose, [-1, 0, 0]))
    if cfg.roll_deg:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(cfg.roll_deg), [1, 0, 0]))
    mesh.apply_translation([-mesh.bounds[0, 0], 0, 0])

    # Centre the body tube on the x axis using mid-length sections (fins are usually aft).
    length = float(mesh.extents[0])
    centres = []
    for f in np.linspace(0.3, 0.6, 7):
        segs = trimesh.intersections.mesh_plane(mesh, [1, 0, 0], [f * length, 0, 0])
        if len(segs):
            pts = segs.reshape(-1, 3)[:, 1:]
            centres.append((pts.min(axis=0) + pts.max(axis=0)) / 2)
    yz = np.median(centres, axis=0) if centres else mesh.bounds.mean(axis=0)[1:]
    mesh.apply_translation([0, -yz[0], -yz[1]])

    # Reference diameter
    radii = _profile(mesh, np.linspace(0.05, 0.98, 60))
    if cfg.ref_diameter:
        ref_d = float(cfg.ref_diameter)
    else:
        if np.all(np.isnan(radii)):
            raise RuntimeError("Could not measure the body diameter; set geometry.ref_diameter")
        ref_d = 2 * float(np.nanmax(radii))
    max_radius = float(np.hypot(mesh.vertices[:, 1], mesh.vertices[:, 2]).max())

    if not mesh.is_watertight:
        warnings.append("Surface is not watertight (has holes or open edges). snappyHexMesh "
                        "needs a closed outer surface; check for hollow tubes or missing faces.")
    if not 0.05 < length < 10:
        warnings.append(f"Rocket length {length:.3g} m looks wrong; check geometry.units")
    if not 3 < length / ref_d < 60:
        warnings.append(f"Fineness ratio {length / ref_d:.1f} is unusual; check ref_diameter")
    for w in warnings:
        log.warning(w)

    return RocketGeometry(mesh=mesh, length=length, ref_diameter=ref_d, max_radius=max_radius,
                          watertight=bool(mesh.is_watertight), source_units=units, warnings=warnings)


def plot_geometry(geo: RocketGeometry, out: Path) -> None:
    """Side and top views of the normalised model, to check orientation and scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    tris = geo.mesh.triangles
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    for ax, (j, label) in zip(axes, [(1, "y (pitch-up) [m]"), (2, "z [m]")]):
        ax.add_collection(PolyCollection(tris[:, :, [0, j]], facecolor="#8aa4c8", edgecolor="#2b4a75",
                                         linewidths=0.1))
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.set_ylabel(label)
        ax.axhline(0, color="k", lw=0.5, ls="--")
    axes[0].set_title(f"Normalised geometry: L = {geo.length:.3f} m, d_ref = {geo.ref_diameter * 1000:.1f} mm "
                      f"(flow comes from the left)")
    axes[1].set_xlabel("x [m]")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
