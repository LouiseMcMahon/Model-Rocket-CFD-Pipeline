"""Tests for the parts of the pipeline that do not need OpenFOAM."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from make_test_rocket import make_rocket  # noqa: E402
from rocketcfd import post  # noqa: E402
from rocketcfd.config import GeometryConfig, config_for_model, load_config  # noqa: E402
from rocketcfd.geometry import prepare_geometry  # noqa: E402
from rocketcfd.mesh import parse_check_mesh, plan_mesh  # noqa: E402
from rocketcfd.regimes import get_regime  # noqa: E402
from rocketcfd.regimes.base import freestream_velocity  # noqa: E402


@pytest.fixture(scope="module")
def rocket_stl(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("models") / "rocket.stl"
    make_rocket(n_fins=3).export(path)  # mm, nose along +z
    return path


# -- config ------------------------------------------------------------------------------------


def test_config_defaults_and_dedupe(tmp_path, rocket_stl):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(f"model: {rocket_stl.as_posix()}\nsweeps:\n  speed: {{aoa: 0, speeds: [20, 50]}}\n"
                        "  aoa: {speed: 50, angles: [0, 4]}\n")
    cfg = load_config(cfg_file)
    assert cfg.name == "rocket"
    assert cfg.sweeps.conditions() == [(20.0, 0.0), (50.0, 0.0), (50.0, 4.0)]


def test_config_rejects_unknown_keys(tmp_path, rocket_stl):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(f"model: {rocket_stl.as_posix()}\nmesh:\n  presett: fine\n")
    with pytest.raises(ValueError, match="presett"):
        load_config(cfg_file)


def test_sweep_can_be_disabled(rocket_stl):
    cfg = config_for_model(rocket_stl)
    cfg.sweeps.aoa = None
    assert all(a == 0 for _, a in cfg.sweeps.conditions())


# -- geometry ----------------------------------------------------------------------------------


def test_geometry_normalisation(rocket_stl):
    geo = prepare_geometry(rocket_stl, GeometryConfig())
    assert geo.source_units == "mm"
    assert geo.length == pytest.approx(0.605, rel=1e-3)
    assert geo.ref_diameter == pytest.approx(0.0416, rel=0.01)
    assert geo.watertight
    # Nose tip at the origin, rocket along +x.
    assert geo.mesh.bounds[0, 0] == pytest.approx(0.0, abs=1e-9)
    tip = geo.mesh.vertices[np.argmin(geo.mesh.vertices[:, 0])]
    assert np.hypot(tip[1], tip[2]) < 1e-3


def test_geometry_nose_detection_reversed(tmp_path):
    m = make_rocket()
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))  # nose now along -y
    path = tmp_path / "r.stl"
    m.export(path)
    geo = prepare_geometry(path, GeometryConfig(units="mm"))
    x = geo.mesh.vertices[:, 0]
    # The slim nose end must be at x = 0: near the front, the radius is small.
    front = geo.mesh.vertices[x < 0.02 * geo.length]
    assert np.hypot(front[:, 1], front[:, 2]).max() < 0.01


def test_step_input(tmp_path):
    gmsh = pytest.importorskip("gmsh")
    path = tmp_path / "rocket.step"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        body = occ.addCylinder(0, 0, 0, 0, 0, 500, 20)          # millimetres, axis +z
        nose = occ.addCone(0, 0, 500, 0, 0, 100, 20, 0)
        fin = occ.addBox(-1.5, 15, 0, 3, 45, 80)
        occ.fuse([(3, body)], [(3, nose), (3, fin)])
        occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    geo = prepare_geometry(path, GeometryConfig())
    assert geo.source_units == "m"
    assert geo.length == pytest.approx(0.6, rel=1e-3)
    assert geo.ref_diameter == pytest.approx(0.04, rel=0.03)
    assert geo.watertight


def test_geometry_ref_diameter_override(rocket_stl):
    geo = prepare_geometry(rocket_stl, GeometryConfig(ref_diameter=0.05))
    assert geo.ref_area == pytest.approx(math.pi * 0.05**2 / 4)


# -- mesh plan -----------------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["coarse", "medium", "fine"])
def test_mesh_plan_resolves_body(rocket_stl, preset):
    cfg = config_for_model(rocket_stl)
    cfg.mesh.preset = preset
    geo = prepare_geometry(rocket_stl, cfg.geometry)
    plan = plan_mesh(geo, cfg)
    target = geo.ref_diameter / cfg.mesh.resolved()["cells_per_diameter"]
    assert plan.surface_size <= target
    seed = np.array(plan.location_in_mesh)
    assert np.all(seed > plan.lo) and np.all(seed < plan.hi)
    assert seed[0] < 0 or np.hypot(seed[1], seed[2]) > geo.max_radius  # outside the rocket


def test_parse_check_mesh():
    text = "    cells:            141684\n    points: 176310\n    Mesh non-orthogonality Max: 44.6 average: 7\n" \
           "    Max skewness = 1.39 OK.\n\nMesh OK.\n"
    stats = parse_check_mesh(text)
    assert stats["cells"] == 141684 and stats["ok"] and stats["max_skewness"] == pytest.approx(1.39)


# -- regimes -----------------------------------------------------------------------------------


def test_freestream_velocity_pitch_up():
    ux, uy, uz = freestream_velocity(50, 4)
    assert ux == pytest.approx(50 * math.cos(math.radians(4)))
    assert uy > 0 and uz == 0


def test_incompressible_rejects_high_mach(rocket_stl):
    cfg = config_for_model(rocket_stl)
    cfg.sweeps.speed.speeds = [250]
    with pytest.raises(ValueError, match="Mach"):
        get_regime("incompressible").check_conditions(cfg, 343)


# -- post-processing ---------------------------------------------------------------------------


FORCE_DAT = """\
# Force
# CofR            : (0 0 0)
#
# Time            \ttotal_x total_y total_z\tpressure_x pressure_y pressure_z\tviscous_x viscous_y viscous_z
1  1.0 0.0 0.0  0.6 0.0 0.0  0.4 0.0 0.0
2  2.0 3.0 0.0  1.2 3.0 0.0  0.8 0.0 0.0
"""
MOMENT_DAT = FORCE_DAT.replace("1.0 0.0 0.0  0.6", "0.0 0.0 0.1  0.0").replace(
    "2  2.0 3.0 0.0  1.2 3.0 0.0  0.8 0.0 0.0", "2  0.0 0.0 1.5  0.0 0.0 1.5  0.0 0.0 0.0")


def test_case_result_coefficients(tmp_path, rocket_stl):
    case = tmp_path / "case"
    fdir = case / "postProcessing" / "forces" / "0"
    fdir.mkdir(parents=True)
    (fdir / "force.dat").write_text(FORCE_DAT)
    (fdir / "moment.dat").write_text(MOMENT_DAT)
    cfg = config_for_model(rocket_stl)
    cfg.solver.average_last = 1
    geo = prepare_geometry(rocket_stl, cfg.geometry)
    r = post.case_result(case, geo, cfg, get_regime("incompressible"), 40.0, 0.0)
    qa = 0.5 * 1.225 * 40**2 * geo.ref_area
    assert r["CA"] == pytest.approx(2.0 / qa)
    assert r["CA_pressure"] + r["CA_viscous"] == pytest.approx(r["CA"])
    assert r["CN"] == pytest.approx(3.0 / qa)
    assert r["Xcp_m"] == pytest.approx(0.5)  # Mz / Fy


def test_read_vtk_polydata(tmp_path):
    vtk = tmp_path / "s.vtk"
    vtk.write_text("# vtk DataFile Version 2.0\ntime='5'\nASCII\nDATASET POLYDATA\nFIELD FieldData 1\n"
                   "TimeValue 1 1 float\n5\nPOINTS 4 float\n0 0 0 1 0 0 1 1 0 0 1 0\n"
                   "POLYGONS 1 5\n4 0 1 2 3\nPOINT_DATA 4\nFIELD FieldData 2\np 1 4 float\n1 2 3 4\n"
                   "U 3 4 float\n1 0 0 1 0 0 1 0 0 1 0 0\n")
    pts, polys, fields = post.read_vtk_polydata(vtk)
    assert pts.shape == (4, 3)
    assert [list(p) for p in polys] == [[0, 1, 2, 3]]
    assert list(fields["p"]) == [1, 2, 3, 4] and fields["U"].shape == (4, 3)
