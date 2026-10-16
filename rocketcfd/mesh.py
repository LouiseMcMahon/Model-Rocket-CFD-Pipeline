"""Background mesh + snappyHexMesh setup around the normalised rocket.

One mesh is built per model and reused by every speed/AoA case: angle of attack is applied
by turning the free-stream velocity, not the geometry.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .foam import header, vec
from .geometry import RocketGeometry
from .runner import write_text


@dataclass
class RefinementBox:
    name: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    level: int


@dataclass
class MeshPlan:
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    cells: tuple[int, int, int]
    base_size: float
    surface_level: int
    surface_size: float
    layers: int
    location_in_mesh: tuple[float, float, float]
    boxes: list[RefinementBox] = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        d["background_cells"] = int(np.prod(self.cells))
        return d


def plan_mesh(geo: RocketGeometry, cfg: Config) -> MeshPlan:
    p = cfg.mesh.resolved()
    L, d, R = geo.length, geo.ref_diameter, geo.max_radius

    h0 = L / p["cells_per_length"]
    level = max(1, math.ceil(math.log2(h0 / (d / p["cells_per_diameter"]))))
    hs = h0 / 2**level

    lateral = max(cfg.mesh.lateral * L, 12 * R)
    nx = math.ceil((cfg.mesh.upstream + 1 + cfg.mesh.downstream) * L / h0)
    nyz = 2 * math.ceil(lateral / h0)
    x0 = -cfg.mesh.upstream * L
    lo = (x0, -nyz / 2 * h0, -nyz / 2 * h0)
    hi = (x0 + nx * h0, nyz / 2 * h0, nyz / 2 * h0)

    boxes = [
        RefinementBox("outer", (-0.3 * L, -4 * R, -4 * R), (2.5 * L, 4 * R, 4 * R), max(1, level - 4)),
        RefinementBox("near", (-0.1 * L, -2 * R, -2 * R), (1.6 * L, 2 * R, 2 * R), max(1, level - 3)),
        RefinementBox("tight", (-0.03 * L, -1.25 * R, -1.25 * R), (1.15 * L, 1.25 * R, 1.25 * R),
                      max(1, level - 2)),
    ]
    # Fluid seed point: upstream, off-axis, and not on a cell face.
    seed = (x0 + 0.37 * h0 + 0.5 * cfg.mesh.upstream * L, 0.413 * lateral + 0.0123 * h0,
            0.287 * lateral + 0.0171 * h0)
    return MeshPlan(lo=lo, hi=hi, cells=(nx, nyz, nyz), base_size=h0, surface_level=level,
                    surface_size=hs, layers=int(p["layers"]), location_in_mesh=seed, boxes=boxes)


def write_mesh_case(case: Path, geo: RocketGeometry, plan: MeshPlan, cfg: Config, regime) -> None:
    (case / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
    geo.mesh.export(case / "constant" / "triSurface" / "rocket.stl")

    write_text(case / "system" / "controlDict", header("controlDict") + f"""\
application     {regime.application};
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
writeFormat     binary;
writePrecision  10;
timeFormat      general;
runTimeModifiable false;
""")
    write_text(case / "system" / "fvSchemes", regime.fv_schemes())
    write_text(case / "system" / "fvSolution", regime.fv_solution(cfg))
    write_text(case / "system" / "decomposeParDict", decompose_dict(cfg.solver.nprocs))
    write_text(case / "system" / "blockMeshDict", _block_mesh_dict(plan))
    write_text(case / "system" / "surfaceFeatureExtractDict", header("surfaceFeatureExtractDict") + """\
rocket.stl
{
    extractionMethod    extractFromSurface;
    includedAngle       150;
    subsetFeatures
    {
        nonManifoldEdges    no;
        openEdges           yes;
    }
    writeObj            no;
}
""")
    write_text(case / "system" / "snappyHexMeshDict", _snappy_dict(plan))


def mesh_commands(nprocs: int) -> list[tuple[str, str]]:
    cmds = [("blockMesh", "blockMesh"), ("surfaceFeatureExtract", "surfaceFeatureExtract")]
    if nprocs > 1:
        cmds += [
            ("decomposePar.mesh", "decomposePar -force"),
            ("snappyHexMesh", f"mpirun -np {nprocs} snappyHexMesh -overwrite -parallel"),
            ("reconstructParMesh", "reconstructParMesh -constant"),
            ("cleanup", "rm -rf processor*"),
        ]
    else:
        cmds.append(("snappyHexMesh", "snappyHexMesh -overwrite"))
    cmds.append(("checkMesh", "checkMesh"))
    return cmds


def parse_check_mesh(log_text: str) -> dict:
    stats: dict = {}
    for key, pat in {"cells": r"^\s*cells:\s+(\d+)", "faces": r"^\s*faces:\s+(\d+)",
                     "points": r"^\s*points:\s+(\d+)",
                     "max_non_orthogonality": r"Mesh non-orthogonality Max:\s+([\d.eE+-]+)",
                     "max_skewness": r"Max skewness = ([\d.eE+-]+)"}.items():
        m = re.search(pat, log_text, re.M)
        if m:
            stats[key] = float(m.group(1)) if "." in m.group(1) or "e" in m.group(1) else int(m.group(1))
    failed = re.search(r"Failed (\d+) mesh checks", log_text)
    stats["failed_checks"] = int(failed.group(1)) if failed else 0
    stats["ok"] = "Mesh OK" in log_text
    return stats


def decompose_dict(nprocs: int) -> str:
    return header("decomposeParDict") + f"numberOfSubdomains {nprocs};\nmethod scotch;\n"


def _block_mesh_dict(plan: MeshPlan) -> str:
    (x0, y0, z0), (x1, y1, z1) = plan.lo, plan.hi
    pts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
           (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    verts = "\n".join(f"    {vec(p)}" for p in pts)
    nx, ny, nz = plan.cells
    return header("blockMeshDict") + f"""\
scale 1;

vertices
(
{verts}
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

boundary
(
    farfield
    {{
        type patch;
        faces
        (
            (0 4 7 3)
            (1 2 6 5)
            (0 1 5 4)
            (3 7 6 2)
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);
"""


def _snappy_dict(plan: MeshPlan) -> str:
    lvl = plan.surface_level
    boxes_geo = "\n".join(
        f"    {b.name}\n    {{\n        type searchableBox;\n        min {vec(b.lo)};\n        max {vec(b.hi)};\n    }}"
        for b in plan.boxes)
    boxes_ref = "\n".join(
        f"        {b.name}\n        {{\n            mode inside;\n            levels ((1e15 {b.level}));\n        }}"
        for b in plan.boxes)
    add_layers = "true" if plan.layers > 0 else "false"
    return header("snappyHexMeshDict") + f"""\
castellatedMesh true;
snap            true;
addLayers       {add_layers};

geometry
{{
    rocket.stl
    {{
        type triSurfaceMesh;
        name rocket;
    }}
{boxes_geo}
}}

castellatedMeshControls
{{
    maxLocalCells       2000000;
    maxGlobalCells      30000000;
    minRefinementCells  10;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 3;

    features
    (
        {{
            file "rocket.eMesh";
            level {lvl};
        }}
    );

    refinementSurfaces
    {{
        rocket
        {{
            level ({lvl} {lvl + 1});
            patchInfo {{ type wall; }}
        }}
    }}

    resolveFeatureAngle 30;

    refinementRegions
    {{
{boxes_ref}
    }}

    locationInMesh {vec(plan.location_in_mesh)};
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      100;
    nRelaxIter      5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes       true;
    layers
    {{
        rocket {{ nSurfaceLayers {max(plan.layers, 1)}; }}
    }}
    expansionRatio      1.25;
    finalLayerThickness 0.4;
    minThickness        0.1;
    nGrow               0;
    featureAngle        120;
    slipFeatureAngle    30;
    nRelaxIter          3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals      3;
    nSmoothThickness    10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle  90;
    nBufferCellsNoExtrude 0;
    nLayerIter          50;
}}

meshQualityControls
{{
    maxNonOrtho         65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave          80;
    minVol              1e-13;
    minTetQuality       1e-15;
    minArea             -1;
    minTwist            0.02;
    minDeterminant      0.001;
    minFaceWeight       0.05;
    minVolRatio         0.01;
    minTriangleTwist    -1;
    nSmoothScale    4;
    errorReduction  0.75;
    relaxed
    {{
        maxNonOrtho 75;
    }}
}}

mergeTolerance 1e-6;
"""
