"""Steady incompressible RANS (simpleFoam, k-omega SST). Valid up to about Mach 0.3."""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..foam import header, num, vec
from ..geometry import RocketGeometry
from ..mesh import decompose_dict
from ..runner import write_text
from .base import Regime, freestream_velocity


class IncompressibleRegime(Regime):
    name = "incompressible"
    application = "simpleFoam"
    max_mach = 0.6

    def fv_schemes(self) -> str:
        return header("fvSchemes") + """\
ddtSchemes      { default steadyState; }

gradSchemes
{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
    grad(k)         cellLimited Gauss linear 1;
    grad(omega)     cellLimited Gauss linear 1;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwindV grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
    div(div(phi,U)) Gauss linear;
}

laplacianSchemes      { default Gauss linear limited corrected 0.33; }
interpolationSchemes  { default linear; }
snGradSchemes         { default limited corrected 0.33; }
wallDist              { method meshWave; }
"""

    def fv_solution(self, cfg: Config) -> str:
        return header("fvSolution") + """\
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.05;
    }
    Phi
    {
        $p;
    }
    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        nSweeps         2;
        tolerance       1e-8;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent      yes;
    residualControl
    {
        p               1e-5;
        U               1e-6;
        "(k|omega)"     1e-6;
    }
}

potentialFlow
{
    nNonOrthogonalCorrectors 5;
}

relaxationFactors
{
    equations
    {
        U               0.8;
        ".*"            0.6;
    }
}
"""

    def write_case(self, case: Path, geo: RocketGeometry, cfg: Config, speed: float, aoa: float) -> None:
        u = freestream_velocity(speed, aoa)
        k = 1.5 * (cfg.air.turbulence_intensity * speed) ** 2
        omega = k / (cfg.air.nu * cfg.air.viscosity_ratio)

        write_text(case / "0" / "U", header("U", "volVectorField") + f"""\
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform {vec(u)};
boundaryField
{{
    farfield
    {{
        type            freestreamVelocity;
        freestreamValue uniform {vec(u)};
    }}
    rocket
    {{
        type            noSlip;
    }}
}}
""")
        write_text(case / "0" / "p", header("p", "volScalarField") + """\
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    farfield
    {
        type            freestreamPressure;
        freestreamValue uniform 0;
    }
    rocket
    {
        type            zeroGradient;
    }
}
""")
        for name, dims, value, wall in [
            ("k", "[0 2 -2 0 0 0 0]", k, "kqRWallFunction"),
            ("omega", "[0 0 -1 0 0 0 0]", omega, "omegaWallFunction"),
        ]:
            write_text(case / "0" / name, header(name, "volScalarField") + f"""\
dimensions      {dims};
internalField   uniform {num(value)};
boundaryField
{{
    farfield
    {{
        type            inletOutlet;
        inletValue      uniform {num(value)};
        value           uniform {num(value)};
    }}
    rocket
    {{
        type            {wall};
        value           uniform {num(value)};
    }}
}}
""")
        write_text(case / "0" / "nut", header("nut", "volScalarField") + """\
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    farfield
    {
        type            calculated;
        value           uniform 0;
    }
    rocket
    {
        type            nutUSpaldingWallFunction;
        value           uniform 0;
    }
}
""")

        write_text(case / "constant" / "transportProperties", header("transportProperties") +
                   f"transportModel  Newtonian;\nnu              {num(cfg.air.nu)};\n")
        write_text(case / "constant" / "turbulenceProperties", header("turbulenceProperties") + """\
simulationType  RAS;
RAS
{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
""")

        write_text(case / "system" / "fvSchemes", self.fv_schemes())
        write_text(case / "system" / "fvSolution", self.fv_solution(cfg))
        write_text(case / "system" / "decomposeParDict", decompose_dict(cfg.solver.nprocs))
        write_text(case / "system" / "controlDict", self.control_dict(
            cfg, f"        rho             rhoInf;\n        rhoInf          {num(cfg.air.rho)};",
            "U p k omega"))

    def solve_commands(self, cfg: Config) -> list[tuple[str, str]]:
        n = cfg.solver.nprocs
        if n <= 1:
            cmds = [("potentialFoam", "potentialFoam -writephi")] if cfg.solver.potential_init else []
            return cmds + [("simpleFoam", "simpleFoam")]
        mpi = f"mpirun -np {n}"
        cmds = [("decomposePar", "decomposePar -force")]
        if cfg.solver.potential_init:
            cmds.append(("potentialFoam", f"{mpi} potentialFoam -parallel -writephi"))
        cmds += [
            ("simpleFoam", f"{mpi} simpleFoam -parallel"),
            ("reconstructPar", "reconstructPar -latestTime"),
            ("cleanup", "rm -rf processor*"),
        ]
        return cmds

    def pressure_coefficient(self, p, speed: float, cfg: Config):
        # simpleFoam's p is kinematic (p/rho) and the far-field value is 0.
        return p / (0.5 * speed**2)
