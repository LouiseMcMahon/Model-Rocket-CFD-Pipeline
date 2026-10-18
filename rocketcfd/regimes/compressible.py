"""Steady compressible RANS (rhoSimpleFoam, k-omega SST, perfect gas with Sutherland viscosity).

Covers subsonic through low supersonic flight (about Mach 0.3 to 1.5). Settings follow OpenFOAM's
rhoSimpleFoam aerofoilNACA0012 tutorial (transonic), which is the most robust starting point
for steady density-based-ish convergence. Shocks are captured with first-order upwinding on
pressure and a second-order linearUpwind scheme elsewhere.

Not yet included: mesh refinement that follows shocks. The shared mesh is the same one the
incompressible regime uses, so shock positions are smeared over a few cells.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..foam import header, num, vec
from ..geometry import RocketGeometry
from ..mesh import decompose_dict
from ..runner import write_text
from .base import Regime, freestream_velocity


class CompressibleRegime(Regime):
    name = "compressible"
    application = "rhoSimpleFoam"
    max_mach = 1.5

    def check_conditions(self, cfg: Config, speed_of_sound: float) -> list[str]:
        warnings = []
        for speed, _ in cfg.sweeps.conditions():
            mach = speed / speed_of_sound
            if mach > self.max_mach:
                raise ValueError(f"{speed} m/s is Mach {mach:.2f}, above the steady compressible solver's "
                                 f"limit of Mach {self.max_mach}.")
            if 0.85 <= mach < 1.0:
                warnings.append(f"{speed} m/s is Mach {mach:.2f} (transonic): shock position and drag rise "
                                "are sensitive to the mesh; use a fine mesh and check convergence.")
            elif mach >= 1.0:
                warnings.append(f"{speed} m/s is Mach {mach:.2f} (supersonic): bow shock is captured on the "
                                "standard mesh only; coefficients are indicative until shock-adapted "
                                "meshing is added.")
        return warnings

    def fv_schemes(self, robust: bool = False) -> str:
        """``robust`` swaps the second-order convection schemes for first-order upwind.

        Steady supersonic flow with a bow shock does not converge with linearUpwind on a
        standard mesh (the shock oscillates), so supersonic cases use upwind throughout.
        """
        scheme = "bounded Gauss upwind" if robust else "bounded Gauss linearUpwind limited"
        return header("fvSchemes") + """\
ddtSchemes      { default steadyState; }

gradSchemes
{
    default         Gauss linear;
    limited         cellLimited Gauss linear 1;
    grad(U)         $limited;
    grad(k)         $limited;
    grad(omega)     $limited;
}

divSchemes
{
    default         none;

    div(phi,U)      @SCHEME@;

    energy          @SCHEME@;
    div(phi,e)      $energy;
    div(phi,h)      $energy;
    div(phi,K)      $energy;
    div(phi,Ekp)    $energy;

    turbulence      bounded Gauss upwind;
    div(phi,k)      $turbulence;
    div(phi,omega)  $turbulence;

    div(phid,p)     Gauss upwind;
    div((phi|interpolate(rho)),p)  bounded Gauss upwind;

    div(((rho*nuEff)*dev2(T(grad(U)))))    Gauss linear;
}

laplacianSchemes      { default Gauss linear corrected; }
interpolationSchemes  { default linear; }
snGradSchemes         { default corrected; }
wallDist              { method meshWave; }
""".replace("@SCHEME@", scheme)

    def fv_solution(self, cfg: Config) -> str:
        return header("fvSolution") + """\
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.01;
    }
    "(U|k|omega|e)"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-8;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    pMinFactor      0.1;
    pMaxFactor      2;
    residualControl
    {
        p               1e-4;
        U               1e-5;
        e               1e-5;
        "(k|omega)"     1e-5;
    }
}

relaxationFactors
{
    fields
    {
        p               0.7;
        rho             0.01;
    }
    equations
    {
        U               0.3;
        e               0.7;
        "(k|omega)"     0.7;
    }
}
"""

    def write_case(self, case: Path, geo: RocketGeometry, cfg: Config, speed: float, aoa: float) -> None:
        air = cfg.air
        u = freestream_velocity(speed, aoa)
        supersonic = "true" if speed > air.speed_of_sound else "false"
        nu = air.mu_sutherland / air.density_ideal_gas
        k = 1.5 * (air.turbulence_intensity * speed) ** 2
        omega = k / (nu * air.viscosity_ratio)

        def scalar(name, dims, value, far, wall, cls="volScalarField"):
            write_text(case / "0" / name, header(name, cls) + f"""\
dimensions      {dims};
internalField   uniform {num(value)};
boundaryField
{{
    farfield
    {{
{far}
    }}
    rocket
    {{
{wall}
    }}
}}
""")

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
        scalar("p", "[1 -1 -2 0 0 0 0]", air.pressure,
               f"        type            freestreamPressure;\n        freestreamValue uniform {num(air.pressure)};\n"
               f"        supersonic      {supersonic};",
               "        type            zeroGradient;")
        scalar("T", "[0 0 0 1 0 0 0]", air.temperature,
               f"        type            inletOutlet;\n        inletValue      uniform {num(air.temperature)};\n"
               f"        value           uniform {num(air.temperature)};",
               "        type            zeroGradient;")  # adiabatic wall
        for name, dims, value, wall in [("k", "[0 2 -2 0 0 0 0]", k, "kqRWallFunction"),
                                        ("omega", "[0 0 -1 0 0 0 0]", omega, "omegaWallFunction")]:
            scalar(name, dims, value,
                   f"        type            inletOutlet;\n        inletValue      uniform {num(value)};\n"
                   f"        value           uniform {num(value)};",
                   f"        type            {wall};\n        value           uniform {num(value)};")
        scalar("nut", "[0 2 -1 0 0 0 0]", 0,
               "        type            calculated;\n        value           uniform 0;",
               "        type            nutUSpaldingWallFunction;\n        value           uniform 0;")
        scalar("alphat", "[1 -1 -1 0 0 0 0]", 0,
               "        type            calculated;\n        value           uniform 0;",
               "        type            compressible::alphatWallFunction;\n        value           uniform 0;")

        write_text(case / "constant" / "thermophysicalProperties", header("thermophysicalProperties") + """\
thermoType
{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       sutherland;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;
}

mixture
{
    specie
    {
        molWeight   28.96;
    }
    thermodynamics
    {
        Cp          1005;
        Hf          0;
    }
    transport
    {
        As          1.458e-06;
        Ts          110.4;
    }
}
""")
        write_text(case / "constant" / "turbulenceProperties", header("turbulenceProperties") + """\
simulationType  RAS;
RAS
{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
""")
        # Temperature limiter: stops the impulsive start from driving T negative in the first iterations.
        write_text(case / "system" / "fvOptions", header("fvOptions") + f"""\
limitT
{{
    type            limitTemperature;
    min             {num(0.5 * air.temperature)};
    max             {num(3.0 * air.temperature)};
    selectionMode   all;
}}
""")
        write_text(case / "system" / "fvSchemes", self.fv_schemes(robust=speed >= 0.95 * air.speed_of_sound))
        write_text(case / "system" / "fvSolution", self.fv_solution(cfg))
        write_text(case / "system" / "decomposeParDict", decompose_dict(cfg.solver.nprocs))
        write_text(case / "system" / "controlDict", self.control_dict(
            cfg, "        rho             rho;", "U p e k omega", sample_fields="p U T"))

    def solve_commands(self, cfg: Config) -> list[tuple[str, str]]:
        n = cfg.solver.nprocs
        if n <= 1:
            return [("rhoSimpleFoam", "rhoSimpleFoam")]
        mpi = f"mpirun -np {n}"
        return [
            ("decomposePar", "decomposePar -force"),
            ("rhoSimpleFoam", f"{mpi} rhoSimpleFoam -parallel"),
            ("reconstructPar", "reconstructPar -latestTime"),
            ("cleanup", "rm -rf processor*"),
        ]

    def pressure_coefficient(self, p, speed: float, cfg: Config):
        q = 0.5 * cfg.air.density_ideal_gas * speed**2
        return (p - cfg.air.pressure) / q
