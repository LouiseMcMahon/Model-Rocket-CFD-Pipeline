from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path

from ..config import Config
from ..foam import header
from ..geometry import RocketGeometry


class Regime(ABC):
    name: str
    application: str        # solver binary, e.g. simpleFoam
    max_mach: float         # above this, the regime refuses to run

    def check_conditions(self, cfg: Config, speed_of_sound: float) -> list[str]:
        """Return warnings; raise ValueError for conditions this regime cannot handle."""
        warnings = []
        for speed, _ in cfg.sweeps.conditions():
            mach = speed / speed_of_sound
            if mach > self.max_mach:
                raise ValueError(f"{speed} m/s is Mach {mach:.2f}, above the {self.name} limit "
                                 f"of Mach {self.max_mach}. Use a compressible regime.")
            if mach > 0.3:
                warnings.append(f"{speed} m/s is Mach {mach:.2f}; compressibility starts to matter "
                                "above Mach 0.3 (drag will be underpredicted).")
        return warnings

    @abstractmethod
    def fv_schemes(self) -> str: ...

    @abstractmethod
    def fv_solution(self, cfg: Config) -> str: ...

    @abstractmethod
    def write_case(self, case: Path, geo: RocketGeometry, cfg: Config, speed: float, aoa: float) -> None:
        """Write 0/, constant/ (except polyMesh) and system/ for one flight condition."""

    def control_dict(self, cfg: Config, force_rho: str, residual_fields: str, sample_fields: str = "p U") -> str:
        """controlDict with force/moment logging, residuals and (optionally) pitch-plane sampling.

        ``force_rho`` is the density lines for the forces function object: a fixed ``rhoInf`` for
        incompressible runs, or the solver's own density field for compressible ones.
        """
        n = cfg.solver.iterations
        sample = ""
        if cfg.visualize:
            sample = f"""
    slices
    {{
        type            surfaces;
        libs            (sampling);
        writeControl    writeTime;
        surfaceFormat   vtk;
        formatOptions   {{ vtk {{ legacy true; format ascii; }} }}
        fields          ({sample_fields});
        interpolationScheme cellPoint;
        surfaces
        {{
            pitchPlane
            {{
                type        cuttingPlane;
                point       (0 0 1e-7);
                normal      (0 0 1);
                interpolate true;
            }}
            rocketSurface
            {{
                type        patch;
                patches     (rocket);
                interpolate true;
            }}
        }}
    }}
"""
        return header("controlDict") + f"""\
application     {self.application};
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {n};
deltaT          1;
writeControl    timeStep;
writeInterval   {n};
purgeWrite      0;
writeFormat     binary;
writePrecision  10;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    forces
    {{
        type            forces;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        patches         (rocket);
{force_rho}
        CofR            (0 0 0);
        log             false;
    }}

    residuals
    {{
        type            solverInfo;
        libs            (utilityFunctionObjects);
        fields          ({residual_fields});
        writeControl    timeStep;
        writeInterval   1;
    }}
{sample}}}
"""

    @abstractmethod
    def solve_commands(self, cfg: Config) -> list[tuple[str, str]]: ...

    @abstractmethod
    def pressure_coefficient(self, p, speed: float, cfg: Config):
        """Convert the solver's pressure field to Cp."""


def freestream_velocity(speed: float, aoa_deg: float) -> tuple[float, float, float]:
    """Relative wind for a nose-up angle of attack: air approaches from below the nose."""
    a = math.radians(aoa_deg)
    return (speed * math.cos(a), speed * math.sin(a), 0.0)
