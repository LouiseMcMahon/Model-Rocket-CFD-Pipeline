from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path

from ..config import Config
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

    @abstractmethod
    def solve_commands(self, cfg: Config) -> list[tuple[str, str]]: ...

    @abstractmethod
    def pressure_coefficient(self, p, speed: float, cfg: Config):
        """Convert the solver's pressure field to Cp."""


def freestream_velocity(speed: float, aoa_deg: float) -> tuple[float, float, float]:
    """Relative wind for a nose-up angle of attack: air approaches from below the nose."""
    a = math.radians(aoa_deg)
    return (speed * math.cos(a), speed * math.sin(a), 0.0)
