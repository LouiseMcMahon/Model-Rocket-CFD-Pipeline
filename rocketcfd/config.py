"""Run configuration, loaded from a YAML file.

Every field has a default so a config can be as small as ``model: my_rocket.stl``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SPEED_OF_SOUND = 343.0  # m/s, sea level standard atmosphere


@dataclass
class GeometryConfig:
    units: str = "auto"            # m | cm | mm | in | auto (STEP files are always read in metres)
    nose_direction: str = "auto"   # auto | +x | -x | +y | -y | +z | -z  (in the input file's frame)
    roll_deg: float = 0.0          # roll about the rocket axis applied after alignment
    ref_diameter: float | None = None  # metres; auto-detected from the body tube if omitted
    union_bodies: bool = True      # boolean-union multiple overlapping solids into one
    cg_from_nose: float | None = None  # metres; if given, stability margin is reported


@dataclass
class AirConfig:
    rho: float = 1.225             # kg/m^3
    nu: float = 1.5e-5             # kinematic viscosity, m^2/s
    turbulence_intensity: float = 0.005
    viscosity_ratio: float = 5.0   # nut/nu at the far field


MESH_PRESETS: dict[str, dict[str, float]] = {
    # cells_per_length: background cells along the rocket length
    # cells_per_diameter: target surface cell count across the body diameter
    "coarse": {"cells_per_length": 6, "cells_per_diameter": 10, "layers": 0},
    "medium": {"cells_per_length": 8, "cells_per_diameter": 16, "layers": 3},
    "fine": {"cells_per_length": 10, "cells_per_diameter": 24, "layers": 5},
}


@dataclass
class MeshConfig:
    preset: str = "medium"
    cells_per_length: float | None = None    # override preset
    cells_per_diameter: float | None = None  # override preset
    layers: int | None = None                # override preset (0 disables boundary layers)
    upstream: float = 1.5     # domain extents, in rocket lengths
    downstream: float = 4.0
    lateral: float = 1.0

    def resolved(self) -> dict[str, float]:
        if self.preset not in MESH_PRESETS:
            raise ValueError(f"Unknown mesh preset '{self.preset}', choose from {list(MESH_PRESETS)}")
        out = dict(MESH_PRESETS[self.preset])
        for key in ("cells_per_length", "cells_per_diameter", "layers"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


@dataclass
class SolverConfig:
    iterations: int = 1500
    average_last: int = 300        # iterations averaged for reported coefficients
    nprocs: int = 8
    potential_init: bool = True    # initialise the flow with potentialFoam


@dataclass
class SpeedSweep:
    aoa: float = 0.0
    speeds: list[float] = field(default_factory=lambda: [10.0, 20.0, 35.0, 50.0, 75.0, 100.0])


@dataclass
class AoaSweep:
    speed: float = 50.0
    angles: list[float] = field(default_factory=lambda: [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])


@dataclass
class SweepConfig:
    speed: SpeedSweep | None = field(default_factory=SpeedSweep)
    aoa: AoaSweep | None = field(default_factory=AoaSweep)

    def conditions(self) -> list[tuple[float, float]]:
        """Unique (speed, aoa) pairs to simulate, in run order."""
        pairs: list[tuple[float, float]] = []
        if self.speed:
            pairs += [(float(v), float(self.speed.aoa)) for v in self.speed.speeds]
        if self.aoa:
            pairs += [(float(self.aoa.speed), float(a)) for a in self.aoa.angles]
        seen: set[tuple[float, float]] = set()
        return [p for p in pairs if not (p in seen or seen.add(p))]


@dataclass
class OpenFOAMConfig:
    runner: str = "auto"           # auto | wsl | native
    wsl_distro: str = "Ubuntu"
    bashrc: str = "/usr/lib/openfoam/openfoam2412/etc/bashrc"
    work_dir: str = "~/rocketcfd_runs"   # Linux-side directory holding the OpenFOAM cases


@dataclass
class Config:
    model: Path
    name: str = ""
    regime: str = "incompressible"
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    air: AirConfig = field(default_factory=AirConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    sweeps: SweepConfig = field(default_factory=SweepConfig)
    visualize: bool = True
    openfoam: OpenFOAMConfig = field(default_factory=OpenFOAMConfig)
    output_dir: Path = Path("runs")

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.model.stem
        self.name = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.name)


def _build(cls: type, data: dict[str, Any] | None, where: str) -> Any:
    """Recursively build a dataclass from a dict, rejecting unknown keys."""
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"'{where}' must be a mapping")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"Unknown key(s) in '{where}': {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        sub = _NESTED.get((cls, key))
        kwargs[key] = _build(sub, value, f"{where}.{key}") if sub else value
    return cls(**kwargs)


_NESTED: dict[tuple[type, str], type] = {
    (Config, "geometry"): GeometryConfig,
    (Config, "air"): AirConfig,
    (Config, "mesh"): MeshConfig,
    (Config, "solver"): SolverConfig,
    (Config, "sweeps"): SweepConfig,
    (Config, "openfoam"): OpenFOAMConfig,
    (SweepConfig, "speed"): SpeedSweep,
    (SweepConfig, "aoa"): AoaSweep,
}


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config. Relative paths in it resolve against the config file's folder."""
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.update(overrides or {})
    if "model" not in data:
        raise ValueError(f"{path}: 'model' (path to an STL or STEP file) is required")
    data["model"] = (path.parent / data["model"]).resolve()
    data["output_dir"] = (path.parent / data.get("output_dir", "runs")).resolve()
    return _build(Config, data, "config")


def config_for_model(model: str | Path, **overrides: Any) -> Config:
    """Default config for a model file, used when no YAML is given."""
    cfg = Config(model=Path(model).resolve(), output_dir=Path("runs").resolve())
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg
