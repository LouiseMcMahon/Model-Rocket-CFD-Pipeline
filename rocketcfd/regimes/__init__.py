"""Flow regimes: each one knows how to set up and run a solver case on the shared mesh.

To add a regime (e.g. compressible/supersonic), subclass ``Regime`` and register it below.
Meshing, sweeps and force post-processing are shared, so a regime only has to supply its
fields, numerics and solver commands.
"""

from __future__ import annotations

from .base import Regime
from .compressible import CompressibleRegime
from .incompressible import IncompressibleRegime

REGIMES: dict[str, type[Regime]] = {
    "incompressible": IncompressibleRegime,
    "compressible": CompressibleRegime,
}


def get_regime(name: str) -> Regime:
    try:
        return REGIMES[name]()
    except KeyError:
        raise ValueError(f"Unknown regime '{name}', choose from {list(REGIMES)}") from None
