"""Placeholder for transonic/supersonic runs.

Planned approach: rhoSimpleFoam (subsonic/transonic, steady) or rhoCentralFoam (supersonic),
with p/T/U free-stream conditions from altitude, and pressure-based Cp. The mesh needs
extra refinement where shocks form (nose tip, fin leading edges). The rest of the pipeline
(mesh, sweeps, force post-processing) already works for any regime.
"""

from __future__ import annotations

from .base import Regime


class CompressibleRegime(Regime):
    name = "compressible"
    application = "rhoSimpleFoam"
    max_mach = 3.0

    def _todo(self, *_, **__):
        raise NotImplementedError("The compressible regime is not implemented yet; use 'incompressible'.")

    fv_schemes = fv_solution = write_case = solve_commands = pressure_coefficient = _todo
    check_conditions = _todo
