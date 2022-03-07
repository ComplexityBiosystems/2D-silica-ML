"""
init file for silicanets.

- version
- some assets
"""
from pathlib import Path

__path__: list

# VERSION
__version__ = "0.1.0"

# ASSETS
__LAMMPS_MINIMIZE_FIRE__ = str(
    Path(__path__[0]) / "assets/lammps_scripts/lammps_minimize_fire.in")
