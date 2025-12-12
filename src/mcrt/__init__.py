"""MCRT: Monte Carlo Radiative Transfer for Particulate Materials.

An interactive simulator for computing reflectance, transmittance, and absorptance
of particulate media using Monte Carlo methods.
"""

__version__ = "0.1.0"

# Convenient imports
from mcrt.fos.wrapper import FOSWrapper
from mcrt.fos.output_parser import FOSResult
from mcrt.materials.database import MaterialDatabase, OpticalConstants, get_common_material
from mcrt.materials.custom import CustomMaterial
from mcrt.thinfilm import ThinFilmStack, ThinFilmResult, Layer, calculate_thin_film

__all__ = [
    "__version__",
    "FOSWrapper",
    "FOSResult",
    "MaterialDatabase",
    "OpticalConstants",
    "CustomMaterial",
    "get_common_material",
    # Thin film
    "ThinFilmStack",
    "ThinFilmResult",
    "Layer",
    "calculate_thin_film",
]
