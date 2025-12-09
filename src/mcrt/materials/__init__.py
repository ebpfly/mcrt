"""Materials and optical constants module.

Provides access to the refractiveindex.info database and custom material handling.
"""

from mcrt.materials.database import (
    MaterialDatabase,
    MaterialInfo,
    OpticalConstants,
    COMMON_MINERALS,
    get_common_material,
)
from mcrt.materials.custom import (
    CustomMaterial,
    CustomMaterialLibrary,
    acrylic,
    air,
)

__all__ = [
    "MaterialDatabase",
    "MaterialInfo",
    "OpticalConstants",
    "COMMON_MINERALS",
    "get_common_material",
    "CustomMaterial",
    "CustomMaterialLibrary",
    "acrylic",
    "air",
]
