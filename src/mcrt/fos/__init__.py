"""FOS (Fast Optical Spectrum) wrapper module.

Provides Python bindings for the FOS Monte Carlo radiative transfer code.
"""

from mcrt.fos.wrapper import FOSWrapper
from mcrt.fos.input_builder import FOSInputBuilder, SimulationConfig, LayerConfig
from mcrt.fos.output_parser import FOSOutputParser, FOSResult

__all__ = [
    "FOSWrapper",
    "FOSInputBuilder",
    "FOSOutputParser",
    "SimulationConfig",
    "LayerConfig",
    "FOSResult",
]
