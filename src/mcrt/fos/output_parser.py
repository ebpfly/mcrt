"""Parse FOS output files and arrays."""

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class FOSResult:
    """Results from a FOS simulation.

    Attributes:
        wavelength_um: Wavelength array in micrometers
        reflectance: Total reflectance (specular + diffuse) at each wavelength
        absorptance: Absorptance at each wavelength
        transmittance: Transmittance at each wavelength
        n_photons: Number of photons used per wavelength
        solar_reflectance: Solar-weighted reflectance (if solar spectrum provided)
        solar_absorptance: Solar-weighted absorptance (if solar spectrum provided)
        solar_transmittance: Solar-weighted transmittance (if solar spectrum provided)
        layer_properties: Per-layer optical properties (if available)
    """

    wavelength_um: np.ndarray
    reflectance: np.ndarray
    absorptance: np.ndarray
    transmittance: np.ndarray
    n_photons: int = 0
    solar_reflectance: float | None = None
    solar_absorptance: float | None = None
    solar_transmittance: float | None = None
    layer_properties: dict[int, dict] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure arrays are numpy arrays
        self.wavelength_um = np.asarray(self.wavelength_um)
        self.reflectance = np.asarray(self.reflectance)
        self.absorptance = np.asarray(self.absorptance)
        self.transmittance = np.asarray(self.transmittance)

    @property
    def n_wavelengths(self) -> int:
        """Number of wavelength points."""
        return len(self.wavelength_um)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "wavelength_um": self.wavelength_um.tolist(),
            "reflectance": self.reflectance.tolist(),
            "absorptance": self.absorptance.tolist(),
            "transmittance": self.transmittance.tolist(),
            "n_photons": self.n_photons,
            "solar_reflectance": self.solar_reflectance,
            "solar_absorptance": self.solar_absorptance,
            "solar_transmittance": self.solar_transmittance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FOSResult":
        """Create from dictionary."""
        return cls(
            wavelength_um=np.array(data["wavelength_um"]),
            reflectance=np.array(data["reflectance"]),
            absorptance=np.array(data["absorptance"]),
            transmittance=np.array(data["transmittance"]),
            n_photons=data.get("n_photons", 0),
            solar_reflectance=data.get("solar_reflectance"),
            solar_absorptance=data.get("solar_absorptance"),
            solar_transmittance=data.get("solar_transmittance"),
        )

    def merge(self, other: "FOSResult", weight: float = 0.5) -> "FOSResult":
        """Merge with another result using weighted average.

        Args:
            other: Another FOSResult to merge with
            weight: Weight for this result (other gets 1-weight)

        Returns:
            New FOSResult with merged values
        """
        if not np.allclose(self.wavelength_um, other.wavelength_um):
            raise ValueError("Cannot merge results with different wavelength grids")

        return FOSResult(
            wavelength_um=self.wavelength_um,
            reflectance=weight * self.reflectance + (1 - weight) * other.reflectance,
            absorptance=weight * self.absorptance + (1 - weight) * other.absorptance,
            transmittance=weight * self.transmittance + (1 - weight) * other.transmittance,
            n_photons=self.n_photons + other.n_photons,
        )

    @classmethod
    def accumulate(cls, results: list["FOSResult"]) -> "FOSResult":
        """Accumulate multiple results with equal weighting.

        Args:
            results: List of FOSResult objects to accumulate

        Returns:
            New FOSResult with averaged values and summed photon counts
        """
        if not results:
            raise ValueError("Cannot accumulate empty list of results")

        wavelength_um = results[0].wavelength_um
        n = len(results)

        reflectance = np.mean([r.reflectance for r in results], axis=0)
        absorptance = np.mean([r.absorptance for r in results], axis=0)
        transmittance = np.mean([r.transmittance for r in results], axis=0)
        n_photons = sum(r.n_photons for r in results)

        return cls(
            wavelength_um=wavelength_um,
            reflectance=reflectance,
            absorptance=absorptance,
            transmittance=transmittance,
            n_photons=n_photons,
        )


class FOSOutputParser:
    """Parse FOS output files and arrays."""

    @staticmethod
    def parse_array(output_array: np.ndarray, n_photons: int = 0) -> FOSResult:
        """Parse the output array from FOS main_func.

        Args:
            output_array: Array with columns [wavelength, R, A, T]
            n_photons: Number of photons used (for metadata)

        Returns:
            FOSResult object
        """
        return FOSResult(
            wavelength_um=output_array[:, 0],
            reflectance=output_array[:, 1],
            absorptance=output_array[:, 2],
            transmittance=output_array[:, 3],
            n_photons=n_photons,
        )

    @staticmethod
    def parse_output_file(filepath: str | Path) -> FOSResult:
        """Parse a FOS output text file.

        Args:
            filepath: Path to the output file

        Returns:
            FOSResult object
        """
        filepath = Path(filepath)

        wavelength = []
        reflectance = []
        absorptance = []
        transmittance = []
        solar_r = None
        solar_a = None
        solar_t = None
        n_photons = 0

        with open(filepath) as f:
            lines = f.readlines()

        in_data = False
        for line in lines:
            line = line.strip()

            # Parse solar values from header
            if line.startswith("Solar R:"):
                solar_r = float(line.split(":")[1].strip())
            elif line.startswith("Solar A:"):
                solar_a = float(line.split(":")[1].strip())
            elif line.startswith("Solar T:"):
                solar_t = float(line.split(":")[1].strip())
            elif line.startswith("Photons:"):
                try:
                    n_photons = int(line.split(":")[1].strip())
                except ValueError:
                    pass

            # Check for data header
            if line.startswith("WL\tR\tA\tT"):
                in_data = True
                continue

            # Parse data lines
            if in_data and line and not line.startswith("Input file"):
                parts = line.split("\t")
                if len(parts) >= 4:
                    try:
                        wavelength.append(float(parts[0]))
                        reflectance.append(float(parts[1]))
                        absorptance.append(float(parts[2]))
                        transmittance.append(float(parts[3]))
                    except ValueError:
                        # End of data section
                        in_data = False

        return FOSResult(
            wavelength_um=np.array(wavelength),
            reflectance=np.array(reflectance),
            absorptance=np.array(absorptance),
            transmittance=np.array(transmittance),
            n_photons=n_photons,
            solar_reflectance=solar_r,
            solar_absorptance=solar_a,
            solar_transmittance=solar_t,
        )
