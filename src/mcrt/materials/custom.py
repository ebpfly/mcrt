"""User-defined custom materials with n,k data."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from mcrt.materials.database import MaterialInfo, OpticalConstants


@dataclass
class CustomMaterial:
    """A user-defined material with custom optical constants.

    Attributes:
        name: Display name for the material
        wavelength_um: Wavelength array in micrometers
        n: Real part of refractive index
        k: Imaginary part (extinction coefficient)
        description: Optional description
    """

    name: str
    wavelength_um: np.ndarray
    n: np.ndarray
    k: np.ndarray
    description: str = ""

    def __post_init__(self):
        self.wavelength_um = np.asarray(self.wavelength_um)
        self.n = np.asarray(self.n)
        self.k = np.asarray(self.k)

        # Validate arrays have same length
        if not (len(self.wavelength_um) == len(self.n) == len(self.k)):
            raise ValueError(
                f"Arrays must have same length: "
                f"wavelength={len(self.wavelength_um)}, n={len(self.n)}, k={len(self.k)}"
            )

    def to_optical_constants(self) -> OpticalConstants:
        """Convert to OpticalConstants object."""
        info = MaterialInfo(
            material_id=f"custom/{self.name}",
            name=self.name,
            shelf="custom",
            book="user",
            page=self.name,
            comments=self.description,
            wavelength_range_um=(
                float(self.wavelength_um.min()),
                float(self.wavelength_um.max()),
            ),
        )
        return OpticalConstants(
            wavelength_um=self.wavelength_um,
            n=self.n,
            k=self.k,
            material_info=info,
        )

    @classmethod
    def from_arrays(
        cls,
        name: str,
        wavelength_um: np.ndarray,
        n: np.ndarray,
        k: np.ndarray,
        description: str = "",
    ) -> "CustomMaterial":
        """Create from numpy arrays."""
        return cls(
            name=name,
            wavelength_um=wavelength_um,
            n=n,
            k=k,
            description=description,
        )

    @classmethod
    def from_constant(
        cls,
        name: str,
        n: float,
        k: float = 0.0,
        wavelength_range_um: tuple[float, float] = (0.3, 2.5),
        n_points: int = 100,
        description: str = "",
    ) -> "CustomMaterial":
        """Create material with constant n, k values.

        Args:
            name: Material name
            n: Constant refractive index
            k: Constant extinction coefficient
            wavelength_range_um: Wavelength range
            n_points: Number of wavelength points
            description: Optional description

        Returns:
            CustomMaterial with constant optical properties
        """
        wavelength = np.linspace(
            wavelength_range_um[0], wavelength_range_um[1], n_points
        )
        return cls(
            name=name,
            wavelength_um=wavelength,
            n=np.full(n_points, n),
            k=np.full(n_points, k),
            description=description or f"Constant n={n}, k={k}",
        )

    @classmethod
    def from_file(cls, filepath: str | Path, name: str | None = None) -> "CustomMaterial":
        """Load from tab-separated file (wavelength, n, k).

        Args:
            filepath: Path to data file
            name: Material name (defaults to filename)

        Returns:
            CustomMaterial loaded from file
        """
        filepath = Path(filepath)
        if name is None:
            name = filepath.stem

        data = np.loadtxt(filepath)
        if data.ndim == 1:
            raise ValueError("File must have at least 2 columns")

        if data.shape[1] == 2:
            # wavelength, n only
            return cls(
                name=name,
                wavelength_um=data[:, 0],
                n=data[:, 1],
                k=np.zeros(len(data)),
            )
        else:
            # wavelength, n, k
            return cls(
                name=name,
                wavelength_um=data[:, 0],
                n=data[:, 1],
                k=data[:, 2],
            )

    def save(self, filepath: str | Path):
        """Save to tab-separated file."""
        filepath = Path(filepath)
        data = np.column_stack([self.wavelength_um, self.n, self.k])
        np.savetxt(filepath, data, fmt="%.6g", delimiter="\t")

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "wavelength_um": self.wavelength_um.tolist(),
            "n": self.n.tolist(),
            "k": self.k.tolist(),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CustomMaterial":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            wavelength_um=np.array(data["wavelength_um"]),
            n=np.array(data["n"]),
            k=np.array(data["k"]),
            description=data.get("description", ""),
        )


class CustomMaterialLibrary:
    """Manage a collection of custom materials."""

    def __init__(self):
        self._materials: dict[str, CustomMaterial] = {}

    def add(self, material: CustomMaterial):
        """Add a material to the library."""
        self._materials[material.name] = material

    def remove(self, name: str):
        """Remove a material from the library."""
        if name in self._materials:
            del self._materials[name]

    def get(self, name: str) -> CustomMaterial:
        """Get a material by name."""
        if name not in self._materials:
            raise KeyError(f"Material not found: {name}")
        return self._materials[name]

    def list(self) -> list[str]:
        """List all material names."""
        return list(self._materials.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._materials

    def __len__(self) -> int:
        return len(self._materials)

    def save(self, filepath: str | Path):
        """Save library to JSON file."""
        import json

        filepath = Path(filepath)
        data = {name: mat.to_dict() for name, mat in self._materials.items()}
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> "CustomMaterialLibrary":
        """Load library from JSON file."""
        import json

        filepath = Path(filepath)
        with open(filepath) as f:
            data = json.load(f)

        library = cls()
        for name, mat_data in data.items():
            library.add(CustomMaterial.from_dict(mat_data))
        return library


# Predefined common materials
def acrylic(wavelength_range_um: tuple[float, float] = (0.3, 2.5)) -> CustomMaterial:
    """PMMA/Acrylic - common binder material.

    Approximate values for PMMA.
    """
    n_points = 100
    wavelength = np.linspace(wavelength_range_um[0], wavelength_range_um[1], n_points)
    # PMMA has n ~ 1.49 with slight dispersion, k ~ 0 in visible
    n = 1.49 - 0.002 * (wavelength - 0.5)  # Simple dispersion model
    k = np.full(n_points, 1e-6)  # Nearly transparent

    return CustomMaterial(
        name="acrylic",
        wavelength_um=wavelength,
        n=n,
        k=k,
        description="PMMA/Acrylic binder (approximate)",
    )


def air() -> CustomMaterial:
    """Air - vacuum/air boundary material."""
    return CustomMaterial.from_constant(
        name="air",
        n=1.0,
        k=0.0,
        description="Air/vacuum",
    )
