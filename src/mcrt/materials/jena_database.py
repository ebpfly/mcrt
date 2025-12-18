"""Load and query the Jena/optool optical constants database.

The Jena database (DOCCD) contains optical constants for cosmic dust materials:
silicates, ices, carbonaceous materials, oxides, sulfides, and more.

Data format is .lnk files with:
- Header comments starting with # containing metadata
- First data line: N_wavelengths  density_g_cm3
- Data lines: wavelength_um  n  k
"""

from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np

from .database import MaterialInfo, OpticalConstants


@dataclass
class JenaMaterialMetadata:
    """Extended metadata from Jena .lnk files."""

    material: str = ""
    reference: str = ""
    name: str = ""
    material_class: str = ""
    state: str = ""
    formula: str = ""
    ads_link: str = ""
    bibtex_key: str = ""
    density_g_cm3: float | None = None
    n_wavelengths: int = 0


def parse_lnk_file(filepath: Path) -> tuple[OpticalConstants, JenaMaterialMetadata]:
    """Parse a Jena .lnk file.

    Args:
        filepath: Path to .lnk file

    Returns:
        Tuple of (OpticalConstants, JenaMaterialMetadata)
    """
    metadata = JenaMaterialMetadata()

    with open(filepath) as f:
        lines = f.readlines()

    # Parse header comments
    data_start = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("#"):
            # Parse metadata fields
            if ":" in line:
                key, _, value = line[1:].partition(":")
                key = key.strip().lower()
                value = value.strip()

                if key == "material":
                    metadata.material = value
                elif key == "reference":
                    metadata.reference = value
                elif key == "name":
                    metadata.name = value
                elif key == "class":
                    metadata.material_class = value
                elif key == "state":
                    metadata.state = value
                elif key == "formula":
                    metadata.formula = value
                elif key == "ads-link":
                    metadata.ads_link = value
                elif key == "bibtex-key":
                    metadata.bibtex_key = value
        elif line and not line.startswith("#"):
            data_start = i
            break

    # Parse first data line (N_wavelengths, density)
    first_data = lines[data_start].split()
    if len(first_data) >= 2:
        metadata.n_wavelengths = int(first_data[0])
        metadata.density_g_cm3 = float(first_data[1])

    # Parse wavelength, n, k data
    wavelength = []
    n = []
    k = []

    for line in lines[data_start + 1:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            wavelength.append(float(parts[0]))
            n.append(float(parts[1]))
            k.append(float(parts[2]))

    # Create MaterialInfo from metadata
    # Use filename as material_id
    material_id = f"jena/{filepath.stem}"

    # Build display name
    display_name = metadata.name or filepath.stem
    if metadata.state:
        display_name = f"{display_name} ({metadata.state})"

    info = MaterialInfo(
        material_id=material_id,
        name=display_name,
        shelf="jena",
        book=metadata.material_class or "Cosmic Dust",
        page=filepath.stem,
        data_path=str(filepath),
        references=metadata.reference,
        comments=f"Formula: {metadata.formula}" if metadata.formula else "",
        wavelength_range_um=(min(wavelength), max(wavelength)) if wavelength else None,
    )

    oc = OpticalConstants(
        wavelength_um=np.array(wavelength),
        n=np.array(n),
        k=np.array(k),
        material_info=info,
    )

    return oc, metadata


class JenaDatabase:
    """Interface to the Jena/optool optical constants database.

    This provides access to cosmic dust optical constants from the
    Jena DOCCD database (via optool curated files).
    """

    def __init__(self, db_path: str | Path | None = None):
        """Initialize the database.

        Args:
            db_path: Path to the jena-database directory.
                     If None, tries to find it relative to this package.
        """
        if db_path is None:
            # Try to find database relative to package
            package_dir = Path(__file__).parent.parent.parent.parent
            db_path = package_dir / "jena-database"
            if not db_path.exists():
                db_path = Path.cwd() / "jena-database"

        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Jena database not found at {self.db_path}. "
                "Please run the download script in jena-database/"
            )

        self._index: dict[str, MaterialInfo] = {}
        self._metadata: dict[str, JenaMaterialMetadata] = {}
        self._build_index()

    def _build_index(self):
        """Build index of available materials from .lnk files."""
        for lnk_file in self.db_path.glob("*.lnk"):
            try:
                oc, metadata = parse_lnk_file(lnk_file)
                if oc.material_info:
                    self._index[oc.material_info.material_id] = oc.material_info
                    self._metadata[oc.material_info.material_id] = metadata
            except Exception as e:
                print(f"Warning: Failed to parse {lnk_file}: {e}")

    def list_materials(
        self,
        material_class: str | None = None,
        search: str | None = None,
    ) -> list[MaterialInfo]:
        """List available materials.

        Args:
            material_class: Filter by class (Silicates, Carbon, etc.)
            search: Search string to filter by name

        Returns:
            List of MaterialInfo objects
        """
        results = list(self._index.values())

        if material_class:
            results = [m for m in results if m.book.lower() == material_class.lower()]

        if search:
            search_lower = search.lower()
            results = [
                m for m in results
                if search_lower in m.name.lower()
                or search_lower in m.material_id.lower()
            ]

        return sorted(results, key=lambda m: m.name)

    def list_classes(self) -> list[str]:
        """List available material classes."""
        return sorted(set(m.book for m in self._index.values()))

    def get_material_info(self, material_id: str) -> MaterialInfo:
        """Get information about a material.

        Args:
            material_id: Material identifier (e.g., "jena/pyr-mg70-Dorschner1995")

        Returns:
            MaterialInfo object
        """
        if material_id not in self._index:
            raise KeyError(f"Material not found: {material_id}")
        return self._index[material_id]

    def get_metadata(self, material_id: str) -> JenaMaterialMetadata:
        """Get extended metadata for a material.

        Args:
            material_id: Material identifier

        Returns:
            JenaMaterialMetadata object
        """
        if material_id not in self._metadata:
            raise KeyError(f"Material not found: {material_id}")
        return self._metadata[material_id]

    def get_optical_constants(
        self,
        material_id: str,
        wavelength_range_um: tuple[float, float] | None = None,
    ) -> OpticalConstants:
        """Load optical constants for a material.

        Args:
            material_id: Material identifier (e.g., "jena/pyr-mg70-Dorschner1995")
            wavelength_range_um: Optional (min, max) wavelength range

        Returns:
            OpticalConstants object
        """
        info = self.get_material_info(material_id)

        # Parse the .lnk file
        filepath = Path(info.data_path)
        if not filepath.is_absolute():
            filepath = self.db_path / f"{info.page}.lnk"

        oc, _ = parse_lnk_file(filepath)

        # Trim to range if specified
        if wavelength_range_um:
            oc = oc.trim(*wavelength_range_um)

        return oc


# Category mapping for Jena materials
JENA_CATEGORIES = {
    "Silicates": ["pyr-", "ol-", "astrosil"],
    "Carbon": ["c-z", "c-p", "c-gra", "c-nano", "c-org"],
    "Oxides": ["sio2", "cor-c"],
    "Metals & Sulfides": ["fe-c", "fes", "sic"],
    "Ices": ["h2o-", "co2-", "co-a", "nh3-", "ch4-", "ch3oh-"],
}


def get_jena_category(filename: str) -> str:
    """Determine category for a Jena material by filename."""
    filename_lower = filename.lower()
    for category, patterns in JENA_CATEGORIES.items():
        for pattern in patterns:
            if pattern in filename_lower:
                return category
    return "Other"
