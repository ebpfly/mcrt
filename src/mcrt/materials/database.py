"""Load and query the refractiveindex.info optical constants database."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import re
import numpy as np
import yaml


@dataclass
class MaterialInfo:
    """Information about a material in the database.

    Attributes:
        material_id: Unique identifier (e.g., "main/Ag/Johnson")
        name: Display name (e.g., "Silver - Johnson")
        shelf: Database shelf (main, organic, glass, etc.)
        book: Material category/element
        page: Specific dataset/reference
        data_path: Relative path to data file from catalog
        references: Reference string from the data file
        comments: Additional comments
        wavelength_range_um: Tuple of (min, max) wavelength in micrometers
    """

    material_id: str
    name: str
    shelf: str
    book: str
    page: str
    data_path: str = ""
    references: str = ""
    comments: str = ""
    wavelength_range_um: tuple[float, float] | None = None


@dataclass
class OpticalConstants:
    """Optical constants (n, k) as a function of wavelength.

    Attributes:
        wavelength_um: Wavelength array in micrometers
        n: Real part of refractive index
        k: Imaginary part (extinction coefficient)
        material_info: Associated material information
    """

    wavelength_um: np.ndarray
    n: np.ndarray
    k: np.ndarray
    material_info: MaterialInfo | None = None

    def __post_init__(self):
        self.wavelength_um = np.asarray(self.wavelength_um)
        self.n = np.asarray(self.n)
        self.k = np.asarray(self.k)

    def interpolate(
        self,
        wavelength_um: np.ndarray,
        bounds_error: bool = False,
        fill_value: float | tuple[float, float] = "extrapolate",
    ) -> "OpticalConstants":
        """Interpolate optical constants to new wavelength grid.

        Args:
            wavelength_um: New wavelength array
            bounds_error: If True, raise error for out-of-bounds
            fill_value: Value for out-of-bounds points

        Returns:
            New OpticalConstants with interpolated values
        """
        from scipy.interpolate import interp1d

        wavelength_um = np.asarray(wavelength_um)

        # Interpolate n
        f_n = interp1d(
            self.wavelength_um,
            self.n,
            kind="linear",
            bounds_error=bounds_error,
            fill_value=fill_value,
        )
        n_interp = f_n(wavelength_um)

        # Interpolate k
        f_k = interp1d(
            self.wavelength_um,
            self.k,
            kind="linear",
            bounds_error=bounds_error,
            fill_value=fill_value,
        )
        k_interp = f_k(wavelength_um)

        return OpticalConstants(
            wavelength_um=wavelength_um,
            n=n_interp,
            k=k_interp,
            material_info=self.material_info,
        )

    def trim(
        self, wavelength_min_um: float, wavelength_max_um: float
    ) -> "OpticalConstants":
        """Trim to wavelength range.

        Args:
            wavelength_min_um: Minimum wavelength
            wavelength_max_um: Maximum wavelength

        Returns:
            New OpticalConstants trimmed to range
        """
        mask = (self.wavelength_um >= wavelength_min_um) & (
            self.wavelength_um <= wavelength_max_um
        )
        return OpticalConstants(
            wavelength_um=self.wavelength_um[mask],
            n=self.n[mask],
            k=self.k[mask],
            material_info=self.material_info,
        )

    def to_fos_format(self) -> str:
        """Export as FOS material file format (tab-separated wavelength, n, k)."""
        lines = []
        for wl, n, k in zip(self.wavelength_um, self.n, self.k):
            lines.append(f"{wl}\t{n}\t{k}")
        return "\n".join(lines)

    def save_fos_file(self, filepath: str | Path) -> Path:
        """Save as FOS material file.

        Args:
            filepath: Output file path

        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        filepath.write_text(self.to_fos_format())
        return filepath


class MaterialDatabase:
    """Interface to the refractiveindex.info optical constants database.

    This class provides access to the refractiveindex.info database,
    which contains tabulated optical constants (n, k) for many materials.
    """

    def __init__(self, db_path: str | Path | None = None):
        """Initialize the database.

        Args:
            db_path: Path to the refractiveindex.info database directory.
                     If None, tries to find it relative to this package.
        """
        if db_path is None:
            # Try to find database relative to package
            package_dir = Path(__file__).parent.parent.parent.parent
            db_path = package_dir / "refractiveindex-database" / "database"
            if not db_path.exists():
                db_path = Path.cwd() / "refractiveindex-database" / "database"

        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {self.db_path}. "
                "Please clone https://github.com/polyanskiy/refractiveindex.info-database"
            )

        self._index: dict[str, MaterialInfo] = {}
        self._build_index()

    def _build_index(self):
        """Build index of available materials from catalog."""
        catalog_path = self.db_path / "catalog-nk.yml"
        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found at {catalog_path}")

        with open(catalog_path) as f:
            catalog = yaml.safe_load(f)

        for shelf_data in catalog:
            shelf = shelf_data.get("SHELF", "")
            shelf_name = shelf_data.get("name", shelf)

            for book_data in shelf_data.get("content", []):
                if "DIVIDER" in book_data:
                    continue

                book = book_data.get("BOOK", "")
                book_name = book_data.get("name", book)

                for page_data in book_data.get("content", []):
                    if "DIVIDER" in page_data:
                        continue

                    page = page_data.get("PAGE", "")
                    page_name = page_data.get("name", page)
                    data_path = page_data.get("data", "")

                    if not data_path:
                        continue

                    material_id = f"{shelf}/{book}/{page}"
                    display_name = f"{book_name}"
                    if page_name and page_name != book_name:
                        display_name = f"{book_name} - {page_name}"

                    self._index[material_id] = MaterialInfo(
                        material_id=material_id,
                        name=display_name,
                        shelf=shelf,
                        book=book,
                        page=page,
                        data_path=data_path,
                    )

    def list_materials(
        self,
        shelf: str | None = None,
        search: str | None = None,
    ) -> list[MaterialInfo]:
        """List available materials.

        Args:
            shelf: Filter by shelf (main, organic, glass, etc.)
            search: Search string to filter by name

        Returns:
            List of MaterialInfo objects
        """
        results = list(self._index.values())

        if shelf:
            results = [m for m in results if m.shelf == shelf]

        if search:
            search_lower = search.lower()
            results = [
                m
                for m in results
                if search_lower in m.name.lower()
                or search_lower in m.material_id.lower()
            ]

        return sorted(results, key=lambda m: m.name)

    def list_shelves(self) -> list[str]:
        """List available shelves (categories)."""
        return sorted(set(m.shelf for m in self._index.values()))

    def list_books(self, shelf: str) -> list[str]:
        """List books (subcategories) in a shelf."""
        return sorted(
            set(m.book for m in self._index.values() if m.shelf == shelf)
        )

    def get_material_info(self, material_id: str) -> MaterialInfo:
        """Get information about a material.

        Args:
            material_id: Material identifier (e.g., "main/Ag/Johnson")

        Returns:
            MaterialInfo object
        """
        if material_id not in self._index:
            raise KeyError(f"Material not found: {material_id}")
        return self._index[material_id]

    def get_optical_constants(
        self,
        material_id: str,
        wavelength_range_um: tuple[float, float] | None = None,
    ) -> OpticalConstants:
        """Load optical constants for a material.

        Args:
            material_id: Material identifier (e.g., "main/Ag/Johnson")
            wavelength_range_um: Optional (min, max) wavelength range

        Returns:
            OpticalConstants object
        """
        info = self.get_material_info(material_id)

        # Use the data path from the catalog
        if info.data_path:
            data_file_path = self.db_path / "data" / info.data_path
        else:
            # Fall back to constructed path for older indexes
            data_file_path = self.db_path / "data" / info.shelf / info.book / "nk" / f"{info.page}.yml"
            if not data_file_path.exists():
                data_file_path = self.db_path / "data" / info.shelf / info.book / f"{info.page}.yml"

        if not data_file_path.exists():
            raise FileNotFoundError(f"Data file not found for {material_id} at {data_file_path}")

        # Parse the YAML file
        with open(data_file_path) as f:
            data = yaml.safe_load(f)

        # Update material info with references/comments
        info.references = data.get("REFERENCES", "")
        info.comments = data.get("COMMENTS", "")

        # Parse optical constants
        wavelength, n, k = self._parse_data(data)

        # Update wavelength range in info
        info.wavelength_range_um = (float(wavelength.min()), float(wavelength.max()))

        result = OpticalConstants(
            wavelength_um=wavelength,
            n=n,
            k=k,
            material_info=info,
        )

        # Trim to range if specified
        if wavelength_range_um:
            result = result.trim(*wavelength_range_um)

        return result

    def _parse_data(
        self, data: dict, n_points: int = 500
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse optical constants from YAML data.

        Supports tabulated nk format and formula-based data.

        Args:
            data: Parsed YAML data dict
            n_points: Number of wavelength points for formula evaluation
        """
        data_section = data.get("DATA", [])
        if not data_section:
            raise ValueError("No DATA section found")

        # First pass: look for tabulated data (preferred)
        for entry in data_section:
            data_type = entry.get("type", "")

            if data_type == "tabulated nk":
                return self._parse_tabulated_nk(entry["data"])
            elif data_type == "tabulated n":
                # Only n provided, k=0
                wl, n = self._parse_tabulated_n(entry["data"])
                return wl, n, np.zeros_like(n)
            elif data_type == "tabulated k":
                # Only k provided - need n from another entry
                wl, k = self._parse_tabulated_k(entry["data"])
                # Look for n in other entries
                n = np.ones_like(k)  # Default n=1
                for other in data_section:
                    if other.get("type") == "tabulated n":
                        wl_n, n_data = self._parse_tabulated_n(other["data"])
                        # Interpolate to k wavelengths
                        from scipy.interpolate import interp1d
                        f = interp1d(wl_n, n_data, fill_value="extrapolate")
                        n = f(wl)
                        break
                return wl, n, k

        # Second pass: try formula-based data
        for entry in data_section:
            data_type = entry.get("type", "")

            if data_type.startswith("formula"):
                formula_num = int(data_type.split()[-1])
                wl_range = entry.get("wavelength_range", "0.3 2.5")
                coeffs_str = entry.get("coefficients", "")

                # Parse wavelength range
                wl_parts = wl_range.split()
                wl_min, wl_max = float(wl_parts[0]), float(wl_parts[1])

                # Parse coefficients
                coeffs = [float(c) for c in coeffs_str.split()]

                # Generate wavelength array
                wavelength = np.linspace(wl_min, wl_max, n_points)

                # Evaluate formula
                n = self._evaluate_formula(formula_num, wavelength, coeffs)
                k = np.zeros_like(n)  # Formulas typically only give n, not k

                return wavelength, n, k

        raise ValueError(f"Unsupported data format in {data_section}")

    def _evaluate_formula(
        self, formula_num: int, wavelength: np.ndarray, coeffs: list[float]
    ) -> np.ndarray:
        """Evaluate refractive index formula.

        Implements formulas from refractiveindex.info:
        - Formula 1: Sellmeier (n² - 1 form)
        - Formula 2: Sellmeier-2 (n² form)
        - Formula 3: Polynomial
        - Formula 4: RefractiveIndex.INFO
        - Formula 5: Cauchy
        - Formula 6: Gases
        - Formula 7: Herzberger
        - Formula 8: Retro
        - Formula 9: Exotic

        Args:
            formula_num: Formula type number (1-9)
            wavelength: Wavelength array in micrometers
            coeffs: Coefficient list from database

        Returns:
            Refractive index array
        """
        wl = wavelength  # shorthand
        wl2 = wl ** 2

        if formula_num == 1:
            # Sellmeier (absolute): n² - 1 = C0 + Σ (Ci * λ²)/(λ² - Ci+1²)
            n_sq = 1 + coeffs[0]
            for i in range(1, len(coeffs) - 1, 2):
                if i + 1 < len(coeffs):
                    B = coeffs[i]
                    C = coeffs[i + 1]
                    n_sq = n_sq + (B * wl2) / (wl2 - C ** 2)
            return np.sqrt(np.maximum(n_sq, 1.0))

        elif formula_num == 2:
            # Sellmeier-2: n² = C0 + Σ (Ci * λ²)/(λ² - Ci+1)
            # Note: Ci+1 is NOT squared in this formula
            n_sq = coeffs[0]
            for i in range(1, len(coeffs) - 1, 2):
                if i + 1 < len(coeffs):
                    B = coeffs[i]
                    C = coeffs[i + 1]
                    n_sq = n_sq + (B * wl2) / (wl2 - C)
            return np.sqrt(np.maximum(n_sq, 1.0))

        elif formula_num == 3:
            # Polynomial: n² = Σ Ci * λ^(2i-2) where powers can be fractional
            # Format: C0 + C1*λ^E1 + C2*λ^E2 + ...
            n_sq = coeffs[0]
            for i in range(1, len(coeffs) - 1, 2):
                if i + 1 < len(coeffs):
                    C = coeffs[i]
                    E = coeffs[i + 1]
                    n_sq = n_sq + C * (wl ** E)
            return np.sqrt(np.maximum(n_sq, 1.0))

        elif formula_num == 4:
            # RefractiveIndex.INFO formula (complex, for k calculation)
            # n² = C0 + (C1*λ^C2)/(λ² - C3^C4) + (C5*λ^C6)/(λ² - C7^C8) + ...
            n_sq = coeffs[0]
            for i in range(1, len(coeffs) - 3, 4):
                if i + 3 < len(coeffs):
                    C1, C2, C3, C4 = coeffs[i:i+4]
                    n_sq = n_sq + (C1 * wl**C2) / (wl2 - C3**C4)
            # Additional polynomial terms
            for i in range(17, len(coeffs) - 1, 2):
                if i + 1 < len(coeffs):
                    C = coeffs[i]
                    E = coeffs[i + 1]
                    n_sq = n_sq + C * (wl ** E)
            return np.sqrt(np.maximum(n_sq, 1.0))

        elif formula_num == 5:
            # Cauchy: n = C0 + C1*λ² + C2/λ² + C3/λ⁴ + C4/λ⁶ + C5/λ⁸
            n = coeffs[0]
            if len(coeffs) > 1:
                n = n + coeffs[1] * wl2
            if len(coeffs) > 2:
                n = n + coeffs[2] / wl2
            if len(coeffs) > 3:
                n = n + coeffs[3] / (wl2 ** 2)
            if len(coeffs) > 4:
                n = n + coeffs[4] / (wl2 ** 3)
            if len(coeffs) > 5:
                n = n + coeffs[5] / (wl2 ** 4)
            return np.maximum(n, 1.0)

        elif formula_num == 6:
            # Gases: n - 1 = C0 + C1/(C2 - λ⁻²) + C3/(C4 - λ⁻²) + ...
            n_minus_1 = coeffs[0]
            inv_wl2 = 1 / wl2
            for i in range(1, len(coeffs) - 1, 2):
                if i + 1 < len(coeffs):
                    C = coeffs[i]
                    D = coeffs[i + 1]
                    n_minus_1 = n_minus_1 + C / (D - inv_wl2)
            return n_minus_1 + 1

        elif formula_num == 7:
            # Herzberger: n = C0 + C1*L + C2*L² + C3*λ² + C4*λ⁴ + C5*λ⁶
            # where L = 1/(λ² - 0.028)
            L = 1 / (wl2 - 0.028)
            n = coeffs[0]
            if len(coeffs) > 1:
                n = n + coeffs[1] * L
            if len(coeffs) > 2:
                n = n + coeffs[2] * L ** 2
            if len(coeffs) > 3:
                n = n + coeffs[3] * wl2
            if len(coeffs) > 4:
                n = n + coeffs[4] * wl2 ** 2
            if len(coeffs) > 5:
                n = n + coeffs[5] * wl2 ** 3
            return np.maximum(n, 1.0)

        elif formula_num == 8:
            # Retro: (n² - 1)/(n² + 2) = C0 + C1*λ² + C2/λ² + C3/λ⁴ + ...
            # Solve for n
            lhs = coeffs[0]
            if len(coeffs) > 1:
                lhs = lhs + coeffs[1] * wl2
            if len(coeffs) > 2:
                lhs = lhs + coeffs[2] / wl2
            if len(coeffs) > 3:
                lhs = lhs + coeffs[3] / wl2 ** 2
            # From (n² - 1)/(n² + 2) = lhs, solve for n²:
            # n² - 1 = lhs * (n² + 2)
            # n² - 1 = lhs*n² + 2*lhs
            # n² - lhs*n² = 1 + 2*lhs
            # n²(1 - lhs) = 1 + 2*lhs
            # n² = (1 + 2*lhs) / (1 - lhs)
            n_sq = (1 + 2 * lhs) / (1 - lhs)
            return np.sqrt(np.maximum(n_sq, 1.0))

        elif formula_num == 9:
            # Exotic: n² = C0 + C1/(λ² - C2) + C3*(λ - C4)/((λ - C4)² + C5)
            n_sq = coeffs[0]
            if len(coeffs) > 2:
                n_sq = n_sq + coeffs[1] / (wl2 - coeffs[2])
            if len(coeffs) > 5:
                diff = wl - coeffs[4]
                n_sq = n_sq + coeffs[3] * diff / (diff ** 2 + coeffs[5])
            return np.sqrt(np.maximum(n_sq, 1.0))

        else:
            raise ValueError(f"Unknown formula type: {formula_num}")

    def _parse_tabulated_nk(
        self, data_str: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse tabulated nk data (wavelength n k, space-separated)."""
        lines = data_str.strip().split("\n")
        wavelength = []
        n = []
        k = []

        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                wavelength.append(float(parts[0]))
                n.append(float(parts[1]))
                k.append(float(parts[2]))

        return np.array(wavelength), np.array(n), np.array(k)

    def _parse_tabulated_n(
        self, data_str: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Parse tabulated n data (wavelength n, space-separated)."""
        lines = data_str.strip().split("\n")
        wavelength = []
        n = []

        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                wavelength.append(float(parts[0]))
                n.append(float(parts[1]))

        return np.array(wavelength), np.array(n)

    def _parse_tabulated_k(
        self, data_str: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Parse tabulated k data (wavelength k, space-separated)."""
        lines = data_str.strip().split("\n")
        wavelength = []
        k = []

        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                wavelength.append(float(parts[0]))
                k.append(float(parts[1]))

        return np.array(wavelength), np.array(k)

    def export_for_fos(
        self,
        material_id: str,
        output_path: str | Path,
        wavelength_range_um: tuple[float, float] | None = None,
    ) -> Path:
        """Export material as FOS-compatible file.

        Args:
            material_id: Material identifier
            output_path: Output file path
            wavelength_range_um: Optional wavelength range to export

        Returns:
            Path to exported file
        """
        oc = self.get_optical_constants(material_id, wavelength_range_um)
        return oc.save_fos_file(output_path)


# Common mineral materials for convenience
COMMON_MINERALS = {
    "olivine": "main/Mg2SiO4/Fabian-o",  # Forsterite, ordinary ray
    "pyroxene": "main/MgSiO3/Dorschner",  # Enstatite
    "quartz": "main/SiO2/Malitson",
    "magnetite": "main/Fe3O4/Querry",
    "hematite": "main/Fe2O3/Longtin-alpha",
    "iron": "main/Fe/Johnson",
    "aluminum": "main/Al/Rakic",
    "silver": "main/Ag/Johnson",
    "gold": "main/Au/Johnson",
    "copper": "main/Cu/Johnson",
    "water": "main/H2O/Hale",
    "ice": "main/H2O/Warren-ice",
}


def get_common_material(name: str, db: MaterialDatabase | None = None) -> OpticalConstants:
    """Get optical constants for a common material by simple name.

    Args:
        name: Simple name (e.g., "olivine", "iron", "water")
        db: MaterialDatabase instance (created if not provided)

    Returns:
        OpticalConstants for the material
    """
    if name.lower() not in COMMON_MINERALS:
        raise KeyError(
            f"Unknown common material: {name}. "
            f"Available: {list(COMMON_MINERALS.keys())}"
        )

    if db is None:
        db = MaterialDatabase()

    material_id = COMMON_MINERALS[name.lower()]
    return db.get_optical_constants(material_id)
