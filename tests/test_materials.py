"""Tests for materials module."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestOpticalConstants:
    """Tests for OpticalConstants class."""

    def test_creation(self):
        """Test creating OpticalConstants."""
        from mcrt.materials.database import OpticalConstants

        oc = OpticalConstants(
            wavelength_um=np.array([0.3, 0.5, 0.7]),
            n=np.array([1.5, 1.48, 1.47]),
            k=np.array([0.01, 0.01, 0.01]),
        )

        assert len(oc.wavelength_um) == 3
        assert len(oc.n) == 3
        assert len(oc.k) == 3

    def test_interpolate(self):
        """Test interpolating optical constants."""
        from mcrt.materials.database import OpticalConstants

        oc = OpticalConstants(
            wavelength_um=np.array([0.3, 0.5, 0.7, 1.0]),
            n=np.array([1.5, 1.48, 1.47, 1.45]),
            k=np.array([0.01, 0.01, 0.01, 0.01]),
        )

        new_wl = np.array([0.4, 0.6, 0.8])
        interpolated = oc.interpolate(new_wl)

        assert len(interpolated.wavelength_um) == 3
        assert np.allclose(interpolated.wavelength_um, new_wl)

    def test_trim(self):
        """Test trimming to wavelength range."""
        from mcrt.materials.database import OpticalConstants

        oc = OpticalConstants(
            wavelength_um=np.array([0.3, 0.5, 0.7, 1.0, 1.5]),
            n=np.array([1.5, 1.48, 1.47, 1.45, 1.44]),
            k=np.array([0.01, 0.01, 0.01, 0.01, 0.01]),
        )

        trimmed = oc.trim(0.4, 1.2)

        assert all(trimmed.wavelength_um >= 0.4)
        assert all(trimmed.wavelength_um <= 1.2)

    def test_to_fos_format(self):
        """Test exporting to FOS format."""
        from mcrt.materials.database import OpticalConstants

        oc = OpticalConstants(
            wavelength_um=np.array([0.3, 0.5]),
            n=np.array([1.5, 1.48]),
            k=np.array([0.01, 0.02]),
        )

        fos_str = oc.to_fos_format()
        lines = fos_str.strip().split("\n")

        assert len(lines) == 2
        assert "0.3" in lines[0]
        assert "1.5" in lines[0]


class TestCustomMaterial:
    """Tests for CustomMaterial class."""

    def test_creation(self):
        """Test creating a custom material."""
        from mcrt.materials.custom import CustomMaterial

        material = CustomMaterial(
            name="test",
            wavelength_um=np.array([0.3, 0.5, 0.7]),
            n=np.array([1.5, 1.48, 1.47]),
            k=np.array([0.01, 0.01, 0.01]),
        )

        assert material.name == "test"
        assert len(material.wavelength_um) == 3

    def test_from_constant(self):
        """Test creating material with constant values."""
        from mcrt.materials.custom import CustomMaterial

        material = CustomMaterial.from_constant(
            name="constant_test",
            n=1.5,
            k=0.01,
            wavelength_range_um=(0.3, 2.5),
            n_points=50,
        )

        assert material.name == "constant_test"
        assert len(material.wavelength_um) == 50
        assert np.allclose(material.n, 1.5)
        assert np.allclose(material.k, 0.01)

    def test_to_optical_constants(self):
        """Test converting to OpticalConstants."""
        from mcrt.materials.custom import CustomMaterial

        material = CustomMaterial(
            name="test",
            wavelength_um=np.array([0.3, 0.5, 0.7]),
            n=np.array([1.5, 1.48, 1.47]),
            k=np.array([0.01, 0.01, 0.01]),
        )

        oc = material.to_optical_constants()

        assert oc.material_info is not None
        assert oc.material_info.name == "test"

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        from mcrt.materials.custom import CustomMaterial

        original = CustomMaterial(
            name="test",
            wavelength_um=np.array([0.3, 0.5, 0.7]),
            n=np.array([1.5, 1.48, 1.47]),
            k=np.array([0.01, 0.01, 0.01]),
            description="Test material",
        )

        data = original.to_dict()
        restored = CustomMaterial.from_dict(data)

        assert restored.name == original.name
        assert np.allclose(restored.wavelength_um, original.wavelength_um)
        assert np.allclose(restored.n, original.n)
        assert np.allclose(restored.k, original.k)

    def test_validation_mismatched_lengths(self):
        """Test validation of mismatched array lengths."""
        from mcrt.materials.custom import CustomMaterial

        with pytest.raises(ValueError):
            CustomMaterial(
                name="test",
                wavelength_um=np.array([0.3, 0.5, 0.7]),
                n=np.array([1.5, 1.48]),  # Wrong length
                k=np.array([0.01, 0.01, 0.01]),
            )


class TestCustomMaterialLibrary:
    """Tests for CustomMaterialLibrary class."""

    def test_add_and_get(self):
        """Test adding and retrieving materials."""
        from mcrt.materials.custom import CustomMaterial, CustomMaterialLibrary

        library = CustomMaterialLibrary()
        material = CustomMaterial(
            name="test",
            wavelength_um=np.array([0.3, 0.5, 0.7]),
            n=np.array([1.5, 1.48, 1.47]),
            k=np.array([0.01, 0.01, 0.01]),
        )

        library.add(material)

        assert "test" in library
        retrieved = library.get("test")
        assert retrieved.name == "test"

    def test_remove(self):
        """Test removing materials."""
        from mcrt.materials.custom import CustomMaterial, CustomMaterialLibrary

        library = CustomMaterialLibrary()
        material = CustomMaterial(
            name="test",
            wavelength_um=np.array([0.3, 0.5, 0.7]),
            n=np.array([1.5, 1.48, 1.47]),
            k=np.array([0.01, 0.01, 0.01]),
        )

        library.add(material)
        library.remove("test")

        assert "test" not in library

    def test_list(self):
        """Test listing materials."""
        from mcrt.materials.custom import CustomMaterial, CustomMaterialLibrary

        library = CustomMaterialLibrary()
        for name in ["a", "b", "c"]:
            library.add(CustomMaterial(
                name=name,
                wavelength_um=np.array([0.3]),
                n=np.array([1.5]),
                k=np.array([0.01]),
            ))

        names = library.list()
        assert len(names) == 3
        assert set(names) == {"a", "b", "c"}


class TestPredefinedMaterials:
    """Tests for predefined materials."""

    def test_acrylic(self):
        """Test acrylic material."""
        from mcrt.materials.custom import acrylic

        material = acrylic()
        assert material.name == "acrylic"
        assert len(material.wavelength_um) > 0

    def test_air(self):
        """Test air material."""
        from mcrt.materials.custom import air

        material = air()
        assert material.name == "air"
        assert np.allclose(material.n, 1.0)
        assert np.allclose(material.k, 0.0)
