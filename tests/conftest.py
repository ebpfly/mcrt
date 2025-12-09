"""Pytest configuration and fixtures for MCRT tests."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


@pytest.fixture
def mock_fos_wrapper():
    """Create a mock FOS wrapper that returns simulated results."""
    from mcrt.fos.output_parser import FOSResult

    mock = MagicMock()

    # Create a mock result
    wavelengths = np.linspace(0.3, 2.5, 221)
    mock_result = FOSResult(
        wavelength_um=wavelengths,
        reflectance=np.random.uniform(0.1, 0.9, len(wavelengths)),
        absorptance=np.random.uniform(0.05, 0.3, len(wavelengths)),
        transmittance=np.random.uniform(0.0, 0.2, len(wavelengths)),
        n_photons=10000,
    )

    mock.run_simple.return_value = mock_result
    mock.work_dir = Path("/tmp/mcrt_test")
    mock.fos_path = Path("/tmp/fos")

    return mock


@pytest.fixture
def mock_material_db():
    """Create a mock material database."""
    from mcrt.materials.database import MaterialInfo, OpticalConstants

    mock = MagicMock()

    # Mock material info
    mock_materials = [
        MaterialInfo(
            material_id="main/Ag/Johnson",
            name="Silver - Johnson",
            shelf="main",
            book="Ag",
            page="Johnson",
            wavelength_range_um=(0.1879, 1.937),
        ),
        MaterialInfo(
            material_id="main/Au/Johnson",
            name="Gold - Johnson",
            shelf="main",
            book="Au",
            page="Johnson",
            wavelength_range_um=(0.1879, 1.937),
        ),
        MaterialInfo(
            material_id="main/SiO2/Malitson",
            name="Silicon dioxide - Malitson",
            shelf="main",
            book="SiO2",
            page="Malitson",
            wavelength_range_um=(0.21, 6.7),
        ),
    ]

    mock.list_materials.return_value = mock_materials
    mock.list_shelves.return_value = ["main", "organic", "glass"]

    # Mock get_material_info
    def get_info(material_id):
        for m in mock_materials:
            if m.material_id == material_id:
                return m
        raise KeyError(f"Material not found: {material_id}")

    mock.get_material_info.side_effect = get_info

    # Mock get_optical_constants
    def get_oc(material_id, wavelength_range=None):
        info = get_info(material_id)
        wavelengths = np.linspace(0.3, 2.5, 100)
        return OpticalConstants(
            wavelength_um=wavelengths,
            n=np.full(100, 1.5),
            k=np.full(100, 0.01),
            material_info=info,
        )

    mock.get_optical_constants.side_effect = get_oc

    return mock


@pytest.fixture
def test_client(mock_fos_wrapper, mock_material_db):
    """Create a test client with mocked dependencies."""
    from mcrt.api import routes

    # Initialize dependencies manually (save originals for cleanup)
    from mcrt.simulation.manager import SimulationManager
    from mcrt.api.sse import SSEManager
    from mcrt.materials.custom import CustomMaterialLibrary

    original_fos = routes._fos_wrapper
    original_db = routes._material_db
    original_manager = routes._simulation_manager
    original_sse = routes._sse_manager
    original_custom = routes._custom_materials

    routes._fos_wrapper = mock_fos_wrapper
    routes._material_db = mock_material_db
    routes._simulation_manager = SimulationManager(mock_fos_wrapper)
    routes._sse_manager = SSEManager()
    routes._custom_materials = CustomMaterialLibrary()

    # Create app without auto-initialization
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(routes.router)

    client = TestClient(app)

    try:
        yield client
    finally:
        # Restore original values
        routes._fos_wrapper = original_fos
        routes._material_db = original_db
        routes._simulation_manager = original_manager
        routes._sse_manager = original_sse
        routes._custom_materials = original_custom


@pytest.fixture
def sample_optical_constants():
    """Sample optical constants for testing."""
    return {
        "wavelength_um": [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5],
        "n": [1.5, 1.48, 1.47, 1.46, 1.45, 1.44, 1.43],
        "k": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    }


@pytest.fixture
def sample_simulation_config(sample_optical_constants):
    """Sample simulation configuration for testing."""
    return {
        "particle_materials": {
            "1": sample_optical_constants,
        },
        "matrix_materials": {
            "1": sample_optical_constants,
        },
        "layers": [
            {
                "matrix_id": 1,
                "thickness_um": 100,
                "particles": [
                    {
                        "material_id": 1,
                        "diameter_um": 0.5,
                        "volume_fraction": 10,
                        "std_dev": 0.0,
                    }
                ],
            }
        ],
        "wavelength_start_um": 0.3,
        "wavelength_end_um": 2.5,
        "wavelength_interval_um": 0.01,
        "photons_target": 10000,
        "n_batches": 2,
    }
