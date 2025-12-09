"""Tests for simulation module."""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch
import json


class TestFOSResult:
    """Tests for FOSResult class."""

    def test_creation(self):
        """Test creating FOSResult."""
        from mcrt.fos.output_parser import FOSResult

        result = FOSResult(
            wavelength_um=np.array([0.3, 0.5, 0.7]),
            reflectance=np.array([0.5, 0.6, 0.7]),
            absorptance=np.array([0.3, 0.2, 0.1]),
            transmittance=np.array([0.2, 0.2, 0.2]),
            n_photons=10000,
        )

        assert len(result.wavelength_um) == 3
        assert result.n_photons == 10000

    def test_n_wavelengths(self):
        """Test n_wavelengths property."""
        from mcrt.fos.output_parser import FOSResult

        result = FOSResult(
            wavelength_um=np.array([0.3, 0.5, 0.7, 1.0, 1.5]),
            reflectance=np.array([0.5, 0.6, 0.7, 0.6, 0.5]),
            absorptance=np.array([0.3, 0.2, 0.1, 0.2, 0.3]),
            transmittance=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        )

        assert result.n_wavelengths == 5

    def test_to_dict(self):
        """Test converting to dictionary."""
        from mcrt.fos.output_parser import FOSResult

        result = FOSResult(
            wavelength_um=np.array([0.3, 0.5]),
            reflectance=np.array([0.5, 0.6]),
            absorptance=np.array([0.3, 0.2]),
            transmittance=np.array([0.2, 0.2]),
            n_photons=10000,
        )

        data = result.to_dict()

        assert "wavelength_um" in data
        assert "reflectance" in data
        assert "absorptance" in data
        assert "transmittance" in data
        assert "n_photons" in data
        assert isinstance(data["wavelength_um"], list)

    def test_from_dict(self):
        """Test creating from dictionary."""
        from mcrt.fos.output_parser import FOSResult

        data = {
            "wavelength_um": [0.3, 0.5],
            "reflectance": [0.5, 0.6],
            "absorptance": [0.3, 0.2],
            "transmittance": [0.2, 0.2],
            "n_photons": 10000,
        }

        result = FOSResult.from_dict(data)

        assert len(result.wavelength_um) == 2
        assert result.n_photons == 10000

    def test_merge(self):
        """Test merging two results."""
        from mcrt.fos.output_parser import FOSResult

        result1 = FOSResult(
            wavelength_um=np.array([0.3, 0.5]),
            reflectance=np.array([0.4, 0.5]),
            absorptance=np.array([0.3, 0.3]),
            transmittance=np.array([0.3, 0.2]),
            n_photons=5000,
        )

        result2 = FOSResult(
            wavelength_um=np.array([0.3, 0.5]),
            reflectance=np.array([0.6, 0.7]),
            absorptance=np.array([0.2, 0.1]),
            transmittance=np.array([0.2, 0.2]),
            n_photons=5000,
        )

        merged = result1.merge(result2, weight=0.5)

        assert merged.n_photons == 10000
        assert np.allclose(merged.reflectance, [0.5, 0.6])

    def test_accumulate(self):
        """Test accumulating multiple results."""
        from mcrt.fos.output_parser import FOSResult

        results = [
            FOSResult(
                wavelength_um=np.array([0.3, 0.5]),
                reflectance=np.array([0.4, 0.5]),
                absorptance=np.array([0.3, 0.3]),
                transmittance=np.array([0.3, 0.2]),
                n_photons=5000,
            )
            for _ in range(3)
        ]

        accumulated = FOSResult.accumulate(results)

        assert accumulated.n_photons == 15000
        assert len(accumulated.wavelength_um) == 2

    def test_accumulate_empty_raises(self):
        """Test that accumulating empty list raises."""
        from mcrt.fos.output_parser import FOSResult

        with pytest.raises(ValueError):
            FOSResult.accumulate([])


class TestSimulationSession:
    """Tests for SimulationSession class."""

    def test_create(self):
        """Test creating a simulation session."""
        from mcrt.simulation.manager import SimulationSession, SimulationStatus
        from mcrt.materials.database import OpticalConstants

        particle_materials = {
            1: OpticalConstants(
                wavelength_um=np.array([0.3, 0.5, 0.7]),
                n=np.array([1.5, 1.48, 1.47]),
                k=np.array([0.01, 0.01, 0.01]),
            )
        }
        matrix_materials = {
            1: OpticalConstants(
                wavelength_um=np.array([0.3, 0.5, 0.7]),
                n=np.array([1.4, 1.4, 1.4]),
                k=np.array([0.0, 0.0, 0.0]),
            )
        }
        layers = [
            {
                "matrix_id": 1,
                "thickness_um": 100,
                "particles": [
                    {"material_id": 1, "diameter_um": 0.5, "volume_fraction": 10}
                ],
            }
        ]

        session = SimulationSession.create(
            particle_materials=particle_materials,
            matrix_materials=matrix_materials,
            layers=layers,
        )

        assert session.session_id is not None
        assert session.status == SimulationStatus.IDLE
        assert session.batches_completed == 0

    def test_progress(self):
        """Test progress calculation."""
        from mcrt.simulation.manager import SimulationSession, SimulationStatus
        from mcrt.materials.database import OpticalConstants

        session = SimulationSession(
            session_id="test",
            status=SimulationStatus.RUNNING,
            config={},
            particle_materials={},
            matrix_materials={},
            photons_completed=50000,
            photons_target=100000,
        )

        assert session.progress == 0.5
        assert session.progress_percent == 50.0

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        from mcrt.simulation.manager import SimulationSession, SimulationStatus
        from mcrt.materials.database import OpticalConstants

        original = SimulationSession(
            session_id="test-session",
            status=SimulationStatus.PAUSED,
            config={"test": "config"},
            particle_materials={
                1: OpticalConstants(
                    wavelength_um=np.array([0.3, 0.5]),
                    n=np.array([1.5, 1.48]),
                    k=np.array([0.01, 0.01]),
                )
            },
            matrix_materials={
                1: OpticalConstants(
                    wavelength_um=np.array([0.3, 0.5]),
                    n=np.array([1.4, 1.4]),
                    k=np.array([0.0, 0.0]),
                )
            },
            batches_completed=5,
            total_batches=10,
            photons_completed=50000,
            photons_target=100000,
        )

        data = original.to_dict()
        restored = SimulationSession.from_dict(data)

        assert restored.session_id == original.session_id
        assert restored.status == original.status
        assert restored.batches_completed == original.batches_completed


class TestSimulationManager:
    """Tests for SimulationManager class."""

    @pytest.fixture
    def mock_fos(self):
        """Create mock FOS wrapper."""
        from mcrt.fos.output_parser import FOSResult

        mock = MagicMock()
        mock.run_simple.return_value = FOSResult(
            wavelength_um=np.linspace(0.3, 2.5, 10),
            reflectance=np.random.uniform(0, 1, 10),
            absorptance=np.random.uniform(0, 1, 10),
            transmittance=np.random.uniform(0, 1, 10),
            n_photons=1000,
        )
        return mock

    def test_create_session(self, mock_fos):
        """Test creating a session through manager."""
        from mcrt.simulation.manager import SimulationManager
        from mcrt.materials.database import OpticalConstants

        manager = SimulationManager(mock_fos)

        particle_materials = {
            1: OpticalConstants(
                wavelength_um=np.array([0.3, 0.5]),
                n=np.array([1.5, 1.48]),
                k=np.array([0.01, 0.01]),
            )
        }
        matrix_materials = {
            1: OpticalConstants(
                wavelength_um=np.array([0.3, 0.5]),
                n=np.array([1.4, 1.4]),
                k=np.array([0.0, 0.0]),
            )
        }
        layers = [
            {
                "matrix_id": 1,
                "thickness_um": 100,
                "particles": [
                    {"material_id": 1, "diameter_um": 0.5, "volume_fraction": 10}
                ],
            }
        ]

        session = manager.create_session(
            particle_materials=particle_materials,
            matrix_materials=matrix_materials,
            layers=layers,
        )

        assert session.session_id is not None
        assert len(manager.list_sessions()) == 1

    def test_get_session(self, mock_fos):
        """Test getting a session."""
        from mcrt.simulation.manager import SimulationManager
        from mcrt.materials.database import OpticalConstants

        manager = SimulationManager(mock_fos)

        session = manager.create_session(
            particle_materials={
                1: OpticalConstants(
                    wavelength_um=np.array([0.3]),
                    n=np.array([1.5]),
                    k=np.array([0.01]),
                )
            },
            matrix_materials={
                1: OpticalConstants(
                    wavelength_um=np.array([0.3]),
                    n=np.array([1.4]),
                    k=np.array([0.0]),
                )
            },
            layers=[{"matrix_id": 1, "thickness_um": 100, "particles": []}],
        )

        retrieved = manager.get_session(session.session_id)
        assert retrieved.session_id == session.session_id

    def test_get_session_not_found(self, mock_fos):
        """Test getting non-existent session."""
        from mcrt.simulation.manager import SimulationManager

        manager = SimulationManager(mock_fos)

        with pytest.raises(KeyError):
            manager.get_session("nonexistent")

    def test_delete_session(self, mock_fos):
        """Test deleting a session."""
        from mcrt.simulation.manager import SimulationManager
        from mcrt.materials.database import OpticalConstants

        manager = SimulationManager(mock_fos)

        session = manager.create_session(
            particle_materials={
                1: OpticalConstants(
                    wavelength_um=np.array([0.3]),
                    n=np.array([1.5]),
                    k=np.array([0.01]),
                )
            },
            matrix_materials={
                1: OpticalConstants(
                    wavelength_um=np.array([0.3]),
                    n=np.array([1.4]),
                    k=np.array([0.0]),
                )
            },
            layers=[{"matrix_id": 1, "thickness_um": 100, "particles": []}],
        )

        manager.delete_session(session.session_id)

        with pytest.raises(KeyError):
            manager.get_session(session.session_id)


class TestStatePersistence:
    """Tests for state persistence."""

    def test_save_and_load_json(self, tmp_path):
        """Test saving and loading JSON state."""
        from mcrt.simulation.manager import SimulationSession, SimulationStatus
        from mcrt.simulation.state import StatePersistence
        from mcrt.materials.database import OpticalConstants

        session = SimulationSession(
            session_id="test-session",
            status=SimulationStatus.PAUSED,
            config={"test": "config"},
            particle_materials={
                1: OpticalConstants(
                    wavelength_um=np.array([0.3, 0.5]),
                    n=np.array([1.5, 1.48]),
                    k=np.array([0.01, 0.01]),
                )
            },
            matrix_materials={
                1: OpticalConstants(
                    wavelength_um=np.array([0.3, 0.5]),
                    n=np.array([1.4, 1.4]),
                    k=np.array([0.0, 0.0]),
                )
            },
            batches_completed=5,
            total_batches=10,
        )

        filepath = tmp_path / "state.json"
        StatePersistence.save_json(session, filepath)

        loaded = StatePersistence.load_json(filepath)

        assert loaded.session_id == session.session_id
        assert loaded.batches_completed == session.batches_completed

    def test_get_state_summary(self, tmp_path):
        """Test getting state summary."""
        from mcrt.simulation.manager import SimulationSession, SimulationStatus
        from mcrt.simulation.state import StatePersistence
        from mcrt.materials.database import OpticalConstants

        session = SimulationSession(
            session_id="summary-test",
            status=SimulationStatus.COMPLETED,
            config={},
            particle_materials={
                1: OpticalConstants(
                    wavelength_um=np.array([0.3]),
                    n=np.array([1.5]),
                    k=np.array([0.01]),
                )
            },
            matrix_materials={
                1: OpticalConstants(
                    wavelength_um=np.array([0.3]),
                    n=np.array([1.4]),
                    k=np.array([0.0]),
                )
            },
            batches_completed=10,
            total_batches=10,
            photons_completed=100000,
            photons_target=100000,
        )

        filepath = tmp_path / "state.json"
        StatePersistence.save_json(session, filepath)

        summary = StatePersistence.get_state_summary(filepath)

        assert summary["session_id"] == "summary-test"
        assert summary["status"] == "completed"
        assert summary["batches_completed"] == 10


class TestBatchUpdate:
    """Tests for BatchUpdate class."""

    def test_creation(self):
        """Test creating BatchUpdate."""
        from mcrt.simulation.progressive import BatchUpdate

        update = BatchUpdate(
            batch_number=5,
            total_batches=10,
            photons_completed=50000,
            photons_target=100000,
            wavelength_um=[0.3, 0.5, 0.7],
            reflectance=[0.5, 0.6, 0.7],
            absorptance=[0.3, 0.2, 0.1],
            transmittance=[0.2, 0.2, 0.2],
            timestamp="2024-01-01T00:00:00",
        )

        assert update.batch_number == 5
        assert update.total_batches == 10

    def test_to_dict(self):
        """Test converting to dictionary."""
        from mcrt.simulation.progressive import BatchUpdate

        update = BatchUpdate(
            batch_number=5,
            total_batches=10,
            photons_completed=50000,
            photons_target=100000,
            wavelength_um=[0.3, 0.5],
            reflectance=[0.5, 0.6],
            absorptance=[0.3, 0.2],
            transmittance=[0.2, 0.2],
            timestamp="2024-01-01T00:00:00",
        )

        data = update.to_dict()

        assert "batch_number" in data
        assert "progress_percent" in data
        assert data["progress_percent"] == 50.0
        assert "results" in data
