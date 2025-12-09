"""Simulation session management."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable
import uuid

import numpy as np

from mcrt.fos.wrapper import FOSWrapper
from mcrt.fos.input_builder import SimulationConfig, LayerConfig, ParticleConfig
from mcrt.fos.output_parser import FOSResult
from mcrt.materials.database import OpticalConstants


class SimulationStatus(Enum):
    """Status of a simulation session."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SimulationSession:
    """Manages a single simulation session.

    Attributes:
        session_id: Unique session identifier
        status: Current simulation status
        config: Simulation configuration
        particle_materials: Optical constants for particles
        matrix_materials: Optical constants for matrices
        current_result: Current accumulated result
        batches_completed: Number of batches completed
        total_batches: Total number of batches
        photons_completed: Total photons simulated
        photons_target: Target photon count
        created_at: Session creation timestamp
        updated_at: Last update timestamp
        error_message: Error message if status is ERROR
    """

    session_id: str
    status: SimulationStatus
    config: dict  # Serialized config for persistence
    particle_materials: dict[int, OpticalConstants]
    matrix_materials: dict[int, OpticalConstants]
    current_result: FOSResult | None = None
    batches_completed: int = 0
    total_batches: int = 10
    photons_completed: int = 0
    photons_target: int = 100000
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    error_message: str | None = None

    @classmethod
    def create(
        cls,
        particle_materials: dict[int, OpticalConstants],
        matrix_materials: dict[int, OpticalConstants],
        layers: list[dict],
        wavelength_start_um: float = 0.3,
        wavelength_end_um: float = 2.5,
        wavelength_interval_um: float = 0.01,
        photons_target: int = 100000,
        total_batches: int = 10,
    ) -> "SimulationSession":
        """Create a new simulation session.

        Args:
            particle_materials: Dict mapping particle ID to OpticalConstants
            matrix_materials: Dict mapping matrix ID to OpticalConstants
            layers: Layer configuration dicts
            wavelength_start_um: Starting wavelength
            wavelength_end_um: Ending wavelength
            wavelength_interval_um: Wavelength step
            photons_target: Total photons to simulate
            total_batches: Number of batches

        Returns:
            New SimulationSession
        """
        config = {
            "layers": layers,
            "wavelength_start_um": wavelength_start_um,
            "wavelength_end_um": wavelength_end_um,
            "wavelength_interval_um": wavelength_interval_um,
            "photons_target": photons_target,
            "total_batches": total_batches,
        }

        return cls(
            session_id=str(uuid.uuid4()),
            status=SimulationStatus.IDLE,
            config=config,
            particle_materials=particle_materials,
            matrix_materials=matrix_materials,
            total_batches=total_batches,
            photons_target=photons_target,
        )

    @property
    def progress(self) -> float:
        """Progress as fraction from 0 to 1."""
        if self.photons_target == 0:
            return 0.0
        return self.photons_completed / self.photons_target

    @property
    def progress_percent(self) -> float:
        """Progress as percentage from 0 to 100."""
        return self.progress * 100

    def to_dict(self) -> dict:
        """Convert to dictionary for persistence/serialization."""
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "config": self.config,
            "batches_completed": self.batches_completed,
            "total_batches": self.total_batches,
            "photons_completed": self.photons_completed,
            "photons_target": self.photons_target,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "error_message": self.error_message,
            "current_result": self.current_result.to_dict() if self.current_result else None,
            "particle_materials": {
                k: {
                    "wavelength_um": v.wavelength_um.tolist(),
                    "n": v.n.tolist(),
                    "k": v.k.tolist(),
                }
                for k, v in self.particle_materials.items()
            },
            "matrix_materials": {
                k: {
                    "wavelength_um": v.wavelength_um.tolist(),
                    "n": v.n.tolist(),
                    "k": v.k.tolist(),
                }
                for k, v in self.matrix_materials.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationSession":
        """Create from dictionary."""
        particle_materials = {
            int(k): OpticalConstants(
                wavelength_um=np.array(v["wavelength_um"]),
                n=np.array(v["n"]),
                k=np.array(v["k"]),
            )
            for k, v in data["particle_materials"].items()
        }
        matrix_materials = {
            int(k): OpticalConstants(
                wavelength_um=np.array(v["wavelength_um"]),
                n=np.array(v["n"]),
                k=np.array(v["k"]),
            )
            for k, v in data["matrix_materials"].items()
        }

        current_result = None
        if data.get("current_result"):
            current_result = FOSResult.from_dict(data["current_result"])

        return cls(
            session_id=data["session_id"],
            status=SimulationStatus(data["status"]),
            config=data["config"],
            particle_materials=particle_materials,
            matrix_materials=matrix_materials,
            current_result=current_result,
            batches_completed=data["batches_completed"],
            total_batches=data["total_batches"],
            photons_completed=data["photons_completed"],
            photons_target=data["photons_target"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            error_message=data.get("error_message"),
        )


class SimulationManager:
    """Manages multiple simulation sessions."""

    def __init__(self, fos_wrapper: FOSWrapper):
        """Initialize the manager.

        Args:
            fos_wrapper: FOSWrapper instance for running simulations
        """
        self.fos = fos_wrapper
        self._sessions: dict[str, SimulationSession] = {}
        self._stop_flags: dict[str, bool] = {}

    def create_session(
        self,
        particle_materials: dict[int, OpticalConstants],
        matrix_materials: dict[int, OpticalConstants],
        layers: list[dict],
        wavelength_start_um: float = 0.3,
        wavelength_end_um: float = 2.5,
        wavelength_interval_um: float = 0.01,
        photons_target: int = 100000,
        total_batches: int = 10,
    ) -> SimulationSession:
        """Create a new simulation session."""
        session = SimulationSession.create(
            particle_materials=particle_materials,
            matrix_materials=matrix_materials,
            layers=layers,
            wavelength_start_um=wavelength_start_um,
            wavelength_end_um=wavelength_end_um,
            wavelength_interval_um=wavelength_interval_um,
            photons_target=photons_target,
            total_batches=total_batches,
        )
        self._sessions[session.session_id] = session
        self._stop_flags[session.session_id] = False
        return session

    def get_session(self, session_id: str) -> SimulationSession:
        """Get a session by ID."""
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        return self._sessions[session_id]

    def list_sessions(self) -> list[SimulationSession]:
        """List all sessions."""
        return list(self._sessions.values())

    def delete_session(self, session_id: str):
        """Delete a session."""
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        del self._sessions[session_id]
        if session_id in self._stop_flags:
            del self._stop_flags[session_id]

    def stop_session(self, session_id: str):
        """Signal a running session to stop."""
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        if session_id in self._stop_flags:
            self._stop_flags[session_id] = True

    def restore_session(self, data: dict) -> SimulationSession:
        """Restore a session from saved state."""
        session = SimulationSession.from_dict(data)
        self._sessions[session.session_id] = session
        self._stop_flags[session.session_id] = False
        return session

    def run_batches(
        self,
        session_id: str,
        n_batches: int | None = None,
        progress_callback: Callable[[SimulationSession], None] | None = None,
    ) -> SimulationSession:
        """Run simulation batches for a session.

        Args:
            session_id: Session to run
            n_batches: Number of batches to run (None = run remaining)
            progress_callback: Called after each batch with updated session

        Returns:
            Updated session
        """
        session = self.get_session(session_id)

        if session.status == SimulationStatus.COMPLETED:
            return session

        # Calculate batches to run
        remaining = session.total_batches - session.batches_completed
        if n_batches is None:
            n_batches = remaining
        else:
            n_batches = min(n_batches, remaining)

        if n_batches <= 0:
            session.status = SimulationStatus.COMPLETED
            return session

        # Set up batch parameters
        photons_per_batch = session.photons_target // session.total_batches
        batch_results: list[FOSResult] = []

        # Include previous results if any
        if session.current_result is not None:
            # Weight existing results appropriately
            batch_results = [session.current_result] * session.batches_completed

        session.status = SimulationStatus.RUNNING
        self._stop_flags[session_id] = False

        try:
            config = session.config

            # Run batches
            for i in range(n_batches):
                # Check stop flag
                if self._stop_flags.get(session_id, False):
                    session.status = SimulationStatus.PAUSED
                    break

                # Run single batch using the simple interface for single-layer
                # For multi-layer, we'd need more complex handling
                layers = config["layers"]
                if len(layers) == 1 and len(layers[0]["particles"]) == 1:
                    # Simple single-layer, single-particle case
                    layer = layers[0]
                    particle = layer["particles"][0]

                    particle_mat = session.particle_materials[particle["material_id"]]
                    matrix_mat = session.matrix_materials[layer["matrix_id"]]

                    result = self.fos.run_simple(
                        particle_wavelength_um=particle_mat.wavelength_um,
                        particle_n=particle_mat.n,
                        particle_k=particle_mat.k,
                        matrix_wavelength_um=matrix_mat.wavelength_um,
                        matrix_n=matrix_mat.n,
                        matrix_k=matrix_mat.k,
                        particle_diameter_um=particle["diameter_um"],
                        particle_volume_fraction=particle["volume_fraction"],
                        layer_thickness_um=layer["thickness_um"],
                        wavelength_start_um=config["wavelength_start_um"],
                        wavelength_end_um=config["wavelength_end_um"],
                        wavelength_interval_um=config["wavelength_interval_um"],
                        n_photons=photons_per_batch,
                        particle_std_dev=particle.get("std_dev", 0.0),
                    )
                else:
                    # Multi-layer not yet implemented
                    raise NotImplementedError(
                        "Multi-layer simulations not yet implemented in manager"
                    )

                batch_results.append(result)
                session.batches_completed += 1
                session.photons_completed += photons_per_batch
                session.current_result = FOSResult.accumulate(batch_results)
                session.updated_at = datetime.now()

                if progress_callback:
                    progress_callback(session)

            # Check if completed
            if session.batches_completed >= session.total_batches:
                session.status = SimulationStatus.COMPLETED
            elif session.status == SimulationStatus.RUNNING:
                session.status = SimulationStatus.PAUSED

        except Exception as e:
            session.status = SimulationStatus.ERROR
            session.error_message = str(e)
            raise

        return session
