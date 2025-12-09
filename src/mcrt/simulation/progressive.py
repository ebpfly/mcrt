"""Progressive simulation execution with async support."""

import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator, Callable
from datetime import datetime

from mcrt.fos.wrapper import FOSWrapper
from mcrt.fos.output_parser import FOSResult
from mcrt.materials.database import OpticalConstants
from mcrt.simulation.manager import SimulationSession, SimulationStatus


@dataclass
class BatchUpdate:
    """Update event from a batch completion.

    Attributes:
        batch_number: Current batch number (1-indexed)
        total_batches: Total number of batches
        photons_completed: Total photons completed so far
        photons_target: Target photon count
        wavelength_um: Wavelength array
        reflectance: Current reflectance estimate
        absorptance: Current absorptance estimate
        transmittance: Current transmittance estimate
        timestamp: When this update was generated
    """

    batch_number: int
    total_batches: int
    photons_completed: int
    photons_target: int
    wavelength_um: list[float]
    reflectance: list[float]
    absorptance: list[float]
    transmittance: list[float]
    timestamp: str

    @classmethod
    def from_session(cls, session: SimulationSession) -> "BatchUpdate":
        """Create update from session state."""
        result = session.current_result
        return cls(
            batch_number=session.batches_completed,
            total_batches=session.total_batches,
            photons_completed=session.photons_completed,
            photons_target=session.photons_target,
            wavelength_um=result.wavelength_um.tolist() if result else [],
            reflectance=result.reflectance.tolist() if result else [],
            absorptance=result.absorptance.tolist() if result else [],
            transmittance=result.transmittance.tolist() if result else [],
            timestamp=datetime.now().isoformat(),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "batch_number": self.batch_number,
            "total_batches": self.total_batches,
            "photons_completed": self.photons_completed,
            "photons_target": self.photons_target,
            "progress_percent": (
                100 * self.photons_completed / self.photons_target
                if self.photons_target > 0
                else 0
            ),
            "results": {
                "wavelength_um": self.wavelength_um,
                "reflectance": self.reflectance,
                "absorptance": self.absorptance,
                "transmittance": self.transmittance,
            },
            "timestamp": self.timestamp,
        }


class ProgressiveSimulation:
    """Run simulations progressively with streaming updates.

    This class wraps FOSWrapper to provide:
    - Async execution with batch updates
    - Stop/pause capability
    - Progress tracking
    """

    def __init__(self, fos_wrapper: FOSWrapper):
        """Initialize with FOSWrapper.

        Args:
            fos_wrapper: Configured FOSWrapper instance
        """
        self.fos = fos_wrapper
        self._stop_requested = False

    def stop(self):
        """Request simulation to stop after current batch."""
        self._stop_requested = True

    def reset(self):
        """Reset stop flag for new simulation."""
        self._stop_requested = False

    async def run_simple_progressive(
        self,
        particle_wavelength_um,
        particle_n,
        particle_k,
        matrix_wavelength_um,
        matrix_n,
        matrix_k,
        particle_diameter_um: float,
        particle_volume_fraction: float,
        layer_thickness_um: float,
        wavelength_start_um: float = 0.3,
        wavelength_end_um: float = 2.5,
        wavelength_interval_um: float = 0.01,
        total_photons: int = 100000,
        n_batches: int = 10,
        particle_std_dev: float = 0.0,
    ) -> AsyncGenerator[BatchUpdate, None]:
        """Run a simple simulation progressively, yielding updates.

        This is an async generator that yields BatchUpdate objects
        as each batch completes.

        Args:
            Same as FOSWrapper.run_simple, plus:
            total_photons: Total photons across all batches
            n_batches: Number of batches

        Yields:
            BatchUpdate objects after each batch
        """
        self._stop_requested = False
        photons_per_batch = total_photons // n_batches
        batch_results: list[FOSResult] = []

        for i in range(n_batches):
            if self._stop_requested:
                break

            # Run batch in executor to not block event loop
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.fos.run_simple(
                    particle_wavelength_um=particle_wavelength_um,
                    particle_n=particle_n,
                    particle_k=particle_k,
                    matrix_wavelength_um=matrix_wavelength_um,
                    matrix_n=matrix_n,
                    matrix_k=matrix_k,
                    particle_diameter_um=particle_diameter_um,
                    particle_volume_fraction=particle_volume_fraction,
                    layer_thickness_um=layer_thickness_um,
                    wavelength_start_um=wavelength_start_um,
                    wavelength_end_um=wavelength_end_um,
                    wavelength_interval_um=wavelength_interval_um,
                    n_photons=photons_per_batch,
                    particle_std_dev=particle_std_dev,
                ),
            )

            batch_results.append(result)
            accumulated = FOSResult.accumulate(batch_results)

            update = BatchUpdate(
                batch_number=i + 1,
                total_batches=n_batches,
                photons_completed=(i + 1) * photons_per_batch,
                photons_target=total_photons,
                wavelength_um=accumulated.wavelength_um.tolist(),
                reflectance=accumulated.reflectance.tolist(),
                absorptance=accumulated.absorptance.tolist(),
                transmittance=accumulated.transmittance.tolist(),
                timestamp=datetime.now().isoformat(),
            )

            yield update

            # Small delay to allow other tasks
            await asyncio.sleep(0.01)

    async def run_session_progressive(
        self,
        session: SimulationSession,
        n_batches: int | None = None,
    ) -> AsyncGenerator[BatchUpdate, None]:
        """Run batches for a session progressively.

        Args:
            session: SimulationSession to run
            n_batches: Number of batches to run (None = run remaining)

        Yields:
            BatchUpdate objects after each batch
        """
        if session.status == SimulationStatus.COMPLETED:
            return

        self._stop_requested = False

        # Calculate batches to run
        remaining = session.total_batches - session.batches_completed
        if n_batches is None:
            n_batches = remaining
        else:
            n_batches = min(n_batches, remaining)

        if n_batches <= 0:
            session.status = SimulationStatus.COMPLETED
            return

        photons_per_batch = session.photons_target // session.total_batches
        batch_results: list[FOSResult] = []

        # Include previous results weighted appropriately
        if session.current_result is not None and session.batches_completed > 0:
            batch_results = [session.current_result] * session.batches_completed

        session.status = SimulationStatus.RUNNING
        config = session.config

        try:
            for i in range(n_batches):
                if self._stop_requested:
                    session.status = SimulationStatus.PAUSED
                    break

                # Get materials and layer config
                layers = config["layers"]
                if len(layers) != 1 or len(layers[0]["particles"]) != 1:
                    raise NotImplementedError("Only single-layer, single-particle supported")

                layer = layers[0]
                particle = layer["particles"][0]

                particle_mat = session.particle_materials[particle["material_id"]]
                matrix_mat = session.matrix_materials[layer["matrix_id"]]

                # Run in executor
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.fos.run_simple(
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
                    ),
                )

                batch_results.append(result)
                session.batches_completed += 1
                session.photons_completed += photons_per_batch
                session.current_result = FOSResult.accumulate(batch_results)
                session.updated_at = datetime.now()

                yield BatchUpdate.from_session(session)

                await asyncio.sleep(0.01)

            # Update final status
            if session.batches_completed >= session.total_batches:
                session.status = SimulationStatus.COMPLETED
            elif session.status == SimulationStatus.RUNNING:
                session.status = SimulationStatus.PAUSED

        except Exception as e:
            session.status = SimulationStatus.ERROR
            session.error_message = str(e)
            raise
