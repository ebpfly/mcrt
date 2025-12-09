"""Python wrapper for FOS (Fast Optical Spectrum) Monte Carlo code."""

import os
import sys
from pathlib import Path
from typing import Callable
import numpy as np

from mcrt.fos.input_builder import FOSInputBuilder, SimulationConfig
from mcrt.fos.output_parser import FOSOutputParser, FOSResult


class FOSWrapper:
    """Wrapper for running FOS simulations from Python.

    This class provides a clean Python interface to the FOS Monte Carlo
    radiative transfer code, handling input file generation, execution,
    and output parsing.
    """

    def __init__(
        self,
        fos_path: str | Path | None = None,
        work_dir: str | Path | None = None,
    ):
        """Initialize the FOS wrapper.

        Args:
            fos_path: Path to FOS source directory containing Main3.py
            work_dir: Working directory for input/output files
        """
        # Find FOS path
        if fos_path is None:
            # Try to find FOS relative to this package
            package_dir = Path(__file__).parent.parent.parent.parent.parent
            fos_path = package_dir / "fos" / "src"
            if not fos_path.exists():
                # Try current directory
                fos_path = Path.cwd() / "fos" / "src"

        self.fos_path = Path(fos_path)
        if not (self.fos_path / "Main3.py").exists():
            raise FileNotFoundError(
                f"FOS Main3.py not found at {self.fos_path}. "
                "Please provide the correct path to the FOS src directory."
            )

        # Set up working directory
        if work_dir is None:
            work_dir = Path.cwd() / ".mcrt_work"
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Input builder
        self.input_builder = FOSInputBuilder(self.work_dir)

        # Track if FOS has been imported
        self._fos_imported = False
        self._main_func = None

    def _import_fos(self):
        """Import FOS main_func on first use."""
        if self._fos_imported:
            return

        # Set matplotlib to use non-interactive backend BEFORE importing FOS
        # This prevents crashes when running in background threads (macOS NSWindow issue)
        import matplotlib
        matplotlib.use('Agg')

        # Add FOS path to sys.path temporarily
        fos_path_str = str(self.fos_path)
        if fos_path_str not in sys.path:
            sys.path.insert(0, fos_path_str)

        # Change to FOS directory (required for relative imports)
        old_cwd = os.getcwd()
        os.chdir(self.fos_path)

        try:
            from Main3 import main_func
            self._main_func = main_func
            self._fos_imported = True
        finally:
            # Restore working directory
            os.chdir(old_cwd)

    def run_simulation(self, config: SimulationConfig) -> FOSResult:
        """Run a FOS simulation with the given configuration.

        Args:
            config: Simulation configuration

        Returns:
            FOSResult with reflectance, absorptance, transmittance spectra
        """
        # This method requires material files to already exist
        # For full control, use run_with_materials instead
        raise NotImplementedError(
            "Use run_with_materials() to provide material data, "
            "or run_simple() for single-particle simulations."
        )

    def run_simple(
        self,
        particle_wavelength_um: np.ndarray,
        particle_n: np.ndarray,
        particle_k: np.ndarray,
        matrix_wavelength_um: np.ndarray,
        matrix_n: np.ndarray,
        matrix_k: np.ndarray,
        particle_diameter_um: float,
        particle_volume_fraction: float,
        layer_thickness_um: float,
        wavelength_start_um: float = 0.3,
        wavelength_end_um: float = 2.5,
        wavelength_interval_um: float = 0.01,
        n_photons: int = 10000,
        particle_std_dev: float = 0.0,
    ) -> FOSResult:
        """Run a simple single-layer, single-particle simulation.

        This is a convenience method for the most common use case.

        Args:
            particle_wavelength_um: Wavelength array for particle optical constants
            particle_n: Particle refractive index (real part)
            particle_k: Particle extinction coefficient (imaginary part)
            matrix_wavelength_um: Wavelength array for matrix optical constants
            matrix_n: Matrix refractive index (real part)
            matrix_k: Matrix extinction coefficient (imaginary part)
            particle_diameter_um: Particle diameter in micrometers
            particle_volume_fraction: Volume fraction as percentage (0-100)
            layer_thickness_um: Layer thickness in micrometers
            wavelength_start_um: Starting wavelength for simulation
            wavelength_end_um: Ending wavelength for simulation
            wavelength_interval_um: Wavelength step size
            n_photons: Number of photons per wavelength
            particle_std_dev: Standard deviation of particle size distribution

        Returns:
            FOSResult with simulation results
        """
        self._import_fos()

        # Generate unique prefix for this run
        import uuid
        output_prefix = f"sim_{uuid.uuid4().hex[:8]}"

        # Build input file
        input_path = self.input_builder.build_simple_simulation(
            particle_wavelength=particle_wavelength_um,
            particle_n=particle_n,
            particle_k=particle_k,
            matrix_wavelength=matrix_wavelength_um,
            matrix_n=matrix_n,
            matrix_k=matrix_k,
            particle_diameter_um=particle_diameter_um,
            particle_volume_fraction=particle_volume_fraction,
            layer_thickness_um=layer_thickness_um,
            wavelength_start_um=wavelength_start_um,
            wavelength_end_um=wavelength_end_um,
            wavelength_interval_um=wavelength_interval_um,
            n_photons=n_photons,
            particle_std_dev=particle_std_dev,
            output_prefix=output_prefix,
        )

        # Run FOS
        old_cwd = os.getcwd()
        os.chdir(self.work_dir)
        try:
            output_array = self._main_func(
                file_name=str(input_path),
                return_array=True,
            )
        finally:
            os.chdir(old_cwd)

        # Parse results
        result = FOSOutputParser.parse_array(output_array, n_photons=n_photons)

        # Clean up temporary files (optional)
        # self._cleanup(output_prefix)

        return result

    def run_batch(
        self,
        particle_wavelength_um: np.ndarray,
        particle_n: np.ndarray,
        particle_k: np.ndarray,
        matrix_wavelength_um: np.ndarray,
        matrix_n: np.ndarray,
        matrix_k: np.ndarray,
        particle_diameter_um: float,
        particle_volume_fraction: float,
        layer_thickness_um: float,
        wavelength_start_um: float = 0.3,
        wavelength_end_um: float = 2.5,
        wavelength_interval_um: float = 0.01,
        total_photons: int = 100000,
        n_batches: int = 10,
        particle_std_dev: float = 0.0,
        progress_callback: Callable[[int, int, FOSResult], None] | None = None,
    ) -> FOSResult:
        """Run simulation in batches with progress callbacks.

        This enables progressive updates of results during long simulations.

        Args:
            particle_wavelength_um: Wavelength array for particle optical constants
            particle_n: Particle refractive index (real part)
            particle_k: Particle extinction coefficient (imaginary part)
            matrix_wavelength_um: Wavelength array for matrix optical constants
            matrix_n: Matrix refractive index (real part)
            matrix_k: Matrix extinction coefficient (imaginary part)
            particle_diameter_um: Particle diameter in micrometers
            particle_volume_fraction: Volume fraction as percentage (0-100)
            layer_thickness_um: Layer thickness in micrometers
            wavelength_start_um: Starting wavelength for simulation
            wavelength_end_um: Ending wavelength for simulation
            wavelength_interval_um: Wavelength step size
            total_photons: Total number of photons across all batches
            n_batches: Number of batches to split the simulation into
            particle_std_dev: Standard deviation of particle size distribution
            progress_callback: Called after each batch with (batch_num, n_batches, accumulated_result)

        Returns:
            FOSResult with accumulated results from all batches
        """
        photons_per_batch = total_photons // n_batches
        batch_results: list[FOSResult] = []

        for i in range(n_batches):
            # Run batch
            result = self.run_simple(
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
            )

            batch_results.append(result)

            # Accumulate results so far
            accumulated = FOSResult.accumulate(batch_results)

            # Call progress callback
            if progress_callback is not None:
                progress_callback(i + 1, n_batches, accumulated)

        return FOSResult.accumulate(batch_results)

    def _cleanup(self, prefix: str):
        """Clean up temporary files."""
        for f in self.work_dir.glob(f"{prefix}*"):
            try:
                f.unlink()
            except OSError:
                pass
