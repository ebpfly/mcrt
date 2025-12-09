"""Build FOS input files from Python configuration objects."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import numpy as np


@dataclass
class ParticleConfig:
    """Configuration for a particle type within a layer.

    Attributes:
        material_id: Reference to material file (1-indexed in FOS)
        diameters_um: Particle diameter(s) in micrometers
        volume_fractions: Volume fraction(s) as percentages (0-100)
        std_devs: Standard deviation(s) of particle size distribution
        is_core: If True, this is a core particle (for core-shell)
        is_shell: If True, this is a shell particle (for core-shell)
        shell_thickness_um: Shell wall thickness in micrometers (for shells)
    """

    material_id: int
    diameters_um: list[float] | float = 0.5
    volume_fractions: list[float] | float = 10.0
    std_devs: list[float] | float = 0.0
    is_core: bool = False
    is_shell: bool = False
    shell_thickness_um: float | None = None

    def __post_init__(self):
        # Convert single values to lists for consistency
        if isinstance(self.diameters_um, (int, float)):
            self.diameters_um = [float(self.diameters_um)]
        if isinstance(self.volume_fractions, (int, float)):
            self.volume_fractions = [float(self.volume_fractions)]
        if isinstance(self.std_devs, (int, float)):
            self.std_devs = [float(self.std_devs)]


@dataclass
class LayerConfig:
    """Configuration for a single layer in the medium.

    Attributes:
        matrix_id: Reference to matrix material file (1-indexed)
        thickness_um: Layer thickness in micrometers
        particles: List of particle configurations in this layer
    """

    matrix_id: int
    thickness_um: float
    particles: list[ParticleConfig] = field(default_factory=list)


@dataclass
class SimulationConfig:
    """Complete configuration for a FOS simulation.

    Attributes:
        layers: List of layer configurations (top to bottom)
        wavelength_start_um: Starting wavelength in micrometers
        wavelength_end_um: Ending wavelength in micrometers
        wavelength_interval_um: Wavelength step size in micrometers
        n_photons: Number of photons per wavelength
        upper_boundary_matrix_id: Matrix ID for upper boundary (None = air)
        lower_boundary_matrix_id: Matrix ID for lower boundary (None = air)
        output_prefix: Prefix for output files
        use_neural_network: If True, use NN instead of MC
    """

    layers: list[LayerConfig]
    wavelength_start_um: float = 0.3
    wavelength_end_um: float = 2.5
    wavelength_interval_um: float = 0.01
    n_photons: int = 10000
    upper_boundary_matrix_id: int | None = None
    lower_boundary_matrix_id: int | None = None
    output_prefix: str = "simulation"
    use_neural_network: bool = False
    seed: int | None = None

    @property
    def n_wavelengths(self) -> int:
        """Number of wavelength points."""
        return int((self.wavelength_end_um - self.wavelength_start_um) / self.wavelength_interval_um) + 1


class FOSInputBuilder:
    """Builds FOS input files from configuration objects."""

    def __init__(self, work_dir: str | Path):
        """Initialize the input builder.

        Args:
            work_dir: Working directory for input/output files
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def build_material_file(
        self,
        wavelength_um: np.ndarray,
        n: np.ndarray,
        k: np.ndarray,
        filename: str,
    ) -> Path:
        """Create a material file in FOS format.

        Args:
            wavelength_um: Wavelength array in micrometers
            n: Real part of refractive index
            k: Imaginary part of refractive index (extinction coefficient)
            filename: Output filename (without path)

        Returns:
            Path to the created material file
        """
        filepath = self.work_dir / filename
        data = np.column_stack([wavelength_um, n, k])
        np.savetxt(filepath, data, fmt="%.6g", delimiter="\t")
        return filepath

    def build_input_file(
        self,
        config: SimulationConfig,
        particle_files: dict[int, str],
        matrix_files: dict[int, str],
        solar_file: str | None = None,
    ) -> Path:
        """Build a complete FOS input file.

        Args:
            config: Simulation configuration
            particle_files: Mapping from particle ID to filename
            matrix_files: Mapping from matrix ID to filename
            solar_file: Optional solar spectrum file

        Returns:
            Path to the created input file
        """
        lines = []

        # Simulation mode
        lines.append("MC" if not config.use_neural_network else "NN")
        lines.append("")

        # Output prefix
        lines.append(f"Output: {config.output_prefix}")
        lines.append("")

        # Material files
        for pid, pfile in sorted(particle_files.items()):
            lines.append(f"Particle {pid}: {pfile}")

        for mid, mfile in sorted(matrix_files.items()):
            lines.append(f"Matrix {mid}: {mfile}")

        lines.append("")

        # Solar file (optional)
        if solar_file:
            lines.append(f"Solar: {solar_file}")
            lines.append("")

        # Photon count
        lines.append(f"Photons: {config.n_photons}")
        lines.append("")

        # Wavelength range
        lines.append(f"Start: {config.wavelength_start_um}")
        lines.append(f"End: {config.wavelength_end_um}")
        lines.append(f"Interval: {config.wavelength_interval_um}")
        lines.append("")

        # Simulation body
        lines.append("### Body ###")
        lines.append("")
        lines.append("Sim 1")

        # Boundary conditions
        if config.upper_boundary_matrix_id is not None:
            lines.append(f"Upper: Matrix {config.upper_boundary_matrix_id}")
        if config.lower_boundary_matrix_id is not None:
            lines.append(f"Lower: Matrix {config.lower_boundary_matrix_id}")

        # Layers
        for i, layer in enumerate(config.layers, start=1):
            lines.append(f"Layer {i}")
            lines.append(f"Matrix {layer.matrix_id}")
            lines.append(f"T: {layer.thickness_um}")

            # Particles in this layer
            for particle in layer.particles:
                lines.append(f"Particle {particle.material_id}")

                if particle.is_core:
                    # Core particle
                    lines.append(f"C: {particle.diameters_um[0]}")
                elif particle.is_shell:
                    # Shell particle
                    lines.append(f"S: {particle.shell_thickness_um}")
                    lines.append(f"VF: {particle.volume_fractions[0]}")
                    if particle.std_devs[0] > 0:
                        lines.append(f"Std: {particle.std_devs[0]}")
                else:
                    # Regular particle
                    if len(particle.diameters_um) == 1:
                        lines.append(f"D: {particle.diameters_um[0]}")
                        lines.append(f"VF: {particle.volume_fractions[0]}")
                        if particle.std_devs[0] > 0:
                            lines.append(f"Std: {particle.std_devs[0]}")
                    else:
                        # Multiple sizes
                        d_str = ", ".join(str(d) for d in particle.diameters_um)
                        vf_str = ", ".join(str(vf) for vf in particle.volume_fractions)
                        std_str = ", ".join(str(s) for s in particle.std_devs)
                        lines.append(f"D: {d_str}")
                        lines.append(f"VF: {vf_str}")
                        if any(s > 0 for s in particle.std_devs):
                            lines.append(f"Std: {std_str}")

        # Write to file
        input_path = self.work_dir / f"{config.output_prefix}_input.txt"
        with open(input_path, "w") as f:
            f.write("\n".join(lines))

        return input_path

    def build_simple_simulation(
        self,
        particle_wavelength: np.ndarray,
        particle_n: np.ndarray,
        particle_k: np.ndarray,
        matrix_wavelength: np.ndarray,
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
        output_prefix: str = "simple_sim",
    ) -> Path:
        """Build a simple single-layer, single-particle simulation.

        This is a convenience method for the most common use case.

        Returns:
            Path to the created input file
        """
        # Create material files
        particle_file = self.build_material_file(
            particle_wavelength, particle_n, particle_k, f"{output_prefix}_particle.txt"
        )
        matrix_file = self.build_material_file(
            matrix_wavelength, matrix_n, matrix_k, f"{output_prefix}_matrix.txt"
        )

        # Build configuration
        particle_config = ParticleConfig(
            material_id=1,
            diameters_um=particle_diameter_um,
            volume_fractions=particle_volume_fraction,
            std_devs=particle_std_dev,
        )

        layer_config = LayerConfig(
            matrix_id=1,
            thickness_um=layer_thickness_um,
            particles=[particle_config],
        )

        sim_config = SimulationConfig(
            layers=[layer_config],
            wavelength_start_um=wavelength_start_um,
            wavelength_end_um=wavelength_end_um,
            wavelength_interval_um=wavelength_interval_um,
            n_photons=n_photons,
            output_prefix=output_prefix,
        )

        # Build input file
        return self.build_input_file(
            sim_config,
            particle_files={1: particle_file.name},
            matrix_files={1: matrix_file.name},
        )
