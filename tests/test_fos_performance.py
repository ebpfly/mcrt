"""Test FOS performance with different wavelength/photon configurations.

The goal is to ensure that 700 wavelengths × 100 photons doesn't take
more than 15x the time of 70 wavelengths × 10K photons.
"""

import time
import sys
sys.path.insert(0, '/Users/eric/repo/mcrt/src')

from mcrt.fos.wrapper import FOSWrapper
from mcrt.materials.database import MaterialDatabase
import numpy as np


def run_simulation(
    wrapper: FOSWrapper,
    particle_wavelength: np.ndarray,
    particle_n: np.ndarray,
    particle_k: np.ndarray,
    wavelength_start: float,
    wavelength_end: float,
    wavelength_interval: float,
    n_photons: int,
) -> float:
    """Run a simulation and return the time taken in seconds."""

    # Air matrix (n=1.0, k=0)
    matrix_wavelength = np.array([0.3, 16.0])
    matrix_n = np.array([1.0, 1.0])
    matrix_k = np.array([0.0, 0.0])

    # Default frontend settings
    particle_diameter_um = 10.0
    particle_volume_fraction = 50.0
    layer_thickness_um = 100.0
    particle_std_dev = 0.5

    start_time = time.perf_counter()

    result = wrapper.run_simple(
        particle_wavelength_um=particle_wavelength,
        particle_n=particle_n,
        particle_k=particle_k,
        matrix_wavelength_um=matrix_wavelength,
        matrix_n=matrix_n,
        matrix_k=matrix_k,
        particle_diameter_um=particle_diameter_um,
        particle_volume_fraction=particle_volume_fraction,
        layer_thickness_um=layer_thickness_um,
        wavelength_start_um=wavelength_start,
        wavelength_end_um=wavelength_end,
        wavelength_interval_um=wavelength_interval,
        n_photons=n_photons,
        particle_std_dev=particle_std_dev,
    )

    elapsed = time.perf_counter() - start_time
    return elapsed, result


def test_performance_ratio():
    """Test that 700 wavelengths doesn't take more than 15x the time of 70 wavelengths."""

    # Load kaolinite material (default particle)
    db = MaterialDatabase()
    kaolinite = db.get_optical_constants("other/kaolinite/Querry")

    # Create FOS wrapper
    wrapper = FOSWrapper()

    particle_wavelength = kaolinite.wavelength_um
    particle_n = kaolinite.n
    particle_k = kaolinite.k

    print("=" * 60)
    print("FOS Performance Test")
    print("=" * 60)
    print(f"Particle: kaolinite ({len(particle_wavelength)} data points)")
    print(f"Particle diameter: 10 µm")
    print(f"Volume fraction: 50%")
    print(f"Size distribution std_dev: 0.5")
    print(f"Layer thickness: 100 µm")
    print(f"Matrix: air (n=1.0)")
    print("=" * 60)

    # Test 1: 70 wavelengths × 10K photons
    # Wavelength range 7-14 µm with 0.1 µm interval = 71 wavelengths
    print("\nTest 1: 70 wavelengths × 10K photons")
    print("-" * 40)

    time_70wl, result1 = run_simulation(
        wrapper=wrapper,
        particle_wavelength=particle_wavelength,
        particle_n=particle_n,
        particle_k=particle_k,
        wavelength_start=7.0,
        wavelength_end=14.0,
        wavelength_interval=0.1,  # 71 wavelengths
        n_photons=10000,
    )

    n_wl_1 = len(result1.wavelength_um)
    print(f"Wavelengths: {n_wl_1}")
    print(f"Photons per wavelength: 10,000")
    print(f"Total photon-wavelengths: {n_wl_1 * 10000:,}")
    print(f"Time: {time_70wl:.2f} seconds")

    # Test 2: 700 wavelengths × 100 photons
    # Wavelength range 7-14 µm with 0.01 µm interval = 701 wavelengths
    print("\nTest 2: 700 wavelengths × 100 photons")
    print("-" * 40)

    time_700wl, result2 = run_simulation(
        wrapper=wrapper,
        particle_wavelength=particle_wavelength,
        particle_n=particle_n,
        particle_k=particle_k,
        wavelength_start=7.0,
        wavelength_end=14.0,
        wavelength_interval=0.01,  # 701 wavelengths
        n_photons=100,
    )

    n_wl_2 = len(result2.wavelength_um)
    print(f"Wavelengths: {n_wl_2}")
    print(f"Photons per wavelength: 100")
    print(f"Total photon-wavelengths: {n_wl_2 * 100:,}")
    print(f"Time: {time_700wl:.2f} seconds")

    # Calculate ratio
    ratio = time_700wl / time_70wl

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Time ratio (700wl / 70wl): {ratio:.2f}x")
    print(f"Maximum allowed ratio: 15x")
    print()

    if ratio <= 15:
        print("✓ PASS: Performance ratio is acceptable")
        return True
    else:
        print(f"✗ FAIL: Performance ratio {ratio:.2f}x exceeds 15x limit")
        return False


if __name__ == "__main__":
    success = test_performance_ratio()
    sys.exit(0 if success else 1)
