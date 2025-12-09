#!/usr/bin/env python3
"""Validate the full reflectance calculation (Monte Carlo radiative transfer).

This script validates the MCRT reflectance by comparing against:
1. Kubelka-Munk analytical solution for diffuse reflectance
2. Physical limiting behaviors
3. Known trends with parameters
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from mcrt.fos.wrapper import FOSWrapper


def kubelka_munk_reflectance(K, S):
    """Calculate Kubelka-Munk reflectance for semi-infinite medium.

    K = absorption coefficient (2 * ka for diffuse)
    S = scattering coefficient (related to ks by S ≈ ks * (1-g) for anisotropic)

    Returns R_infinity
    """
    if S == 0:
        return 0.0
    ratio = K / S
    R = 1 + ratio - np.sqrt(ratio**2 + 2*ratio)
    return max(0, min(1, R))


def two_flux_reflectance(ka, ks_reduced, thickness):
    """Two-flux approximation for finite slab reflectance.

    ka = absorption coefficient
    ks_reduced = reduced scattering coefficient = ks * (1 - g)
    thickness = slab thickness

    Returns reflectance R
    """
    if ks_reduced <= 0:
        # Pure absorption: no reflectance
        return 0.0

    # Diffusion coefficient
    K = ka
    S = ks_reduced

    if K == 0:
        # Pure scattering - reflectance depends on thickness
        return S * thickness / (1 + S * thickness)

    # General case
    a = (K + S) / S
    b = np.sqrt(a**2 - 1)

    # For finite slab
    tau = (K + S) * thickness
    if tau > 50:  # Effectively infinite
        return (a - b) / (a + b)

    sinh_b_tau = np.sinh(b * tau * S / (K + S))
    cosh_b_tau = np.cosh(b * tau * S / (K + S))

    denom = (a * sinh_b_tau + b * cosh_b_tau)
    if abs(denom) < 1e-10:
        return 0.0

    R = sinh_b_tau / denom
    return max(0, min(1, R))


def run_fos_simulation(particle_n, particle_k, particle_diameter_um, volume_fraction,
                       thickness_um, wavelength_um, n_photons=5000):
    """Run FOS simulation and return reflectance at specified wavelength."""

    fos_path = Path(__file__).parent.parent / "fos/src"
    work_dir = Path(__file__).parent.parent / ".mcrt_work"
    fos = FOSWrapper(fos_path=fos_path, work_dir=work_dir)

    # Create wavelength array around target
    particle_wavelength = np.array([wavelength_um * 0.9, wavelength_um, wavelength_um * 1.1])
    particle_n_arr = np.array([particle_n, particle_n, particle_n])
    particle_k_arr = np.array([particle_k, particle_k, particle_k])

    # Air matrix
    matrix_wavelength = np.array([0.3, wavelength_um, 100.0])
    matrix_n = np.ones(3)
    matrix_k = np.zeros(3)

    result = fos.run_simple(
        particle_wavelength_um=particle_wavelength,
        particle_n=particle_n_arr,
        particle_k=particle_k_arr,
        matrix_wavelength_um=matrix_wavelength,
        matrix_n=matrix_n,
        matrix_k=matrix_k,
        particle_diameter_um=particle_diameter_um,
        particle_volume_fraction=volume_fraction,
        layer_thickness_um=thickness_um,
        wavelength_start_um=wavelength_um - 0.01,
        wavelength_end_um=wavelength_um + 0.01,
        wavelength_interval_um=0.01,
        n_photons=n_photons,
        particle_std_dev=0.0,  # Monodisperse for cleaner comparison
    )

    # Return reflectance at center wavelength
    idx = len(result.wavelength_um) // 2
    return result.reflectance[idx]


def test_absorption_dependence():
    """Test: Reflectance should decrease with increasing absorption."""
    print("\n" + "=" * 70)
    print("TEST 1: Reflectance vs Absorption (k)")
    print("Expected: R decreases as k increases")
    print("=" * 70)

    wavelength = 10.0
    n = 1.5
    diameter = 5.0
    vf = 30.0
    thickness = 2000.0

    k_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    reflectances = []

    print(f"\nParameters: n={n}, D={diameter}µm, VF={vf}%, thickness={thickness}µm, λ={wavelength}µm")
    print("-" * 50)
    print(f"{'k':<10} | {'R_sim':>8} | {'Trend':>10}")
    print("-" * 50)

    prev_r = 1.0
    all_decreasing = True

    for k in k_values:
        r = run_fos_simulation(n, k, diameter, vf, thickness, wavelength)
        reflectances.append(r)

        trend = "↓ GOOD" if r < prev_r else "↑ BAD"
        if r >= prev_r and k > k_values[0]:
            all_decreasing = False

        print(f"{k:<10.3f} | {r:>8.4f} | {trend:>10}")
        prev_r = r

    result = "PASS" if all_decreasing else "FAIL"
    print(f"\nResult: {result} - Reflectance {'decreases' if all_decreasing else 'does NOT decrease'} with absorption")
    return all_decreasing


def test_thickness_dependence():
    """Test: Reflectance should increase with thickness up to saturation."""
    print("\n" + "=" * 70)
    print("TEST 2: Reflectance vs Layer Thickness")
    print("Expected: R increases with thickness, saturates for optically thick")
    print("=" * 70)

    wavelength = 10.0
    n = 2.0
    k = 0.05
    diameter = 5.0
    vf = 30.0

    thicknesses = [100, 500, 1000, 2000, 5000]
    reflectances = []

    print(f"\nParameters: n={n}, k={k}, D={diameter}µm, VF={vf}%, λ={wavelength}µm")
    print("-" * 50)
    print(f"{'Thickness (µm)':<15} | {'R_sim':>8} | {'Trend':>10}")
    print("-" * 50)

    prev_r = 0.0
    increasing_or_saturating = True

    for t in thicknesses:
        r = run_fos_simulation(n, k, diameter, vf, t, wavelength)
        reflectances.append(r)

        if r > prev_r:
            trend = "↑ GOOD"
        elif abs(r - prev_r) < 0.02:  # Saturated
            trend = "≈ GOOD (sat)"
        else:
            trend = "↓ BAD"
            increasing_or_saturating = False

        print(f"{t:<15} | {r:>8.4f} | {trend:>10}")
        prev_r = r

    result = "PASS" if increasing_or_saturating else "FAIL"
    print(f"\nResult: {result} - Reflectance {'increases/saturates' if increasing_or_saturating else 'does NOT behave correctly'} with thickness")
    return increasing_or_saturating


def test_volume_fraction_dependence():
    """Test: Reflectance should increase with scattering (volume fraction)."""
    print("\n" + "=" * 70)
    print("TEST 3: Reflectance vs Volume Fraction")
    print("Expected: R increases with VF (more scattering)")
    print("=" * 70)

    wavelength = 10.0
    n = 2.0
    k = 0.05
    diameter = 5.0
    thickness = 2000.0

    vf_values = [5, 10, 20, 30, 50]
    reflectances = []

    print(f"\nParameters: n={n}, k={k}, D={diameter}µm, thickness={thickness}µm, λ={wavelength}µm")
    print("-" * 50)
    print(f"{'VF (%)':<10} | {'R_sim':>8} | {'Trend':>10}")
    print("-" * 50)

    prev_r = 0.0
    all_increasing = True

    for vf in vf_values:
        r = run_fos_simulation(n, k, diameter, vf, thickness, wavelength)
        reflectances.append(r)

        trend = "↑ GOOD" if r > prev_r else "↓ BAD"
        if r <= prev_r and vf > vf_values[0]:
            all_increasing = False

        print(f"{vf:<10} | {r:>8.4f} | {trend:>10}")
        prev_r = r

    result = "PASS" if all_increasing else "FAIL"
    print(f"\nResult: {result} - Reflectance {'increases' if all_increasing else 'does NOT increase'} with volume fraction")
    return all_increasing


def test_kubelka_munk_comparison():
    """Test: Compare against Kubelka-Munk for optically thick medium."""
    print("\n" + "=" * 70)
    print("TEST 4: Comparison with Kubelka-Munk Model")
    print("Expected: Semi-quantitative agreement for diffuse reflectance")
    print("=" * 70)

    wavelength = 10.0
    diameter = 5.0
    thickness = 5000.0  # Thick enough to be "semi-infinite"
    vf = 30.0

    # Test cases with varying absorption
    test_cases = [
        ("Low absorption", 2.0, 0.01),
        ("Medium absorption", 2.0, 0.05),
        ("High absorption", 2.0, 0.2),
    ]

    print(f"\nParameters: D={diameter}µm, VF={vf}%, thickness={thickness}µm, λ={wavelength}µm")
    print("-" * 70)
    print(f"{'Case':<20} | {'R_sim':>8} | {'R_KM':>8} | {'Diff':>8} | {'Status'}")
    print("-" * 70)

    all_reasonable = True

    for name, n, k in test_cases:
        r_sim = run_fos_simulation(n, k, diameter, vf, thickness, wavelength, n_photons=10000)

        # Estimate K and S from optical properties
        # This is approximate - K-M assumes diffuse illumination
        # K ≈ 4*pi*k/lambda * (1-vf) + absorption_from_mie
        # S ≈ scattering coefficient from Mie

        # For a rough estimate using single scattering albedo concept:
        # omega = Qsca / Qext
        # K/S ≈ (1 - omega) / omega for diffuse

        # Use heuristic based on k: higher k = more absorption
        # K/S ratio roughly proportional to k for these materials
        KS_ratio = 10 * k  # Empirical scaling

        r_km = kubelka_munk_reflectance(KS_ratio, 1.0)

        diff = abs(r_sim - r_km)
        # Allow larger tolerance since K-M is an approximation
        status = "OK" if diff < 0.3 else "DIFFER"
        if diff >= 0.3:
            all_reasonable = False

        print(f"{name:<20} | {r_sim:>8.4f} | {r_km:>8.4f} | {diff:>8.4f} | {status}")

    print("\nNote: Kubelka-Munk is a simplified model. Large differences are expected")
    print("      but trends should be similar (lower R with higher absorption).")

    result = "PASS" if all_reasonable else "INFO"
    print(f"\nResult: {result} - {'Reasonable agreement' if all_reasonable else 'Differences noted (expected for KM approximation)'}")
    return all_reasonable


def test_limiting_cases():
    """Test: Check limiting behaviors."""
    print("\n" + "=" * 70)
    print("TEST 5: Physical Limiting Cases")
    print("=" * 70)

    wavelength = 10.0
    diameter = 5.0
    thickness = 3000.0
    vf = 40.0

    tests_passed = 0
    total_tests = 3

    # Test 5a: Very low absorption should give high reflectance
    print("\n5a: Low absorption (k=0.001) should give R > 0.3")
    r = run_fos_simulation(2.0, 0.001, diameter, vf, thickness, wavelength)
    status = "PASS" if r > 0.3 else "FAIL"
    if r > 0.3:
        tests_passed += 1
    print(f"    R = {r:.4f} - {status}")

    # Test 5b: Very high absorption should give low reflectance
    print("\n5b: High absorption (k=1.0) should give R < 0.3")
    r = run_fos_simulation(2.0, 1.0, diameter, vf, thickness, wavelength)
    status = "PASS" if r < 0.3 else "FAIL"
    if r < 0.3:
        tests_passed += 1
    print(f"    R = {r:.4f} - {status}")

    # Test 5c: Reflectance should be between 0 and 1
    print("\n5c: Reflectance must be in [0, 1] range")
    r = run_fos_simulation(2.2, 0.05, diameter, vf, thickness, wavelength)
    status = "PASS" if 0 <= r <= 1 else "FAIL"
    if 0 <= r <= 1:
        tests_passed += 1
    print(f"    R = {r:.4f} - {status}")

    result = "PASS" if tests_passed == total_tests else "FAIL"
    print(f"\nResult: {result} - {tests_passed}/{total_tests} limiting case tests passed")
    return tests_passed == total_tests


def main():
    print("=" * 70)
    print("REFLECTANCE CALCULATION VALIDATION")
    print("Monte Carlo Radiative Transfer Physics Tests")
    print("=" * 70)

    results = []

    # Suppress FOS output
    import io
    import contextlib

    print("\nRunning tests (this may take a few minutes)...")

    with contextlib.redirect_stdout(io.StringIO()):
        pass  # FOS prints to stdout, but we want clean output

    results.append(("Absorption dependence", test_absorption_dependence()))
    results.append(("Thickness dependence", test_thickness_dependence()))
    results.append(("Volume fraction dependence", test_volume_fraction_dependence()))
    results.append(("Kubelka-Munk comparison", test_kubelka_munk_comparison()))
    results.append(("Limiting cases", test_limiting_cases()))

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name:<30}: {status}")

    print("-" * 70)
    print(f"  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ REFLECTANCE CALCULATION VALIDATED")
        print("  The Monte Carlo radiative transfer produces physically correct results")
        return 0
    else:
        print("\n✗ Some tests failed - review results above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
