#!/usr/bin/env python3
"""Validate FOS Mie scattering calculations against miepython reference.

This script compares Qext, Qsca, and asymmetry parameter (g) computed by
FOS against the well-tested miepython library for various test cases.
"""

import sys
from pathlib import Path
import numpy as np
import miepython

# Add FOS to path
sys.path.insert(0, str(Path(__file__).parent.parent / "fos/src"))

from MieTheory3 import mie_theory


def compute_fos_mie(wavelength_um, particle_radius_um, n_particle, k_particle, n_matrix=1.0, k_matrix=0.0):
    """Compute Mie properties using FOS implementation.

    Returns: qext, qsca, qabs, g (asymmetry parameter)
    """
    # FOS expects arrays of [wavelength, n, k]
    paint = np.array([[wavelength_um, n_particle, k_particle]])
    acr = np.array([[wavelength_um, n_matrix, k_matrix]])

    r1 = np.array([particle_radius_um])
    fv1 = np.array([0.01])  # Low volume fraction for single particle
    thickness = 1000.0  # Arbitrary
    dist = np.array([0.0])  # No distribution

    prop = mie_theory(r1, fv1, paint, acr, thickness, dist, particle_type=0)

    # prop contains: [n_m, qa (absorption coef), qs (scattering coef), asy, thickness]
    # qa and qs are in units of 1/cm, need to convert back to Q values
    qa = prop[1, 0]  # Absorption coefficient
    qs = prop[2, 0]  # Scattering coefficient
    asy = prop[3, 0]  # Asymmetry parameter weighted by qs

    # Convert back to Q values: coefficient = 1.5 * Q * fv / (2*r) * 10^4
    # So Q = coefficient * 2 * r / (1.5 * fv * 10^4)
    fv = fv1[0]
    r = r1[0]
    qsca = qs * 2 * r / (1.5 * fv * 1e4)
    qabs = qa * 2 * r / (1.5 * fv * 1e4)
    qext = qsca + qabs

    # g is weighted: asy = qs * g, so g = asy / qs
    if qs > 0:
        g = asy / qs
    else:
        g = 0

    return qext, qsca, qabs, g


def compute_miepython_mie(wavelength_um, particle_radius_um, n_particle, k_particle, n_matrix=1.0):
    """Compute Mie properties using miepython reference implementation.

    Returns: qext, qsca, qabs, g (asymmetry parameter)
    """
    # miepython.efficiencies(m, d, lambda0, n_env=1.0)
    # m = complex refractive index of sphere
    # d = diameter of sphere
    # lambda0 = vacuum wavelength
    # n_env = refractive index of surrounding medium

    m = complex(n_particle, k_particle)
    d = 2 * particle_radius_um  # diameter
    lambda0 = wavelength_um

    qext, qsca, qback, g = miepython.efficiencies(m, d, lambda0, n_env=n_matrix)
    qabs = qext - qsca

    return qext, qsca, qabs, g


def run_validation():
    """Run validation tests comparing FOS against miepython."""

    print("=" * 70)
    print("MIE SCATTERING CODE VALIDATION: FOS vs miepython")
    print("=" * 70)

    # Test cases: (name, wavelength, radius, n_particle, k_particle)
    test_cases = [
        # Non-absorbing dielectrics
        ("Glass sphere (small)", 0.5, 0.1, 1.5, 0.0),
        ("Glass sphere (medium)", 0.5, 0.5, 1.5, 0.0),
        ("Glass sphere (large)", 0.5, 2.0, 1.5, 0.0),
        ("High-n dielectric", 1.0, 0.5, 3.0, 0.0),

        # Absorbing particles
        ("Weakly absorbing", 1.0, 0.5, 1.5, 0.01),
        ("Moderately absorbing", 1.0, 0.5, 1.5, 0.1),
        ("Strongly absorbing", 1.0, 0.5, 1.5, 0.5),

        # Metal-like (high k)
        ("Metal-like", 1.0, 0.1, 0.5, 2.0),

        # Thermal IR cases (like Fe2O3)
        ("Fe2O3-like at 10um", 10.0, 3.0, 2.2, 0.05),
        ("Fe2O3-like at 8um", 8.0, 3.0, 2.4, 0.04),

        # Various size parameters
        ("x=0.1 (Rayleigh)", 1.0, 0.0159, 1.5, 0.01),
        ("x=1.0 (Mie)", 1.0, 0.159, 1.5, 0.01),
        ("x=5.0 (Mie)", 1.0, 0.796, 1.5, 0.01),
        ("x=10.0 (Mie)", 1.0, 1.59, 1.5, 0.01),
    ]

    print("\nTest results (comparing Qext, Qsca, g):")
    print("-" * 70)
    print(f"{'Test Case':<25} {'x':>6} | {'Qext':>8} {'Qext_ref':>8} {'err%':>7} | {'g':>6} {'g_ref':>6}")
    print("-" * 70)

    all_passed = True
    qext_errors = []
    qsca_errors = []
    g_errors = []

    for name, wl, r, n_p, k_p in test_cases:
        # Calculate size parameter
        x = 2 * np.pi * r / wl

        # FOS calculation
        try:
            qext_fos, qsca_fos, qabs_fos, g_fos = compute_fos_mie(wl, r, n_p, k_p)
        except Exception as e:
            print(f"{name:<25} {x:>6.2f} | ERROR: {e}")
            all_passed = False
            continue

        # miepython calculation
        qext_ref, qsca_ref, qabs_ref, g_ref = compute_miepython_mie(wl, r, n_p, k_p)

        # Calculate errors
        if qext_ref > 0.001:
            qext_err = abs(qext_fos - qext_ref) / qext_ref * 100
        else:
            qext_err = abs(qext_fos - qext_ref) * 100

        if abs(g_ref) > 0.001:
            g_err = abs(g_fos - g_ref) / abs(g_ref) * 100
        else:
            g_err = abs(g_fos - g_ref) * 100

        qext_errors.append(qext_err)
        g_errors.append(g_err)

        # Check if passed (within 5% tolerance)
        passed = qext_err < 5.0
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"{name:<25} {x:>6.2f} | {qext_fos:>8.4f} {qext_ref:>8.4f} {qext_err:>6.2f}% | {g_fos:>6.3f} {g_ref:>6.3f}  [{status}]")

    print("-" * 70)

    # Summary statistics
    print(f"\nSummary:")
    print(f"  Mean Qext error: {np.mean(qext_errors):.2f}%")
    print(f"  Max Qext error:  {np.max(qext_errors):.2f}%")
    print(f"  Mean g error:    {np.mean(g_errors):.2f}%")

    if all_passed:
        print("\n" + "=" * 70)
        print("VALIDATION PASSED: All Mie calculations within 5% of reference")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("VALIDATION FAILED: Some tests exceeded 5% error threshold")
        print("=" * 70)
        return 1


def detailed_comparison():
    """Show detailed comparison for a single test case."""
    print("\n" + "=" * 70)
    print("DETAILED COMPARISON: Fe2O3-like particle at 10 µm")
    print("=" * 70)

    # Fe2O3-like optical constants at 10 µm
    wavelength = 10.0
    n_particle = 2.212
    k_particle = 0.056

    print(f"\nOptical constants: n = {n_particle}, k = {k_particle}")
    print(f"Wavelength: {wavelength} µm")
    print("\nRadius (µm) | x      | Qext_FOS | Qext_ref | Qsca_FOS | Qsca_ref | g_FOS  | g_ref")
    print("-" * 90)

    for radius in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
        x = 2 * np.pi * radius / wavelength

        qext_fos, qsca_fos, qabs_fos, g_fos = compute_fos_mie(wavelength, radius, n_particle, k_particle)
        qext_ref, qsca_ref, qabs_ref, g_ref = compute_miepython_mie(wavelength, radius, n_particle, k_particle)

        print(f"{radius:>10.1f}  | {x:>6.2f} | {qext_fos:>8.4f} | {qext_ref:>8.4f} | {qsca_fos:>8.4f} | {qsca_ref:>8.4f} | {g_fos:>6.3f} | {g_ref:>6.3f}")


if __name__ == "__main__":
    result = run_validation()
    detailed_comparison()
    sys.exit(result)
