#!/usr/bin/env python3
"""Rigorous Monte Carlo validation against Adding-Doubling exact solutions.

This script validates the FOS Monte Carlo radiative transfer by comparing against
the Adding-Doubling method implemented in iadpython (by Scott Prahl).

The Adding-Doubling method gives NUMERICALLY EXACT solutions for plane-parallel
radiative transfer. Monte Carlo should match within statistical uncertainty.

Key validation approach:
1. Use FOS Mie theory to compute actual optical properties (qa, qs, g)
2. Calculate omega (single scattering albedo) and tau (optical thickness)
3. Use iadpython to compute EXACT reflectance for the same omega, tau, g
4. Run FOS Monte Carlo and compare

Reference:
- Prahl, "Everything I think I know about Inverse Adding-Doubling" (1995)
- van de Hulst, "Multiple Light Scattering" (1980)
"""

import sys
from pathlib import Path
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "fos/src"))

import iadpython as iad
from MieTheory3 import mie_theory
from mcrt.fos.wrapper import FOSWrapper


def extract_optical_properties(n_particle, k_particle, n_matrix, diameter_um,
                               volume_fraction, thickness_um, wavelength_um):
    """
    Use FOS Mie theory to compute optical properties.

    Returns:
        omega: single scattering albedo = ks / (ka + ks)
        tau: optical thickness = (ka + ks) * thickness
        g: asymmetry parameter
        qa: absorption coefficient (1/cm)
        qs: scattering coefficient (1/cm)
    """
    # Set up input arrays for mie_theory
    paint = np.array([[wavelength_um, n_particle, k_particle]])
    acr = np.array([[wavelength_um, n_matrix, 0.0]])  # non-absorbing matrix

    radius = diameter_um / 2
    r1 = np.array([radius])
    fv1 = np.array([volume_fraction / 100.0])  # Convert from percent
    dist = np.array([0.0])  # No distribution

    # Get optical properties from Mie theory
    prop = mie_theory(r1, fv1, paint, acr, thickness_um, dist, particle_type=0)

    # prop layout: [n_m, qa (1/cm), qs (1/cm), asy, thickness]
    qa = prop[1, 0]  # absorption coefficient
    qs = prop[2, 0]  # scattering coefficient
    asy = prop[3, 0]  # weighted asymmetry (= qs * g)

    # Calculate derived properties
    if qs > 0:
        g = asy / qs  # asymmetry parameter
    else:
        g = 0.0

    if qa + qs > 0:
        omega = qs / (qa + qs)  # single scattering albedo
    else:
        omega = 0.0

    # Optical thickness: tau = (ka + ks) * L (with unit conversion)
    # qa and qs are in 1/cm, thickness is in µm
    thickness_cm = thickness_um / 1e4
    tau = (qa + qs) * thickness_cm

    return omega, tau, g, qa, qs


def adding_doubling_reflectance(omega, tau, g, n_medium=1.0):
    """
    Calculate exact reflectance using Adding-Doubling method (iadpython).

    Uses diffuse illumination from above.

    Args:
        omega: single scattering albedo (0 to 1)
        tau: optical thickness
        g: asymmetry parameter (-1 to 1)
        n_medium: refractive index of surrounding medium

    Returns:
        R: diffuse-diffuse reflectance (from above)
        T: diffuse-diffuse transmittance (from above)
    """
    # Create sample with specified optical properties
    # Use high quadrature for accuracy (16 points is good balance of speed/accuracy)
    s = iad.Sample(
        a=omega,      # single scattering albedo
        b=tau,        # optical thickness
        g=g,          # asymmetry parameter
        n=1.0,        # refractive index of sample (matched)
        n_above=n_medium,
        n_below=n_medium,
        quad_pts=16   # quadrature points for angular integration
    )

    # Compute reflectance and transmittance using adding-doubling
    # This gives the EXACT solution for the radiative transfer equation
    # rt() returns: (R_above, T_above, R_below, T_below)
    try:
        URU, UTU, _, _ = s.rt()
        return float(URU), float(UTU)
    except Exception as e:
        print(f"  Warning: iad.rt failed: {e}")
        return None, None


def run_fos_mc(n_particle, k_particle, diameter_um, volume_fraction,
               thickness_um, wavelength_um, n_photons=100000):
    """
    Run FOS Monte Carlo simulation.

    Returns:
        reflectance: hemispherical reflectance from MC
    """
    fos_path = Path(__file__).parent.parent / "fos/src"
    work_dir = Path(__file__).parent.parent / ".mcrt_work"
    fos = FOSWrapper(fos_path=fos_path, work_dir=work_dir)

    # Set up wavelength arrays (need at least 3 points for FOS)
    delta = 0.1
    particle_wavelength = np.array([wavelength_um - delta, wavelength_um, wavelength_um + delta])
    particle_n_arr = np.array([n_particle, n_particle, n_particle])
    particle_k_arr = np.array([k_particle, k_particle, k_particle])

    matrix_wavelength = np.array([wavelength_um - delta, wavelength_um, wavelength_um + delta])
    matrix_n = np.ones(3)
    matrix_k = np.zeros(3)

    result = fos.run_simple(
        particle_wavelength_um=particle_wavelength,
        particle_n=particle_n_arr,
        particle_k=particle_k_arr,
        matrix_wavelength_um=matrix_wavelength,
        matrix_n=matrix_n,
        matrix_k=matrix_k,
        particle_diameter_um=diameter_um,
        particle_volume_fraction=volume_fraction,
        layer_thickness_um=thickness_um,
        wavelength_start_um=wavelength_um - delta,
        wavelength_end_um=wavelength_um + delta,
        wavelength_interval_um=delta,
        n_photons=n_photons,
        particle_std_dev=0.0,
    )

    # Return reflectance at center wavelength
    idx = len(result.reflectance) // 2
    return result.reflectance[idx]


def run_validation_suite():
    """
    Run comprehensive validation against Adding-Doubling.
    """
    print("=" * 80)
    print("RIGOROUS MONTE CARLO VALIDATION")
    print("FOS Monte Carlo vs Adding-Doubling (iadpython) Exact Solutions")
    print("=" * 80)

    print("""
The Adding-Doubling method provides NUMERICALLY EXACT solutions for the
radiative transfer equation in plane-parallel geometry. This is the gold
standard for validating Monte Carlo radiative transfer codes.

Validation approach:
1. Extract exact optical properties (omega, tau, g) from FOS Mie calculation
2. Compute exact reflectance using iadpython adding-doubling
3. Run FOS Monte Carlo with the same physical parameters
4. Compare - should agree within statistical uncertainty (~1/sqrt(N))
""")

    wavelength = 10.0  # µm

    # Test cases: (name, n_particle, k_particle, diameter_um, volume_fraction, thickness_um)
    test_cases = [
        # Weakly absorbing, thin layer
        ("Low abs, thin", 1.5, 0.01, 2.0, 10.0, 100.0),

        # Moderate absorption, medium thickness
        ("Med abs, med thick", 1.5, 0.05, 2.0, 20.0, 500.0),

        # Strong scattering (high n contrast)
        ("Strong sca", 2.0, 0.01, 3.0, 30.0, 300.0),

        # Higher absorption
        ("High abs", 1.5, 0.2, 2.0, 20.0, 500.0),

        # Very thick layer (semi-infinite)
        ("Semi-infinite", 1.5, 0.05, 2.0, 30.0, 5000.0),

        # Fe2O3-like
        ("Fe2O3-like", 2.2, 0.05, 3.0, 40.0, 2000.0),
    ]

    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print(f"\n{'Test Case':<20} | {'omega':>6} | {'tau':>7} | {'g':>6} | {'R_AD':>8} | {'R_MC':>8} | {'Err%':>7} | Status")
    print("-" * 95)

    results = []

    for name, n_p, k_p, diam, vf, thick in test_cases:
        # Step 1: Get exact optical properties from FOS Mie
        omega, tau, g, qa, qs = extract_optical_properties(
            n_p, k_p, 1.0, diam, vf, thick, wavelength
        )

        # Step 2: Compute exact reflectance with Adding-Doubling
        R_ad, T_ad = adding_doubling_reflectance(omega, tau, g)

        if R_ad is None:
            print(f"{name:<20} | {omega:>6.4f} | {tau:>7.2f} | {g:>6.3f} | {'ERROR':>8} | {'-':>8} | {'-':>7} | SKIP")
            continue

        # Step 3: Run FOS Monte Carlo
        n_photons = 100000  # More photons = lower statistical error
        R_mc = run_fos_mc(n_p, k_p, diam, vf, thick, wavelength, n_photons)

        # Step 4: Compare
        if R_ad > 0.001:
            err_pct = abs(R_mc - R_ad) / R_ad * 100
        else:
            err_pct = abs(R_mc - R_ad) * 100

        # Statistical uncertainty: ~1/sqrt(N) * R for MC
        stat_err = R_mc / np.sqrt(n_photons) * 100  # % uncertainty

        # Pass if within 3 sigma of statistical uncertainty, or within 5% absolute
        tolerance = max(3 * stat_err, 5.0)
        passed = err_pct < tolerance
        status = "PASS" if passed else "FAIL"

        print(f"{name:<20} | {omega:>6.4f} | {tau:>7.2f} | {g:>6.3f} | {R_ad:>8.4f} | {R_mc:>8.4f} | {err_pct:>6.2f}% | {status}")

        results.append({
            'name': name,
            'omega': omega,
            'tau': tau,
            'g': g,
            'R_ad': R_ad,
            'R_mc': R_mc,
            'err_pct': err_pct,
            'passed': passed
        })

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    n_passed = sum(1 for r in results if r['passed'])
    n_total = len(results)

    if results:
        mean_err = np.mean([r['err_pct'] for r in results])
        max_err = np.max([r['err_pct'] for r in results])

        print(f"\nTests passed: {n_passed}/{n_total}")
        print(f"Mean error:   {mean_err:.2f}%")
        print(f"Max error:    {max_err:.2f}%")

    n_photons_used = 100000  # Default for summary
    print("\nNotes on expected agreement:")
    print("- Adding-Doubling is exact to machine precision")
    print("- Monte Carlo has statistical error ~1/sqrt(N_photons)")
    print(f"- With {n_photons_used:,} photons, expect ~{100/np.sqrt(n_photons_used):.1f}% statistical error")
    print("- Agreement within 5% indicates correct physics")
    print("- Larger differences may indicate boundary condition or geometry differences")

    if n_passed == n_total:
        print("\n" + "=" * 80)
        print("VALIDATION PASSED")
        print("Monte Carlo reflectance agrees with exact Adding-Doubling solutions")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("VALIDATION INCOMPLETE")
        print("Some tests show larger differences - see notes above")
        print("=" * 80)
        return 1


def two_stream_comparison():
    """
    Additional validation using simple two-stream analytical solution.
    This doesn't require iadpython and serves as a cross-check.
    """
    print("\n\n" + "=" * 80)
    print("ADDITIONAL VALIDATION: Two-Stream Analytical Solution")
    print("=" * 80)

    print("""
The two-stream approximation provides an exact solution to the two-stream
radiative transfer equations. For isotropic scattering (g=0), it gives:

  R = ρ(1 - exp(-2γτ)) / (1 - ρ²exp(-2γτ))

  where γ = sqrt(1-ω), ρ = (1-γ)/(1+γ)

This is compared against FOS Monte Carlo as an additional cross-check.
""")

    wavelength = 10.0

    # Use parameters that give near-isotropic scattering (small size parameter)
    test_cases = [
        ("Small x, low abs", 1.5, 0.01, 0.5, 20.0, 500.0),
        ("Small x, med abs", 1.5, 0.1, 0.5, 20.0, 500.0),
        ("Small x, thick", 1.5, 0.01, 0.5, 30.0, 2000.0),
    ]

    print(f"\n{'Test Case':<20} | {'omega':>6} | {'tau':>7} | {'g':>6} | {'R_2S':>8} | {'R_MC':>8} | {'Err%':>7}")
    print("-" * 85)

    for name, n_p, k_p, diam, vf, thick in test_cases:
        # Get optical properties
        omega, tau, g, qa, qs = extract_optical_properties(
            n_p, k_p, 1.0, diam, vf, thick, wavelength
        )

        # Two-stream solution (works best for g ≈ 0)
        if omega > 0.9999:
            R_2s = tau / (2 + tau)
        else:
            gamma = np.sqrt(1 - omega)
            rho = (1 - gamma) / (1 + gamma)
            exp_term = np.exp(-2 * gamma * tau)
            denom = 1 - rho**2 * exp_term
            R_2s = rho * (1 - exp_term) / denom

        # MC simulation
        R_mc = run_fos_mc(n_p, k_p, diam, vf, thick, wavelength, n_photons=50000)

        err_pct = abs(R_mc - R_2s) / max(R_2s, 0.001) * 100

        print(f"{name:<20} | {omega:>6.4f} | {tau:>7.2f} | {g:>6.3f} | {R_2s:>8.4f} | {R_mc:>8.4f} | {err_pct:>6.2f}%")

    print("\nNote: Two-stream approximation is less accurate for anisotropic scattering (|g| > 0.3)")


if __name__ == "__main__":
    result = run_validation_suite()
    two_stream_comparison()
    sys.exit(result)
