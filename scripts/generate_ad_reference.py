#!/usr/bin/env python3
"""Generate Adding-Doubling reference spectrum for validation.

This script computes the EXACT analytical reflectance spectrum using
the Adding-Doubling method with the same Mie properties that FOS uses.

The output can be used as a reference to validate Monte Carlo simulations.
"""

import sys
from pathlib import Path
import numpy as np
import json

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "fos/src"))

import iadpython as iad
from MieTheory3 import mie_theory


def get_mie_properties_spectrum(n_particle_arr, k_particle_arr, wavelength_arr,
                                 diameter_um, volume_fraction, thickness_um):
    """
    Compute Mie optical properties across wavelength spectrum.

    Returns arrays of omega, tau, g at each wavelength.
    """
    # Set up input arrays for mie_theory
    paint = np.column_stack([wavelength_arr, n_particle_arr, k_particle_arr])
    acr = np.column_stack([wavelength_arr, np.ones_like(wavelength_arr), np.zeros_like(wavelength_arr)])

    radius = diameter_um / 2
    r1 = np.array([radius])
    fv1 = np.array([volume_fraction / 100.0])
    dist = np.array([0.0])

    # Get optical properties from Mie theory
    prop = mie_theory(r1, fv1, paint, acr, thickness_um, dist, particle_type=0)

    # prop layout: [n_m, qa (1/cm), qs (1/cm), asy, thickness]
    n_wavelengths = len(wavelength_arr)
    omega = np.zeros(n_wavelengths)
    tau = np.zeros(n_wavelengths)
    g = np.zeros(n_wavelengths)

    thickness_cm = thickness_um / 1e4

    for i in range(n_wavelengths):
        qa = prop[1, i]
        qs = prop[2, i]
        asy = prop[3, i]

        if qs > 0:
            g[i] = asy / qs
        else:
            g[i] = 0.0

        if qa + qs > 0:
            omega[i] = qs / (qa + qs)
        else:
            omega[i] = 0.0

        tau[i] = (qa + qs) * thickness_cm

    return omega, tau, g


def adding_doubling_spectrum(omega_arr, tau_arr, g_arr):
    """
    Compute Adding-Doubling reflectance at each wavelength.
    """
    n_wavelengths = len(omega_arr)
    R_ad = np.zeros(n_wavelengths)
    T_ad = np.zeros(n_wavelengths)

    for i in range(n_wavelengths):
        omega = omega_arr[i]
        tau = tau_arr[i]
        g = g_arr[i]

        # Handle edge cases
        if tau < 0.001:
            R_ad[i] = 0.0
            T_ad[i] = 1.0
            continue
        if omega < 0.001:
            R_ad[i] = 0.0
            T_ad[i] = np.exp(-tau)
            continue

        try:
            s = iad.Sample(
                a=min(omega, 0.9999),  # Avoid singularity at omega=1
                b=tau,
                g=g,
                n=1.0,
                n_above=1.0,
                n_below=1.0,
                quad_pts=16
            )
            URU, UTU, _, _ = s.rt()
            R_ad[i] = float(URU)
            T_ad[i] = float(UTU)
        except Exception as e:
            print(f"Warning at wavelength {i}: {e}")
            R_ad[i] = 0.0
            T_ad[i] = 1.0

    return R_ad, T_ad


def load_optical_constants(material_path):
    """Load optical constants from refractiveindex.info format."""
    with open(material_path, 'r') as f:
        content = f.read()

    # Parse YAML-like format
    lines = content.split('\n')
    wavelength = []
    n_vals = []
    k_vals = []

    in_data = False
    for line in lines:
        line = line.strip()
        if 'data: |' in line:
            in_data = True
            continue
        if in_data:
            if line.startswith('-') or line == '' or ':' in line:
                if line.startswith('-'):
                    continue
                in_data = False
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    wl = float(parts[0])
                    n = float(parts[1])
                    k = float(parts[2]) if len(parts) >= 3 else 0.0
                    wavelength.append(wl)
                    n_vals.append(n)
                    k_vals.append(k)
                except ValueError:
                    continue

    return np.array(wavelength), np.array(n_vals), np.array(k_vals)


def generate_validation_reference():
    """Generate reference spectrum for validation demonstration."""

    # Use the validated Fe2O3-like parameters
    # n=2.0, k varies with wavelength (use simple model)
    # diameter=3µm, VF=40%, thickness=2000µm

    diameter_um = 3.0
    volume_fraction = 40.0
    thickness_um = 2000.0

    # Wavelength range
    wavelength_start = 7.0
    wavelength_end = 14.0
    wavelength_interval = 0.1
    wavelengths = np.arange(wavelength_start, wavelength_end + wavelength_interval/2, wavelength_interval)

    # Load Fe2O3 optical constants
    fe2o3_path = Path(__file__).parent.parent / "refractiveindex-database/database/data/main/Fe2O3/nk/Querry-o.yml"

    if fe2o3_path.exists():
        print(f"Loading Fe2O3 optical constants from {fe2o3_path}")
        wl_data, n_data, k_data = load_optical_constants(fe2o3_path)

        # Interpolate to our wavelengths
        n_particle = np.interp(wavelengths, wl_data, n_data)
        k_particle = np.interp(wavelengths, wl_data, k_data)
    else:
        print("Fe2O3 data not found, using constant values")
        n_particle = np.full_like(wavelengths, 2.2)
        k_particle = np.full_like(wavelengths, 0.05)

    print(f"\nParameters:")
    print(f"  Diameter: {diameter_um} µm")
    print(f"  Volume fraction: {volume_fraction}%")
    print(f"  Thickness: {thickness_um} µm")
    print(f"  Wavelength range: {wavelength_start}-{wavelength_end} µm")
    print(f"  Points: {len(wavelengths)}")

    # Get Mie properties
    print("\nComputing Mie properties...")
    omega, tau, g = get_mie_properties_spectrum(
        n_particle, k_particle, wavelengths,
        diameter_um, volume_fraction, thickness_um
    )

    print(f"  omega range: {omega.min():.3f} - {omega.max():.3f}")
    print(f"  tau range: {tau.min():.1f} - {tau.max():.1f}")
    print(f"  g range: {g.min():.3f} - {g.max():.3f}")

    # Compute Adding-Doubling reflectance
    print("\nComputing Adding-Doubling exact solution...")
    R_ad, T_ad = adding_doubling_spectrum(omega, tau, g)

    print(f"  R range: {R_ad.min():.4f} - {R_ad.max():.4f}")

    # Save reference data
    output_dir = Path(__file__).parent.parent / "src/mcrt/data/reference"
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_data = {
        "name": "Adding-Doubling Validation",
        "description": "Exact analytical solution using Adding-Doubling method (iadpython)",
        "source": "Computed using iadpython v0.5.3 (Scott Prahl)",
        "parameters": {
            "particle_material": "Fe2O3 (Querry-o)",
            "diameter_um": diameter_um,
            "volume_fraction_percent": volume_fraction,
            "thickness_um": thickness_um,
            "matrix": "Air (n=1.0)"
        },
        "wavelength_um": wavelengths.tolist(),
        "reflectance": R_ad.tolist(),
        "transmittance": T_ad.tolist(),
        "optical_properties": {
            "omega": omega.tolist(),
            "tau": tau.tolist(),
            "g": g.tolist()
        }
    }

    output_path = output_dir / "adding_doubling_fe2o3.json"
    with open(output_path, 'w') as f:
        json.dump(reference_data, f, indent=2)

    print(f"\nSaved reference data to {output_path}")

    # Also save CSV for easy viewing
    csv_path = output_dir / "adding_doubling_fe2o3.csv"
    with open(csv_path, 'w') as f:
        f.write("wavelength_um,reflectance,transmittance,omega,tau,g\n")
        for i in range(len(wavelengths)):
            f.write(f"{wavelengths[i]:.2f},{R_ad[i]:.6f},{T_ad[i]:.6f},{omega[i]:.4f},{tau[i]:.2f},{g[i]:.4f}\n")

    print(f"Saved CSV to {csv_path}")

    return reference_data


if __name__ == "__main__":
    generate_validation_reference()
