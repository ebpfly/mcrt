#!/usr/bin/env python3
"""Parse JCAMP-DX file and convert to wavelength (µm) vs reflectance CSV."""

import re
import json

def parse_jcamp(filepath: str) -> tuple[list[float], list[float]]:
    """Parse JCAMP-DX file and return wavenumbers and reflectance values."""
    with open(filepath, 'r', encoding='latin-1') as f:
        content = f.read()

    # Extract metadata
    yfactor_match = re.search(r'##YFACTOR=([0-9.e+-]+)', content)
    yfactor = float(yfactor_match.group(1)) if yfactor_match else 1.0

    deltax_match = re.search(r'##DELTAX=([0-9.e+-]+)', content)
    deltax = float(deltax_match.group(1)) if deltax_match else -0.96450084

    # Find start of data
    data_match = re.search(r'##XYDATA=\(X\+\+\(Y\.\.Y\)\)\n(.+)', content, re.DOTALL)
    if not data_match:
        raise ValueError("Could not find XYDATA section")

    data_section = data_match.group(1)

    wavenumbers = []
    reflectances = []

    for line in data_section.strip().split('\n'):
        if line.startswith('##'):
            break

        # Parse line: first value is X, rest are Y values
        parts = line.replace('+', ' ').replace('-', ' -').split()
        if not parts:
            continue

        try:
            x_start = float(parts[0])
            y_values = [int(p) for p in parts[1:]]

            for i, y_int in enumerate(y_values):
                wn = x_start + i * deltax
                refl = y_int * yfactor
                wavenumbers.append(wn)
                reflectances.append(refl)
        except (ValueError, IndexError):
            continue

    return wavenumbers, reflectances


def wavenumber_to_wavelength_um(wn: float) -> float:
    """Convert wavenumber (cm⁻¹) to wavelength (µm)."""
    return 10000.0 / wn


def main():
    # Parse the NIST quartz data
    wavenumbers, reflectances = parse_jcamp('/tmp/quartz_nist.jdx')

    # Convert to wavelength and filter to LWIR range (7-15 µm)
    data = []
    for wn, refl in zip(wavenumbers, reflectances):
        if wn > 0:
            wl = wavenumber_to_wavelength_um(wn)
            if 1.0 <= wl <= 17.0:  # Keep broad range
                data.append({'wavelength_um': round(wl, 4), 'reflectance': round(refl, 6)})

    # Sort by wavelength
    data.sort(key=lambda x: x['wavelength_um'])

    # Subsample to reasonable number of points (every 10th point for ~700 points)
    subsampled = data[::10]

    # Create reference data structure
    reference = {
        'id': 'quartz_sand_nist',
        'name': 'Quartz Sand (50-70 mesh)',
        'source': 'NIST WebBook / PNNL (March 2017)',
        'description': 'Hemispherical (diffuse) reflectance of quartz sand measured with integrating sphere',
        'measurement_type': 'diffuse_reflectance',
        'particle_size': '50-70 mesh (212-300 µm)',
        'instrument': 'Bruker Tensor 37 FTIR with A 562-G integrating sphere',
        'wavelength_range_um': [subsampled[0]['wavelength_um'], subsampled[-1]['wavelength_um']],
        'citation': 'Pacific Northwest National Laboratory under IARPA contract, CAS 14808-60-7',
        'url': 'https://webbook.nist.gov/cgi/cbook.cgi?ID=C14808607&Contrib=IARPA-IR-S&Type=IR-SPEC&Index=0',
        'data': {
            'wavelength_um': [d['wavelength_um'] for d in subsampled],
            'reflectance': [d['reflectance'] for d in subsampled]
        }
    }

    print(f"Parsed {len(data)} points, subsampled to {len(subsampled)} points")
    print(f"Wavelength range: {reference['wavelength_range_um'][0]:.2f} - {reference['wavelength_range_um'][1]:.2f} µm")
    print(f"Reflectance range: {min(reference['data']['reflectance']):.4f} - {max(reference['data']['reflectance']):.4f}")

    # Save to reference data directory
    output_path = '/Users/eric/repo/mcrt/src/mcrt/reference_data/quartz_sand_nist.json'
    with open(output_path, 'w') as f:
        json.dump(reference, f, indent=2)

    print(f"\nSaved to: {output_path}")

    # Also print a sample of data points in LWIR region
    print("\nSample data points in LWIR (8-14 µm):")
    lwir_points = [d for d in subsampled if 8.0 <= d['wavelength_um'] <= 14.0]
    for d in lwir_points[::10]:  # Every 10th point in LWIR
        print(f"  {d['wavelength_um']:.2f} µm: {d['reflectance']*100:.1f}%")


if __name__ == '__main__':
    main()
