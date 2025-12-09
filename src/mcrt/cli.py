"""Command-line interface for MCRT."""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mcrt",
        description="Monte Carlo Radiative Transfer simulator for particulate materials",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Server command
    server_parser = subparsers.add_parser("serve", help="Start the web API server")
    server_parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
    )
    server_parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (default: 8000)"
    )
    server_parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    # Version command
    subparsers.add_parser("version", help="Show version information")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Run validation tests")
    validate_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )

    args = parser.parse_args()

    if args.command == "serve":
        run_server(args)
    elif args.command == "version":
        show_version()
    elif args.command == "validate":
        run_validation(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_server(args):
    """Start the web API server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Install with: pip install mcrt[web]")
        sys.exit(1)

    uvicorn.run(
        "mcrt.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def show_version():
    """Show version information."""
    from mcrt import __version__
    print(f"mcrt {__version__}")


def run_validation(args):
    """Run validation tests against Adding-Doubling exact solutions."""
    import numpy as np
    from mcrt import FOSWrapper, MaterialDatabase

    print("MCRT Validation Test")
    print("=" * 50)
    print("Comparing Monte Carlo against Adding-Doubling reference")
    print()

    # Load reference data
    import json
    from pathlib import Path

    ref_path = Path(__file__).parent / "reference_data" / "adding_doubling_fe2o3.json"
    if not ref_path.exists():
        print(f"Error: Reference data not found at {ref_path}")
        sys.exit(1)

    with open(ref_path) as f:
        ref_data = json.load(f)

    print(f"Reference: {ref_data['name']}")
    print(f"Source: {ref_data['source']}")
    print()

    # Get parameters from reference
    params = ref_data["parameters"]
    print(f"Parameters:")
    print(f"  Particle: {params['particle_material']}")
    print(f"  Diameter: {params['diameter_um']} um")
    print(f"  Volume fraction: {params['volume_fraction_percent']}%")
    print(f"  Thickness: {params['thickness_um']} um")
    print()

    # Test at single wavelength (10 um) for quick validation
    test_wavelength = 10.0

    # Find reference value at test wavelength
    ref_wavelengths = ref_data["data"]["wavelength_um"]
    ref_reflectance = ref_data["data"]["reflectance"]
    idx = ref_wavelengths.index(test_wavelength)
    ref_value = ref_reflectance[idx]

    print(f"Testing at {test_wavelength} um...")
    print(f"  Reference (Adding-Doubling): {ref_value:.4f}")

    try:
        # Load Fe2O3 optical constants
        db = MaterialDatabase()
        fe2o3 = db.get_material("main/Fe2O3/Querry-o")

        # Run Monte Carlo simulation
        fos = FOSWrapper()
        result = fos.run_simple(
            particle_wavelength_um=np.array(fe2o3.wavelength_um),
            particle_n=np.array(fe2o3.n),
            particle_k=np.array(fe2o3.k),
            matrix_wavelength_um=np.array([0.3, 16.0]),
            matrix_n=np.array([1.0, 1.0]),
            matrix_k=np.array([0.0, 0.0]),
            particle_diameter_um=params["diameter_um"],
            particle_volume_fraction=params["volume_fraction_percent"],
            layer_thickness_um=params["thickness_um"],
            wavelength_start_um=test_wavelength,
            wavelength_end_um=test_wavelength,
            wavelength_interval_um=0.1,
            n_photons=50000,
        )

        mc_value = result.reflectance[0]
        error_pct = abs(mc_value - ref_value) / ref_value * 100

        print(f"  Monte Carlo result: {mc_value:.4f}")
        print(f"  Difference: {error_pct:.2f}%")
        print()

        if error_pct < 5.0:
            print("PASSED: Monte Carlo agrees with Adding-Doubling within 5%")
            sys.exit(0)
        else:
            print("WARNING: Difference exceeds 5% - may need more photons")
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure FOS submodule is initialized: git submodule update --init")
        sys.exit(1)
    except Exception as e:
        print(f"Error running simulation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
