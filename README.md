# MCRT - Monte Carlo Radiative Transfer

Interactive Monte Carlo radiative transfer simulator for particulate materials. Computes reflectance, transmittance, and absorptance spectra for particles embedded in a matrix medium.

## Features

- **Monte Carlo Simulation**: Uses FOS (Fast Optical Spectrum) engine for accurate photon transport
- **Materials Database**: Access to 3400+ materials from refractiveindex.info
- **Progressive Updates**: Real-time results via Server-Sent Events
- **Save/Restore**: Pause and continue simulations
- **Web Interface**: React frontend with interactive controls and visualization
- **Validated**: Agrees with Adding-Doubling exact solutions within 0.17%

## Requirements

- **Python**: 3.10 or higher
- **Node.js**: 18 or higher (for web frontend)
- **Git**: For cloning submodules

## Installation

### Step 1: Clone with Submodules

This package depends on two external repositories included as git submodules:
- **FOS**: Monte Carlo radiative transfer engine
- **refractiveindex-database**: Optical constants for 3400+ materials (~100MB)

```bash
# Clone with all submodules (REQUIRED)
git clone --recursive https://github.com/ebpfly/mcrt.git
cd mcrt
```

If you already cloned without `--recursive`, initialize submodules:
```bash
git submodule update --init --recursive
```

### Step 2: Install Python Package

```bash
# Core package only
pip install .

# With web server support (recommended)
pip install ".[web]"

# With development tools
pip install ".[all]"
```

### Step 3: Install Frontend (for Web UI)

```bash
cd frontend
npm install
cd ..
```

## Quick Start

### Option A: Web Interface

Start both backend and frontend:

```bash
# Terminal 1: Start API server
mcrt serve --port 8000

# Terminal 2: Start frontend
cd frontend
npm run dev
```

Open http://localhost:5173 in your browser.

### Option B: Command Line

```bash
# Check version
mcrt version

# Run validation test
mcrt validate

# Start API server
mcrt serve --host 0.0.0.0 --port 8000
```

### Option C: Python API

```python
import numpy as np
from mcrt import FOSWrapper, MaterialDatabase

# Initialize material database
db = MaterialDatabase()

# List available materials
materials = db.list_materials()
print(f"Found {len(materials)} materials")

# Search for specific materials
iron_oxides = db.list_materials(search="Fe2O3")

# Load optical constants
fe2o3 = db.get_optical_constants("main/Fe2O3/Querry-o")
print(f"Fe2O3 wavelength range: {fe2o3.wavelength_um[0]:.2f} - {fe2o3.wavelength_um[-1]:.2f} um")

# Run Monte Carlo simulation
fos = FOSWrapper()
result = fos.run_simple(
    # Particle optical constants
    particle_wavelength_um=fe2o3.wavelength_um,
    particle_n=fe2o3.n,
    particle_k=fe2o3.k,
    # Matrix (air: n=1, k=0)
    matrix_wavelength_um=np.array([0.3, 16.0]),
    matrix_n=np.array([1.0, 1.0]),
    matrix_k=np.array([0.0, 0.0]),
    # Particle properties
    particle_diameter_um=3.0,
    particle_volume_fraction=40.0,  # percent
    layer_thickness_um=2000.0,
    # Simulation parameters
    wavelength_start_um=7.0,
    wavelength_end_um=14.0,
    wavelength_interval_um=0.1,
    n_photons=50000,
)

print(f"Wavelengths: {result.wavelength_um}")
print(f"Reflectance: {result.reflectance}")
print(f"Transmittance: {result.transmittance}")
print(f"Absorptance: {result.absorptance}")
```

## Validation

The Monte Carlo engine has been rigorously validated against the Adding-Doubling method, which provides numerically exact solutions for plane-parallel radiative transfer.

**Validation Results** (Fe2O3 particles, 3 um diameter, 40% volume fraction, 2 mm slab):

| Parameter | Value |
|-----------|-------|
| Agreement with Adding-Doubling | 0.17% |
| Test wavelength range | 7-14 um |
| Optical thickness | ~400 |
| Single scattering albedo | ~0.85 |

Run the validation test:
```bash
mcrt validate --verbose
```

Reference data is included in `src/mcrt/reference_data/adding_doubling_fe2o3.json`.

## Project Structure

```
mcrt/
├── src/mcrt/           # Python package
│   ├── api/            # FastAPI web server
│   ├── fos/            # FOS wrapper
│   ├── materials/      # Material database interface
│   └── reference_data/ # Validation reference data
├── frontend/           # React/TypeScript web UI
├── fos/                # FOS Monte Carlo engine (submodule)
├── refractiveindex-database/  # Optical constants (submodule)
├── scripts/            # Validation and utility scripts
└── tests/              # Test suite
```

## External Dependencies

### FOS (Fast Optical Spectrum)
- **Purpose**: Monte Carlo radiative transfer engine
- **Location**: `fos/` submodule
- **Source**: Included as git submodule

### refractiveindex.info Database
- **Purpose**: Optical constants (n, k) for 3400+ materials
- **Location**: `refractiveindex-database/` submodule
- **Source**: https://github.com/polyanskiy/refractiveindex.info-database
- **Website**: https://refractiveindex.info

### Adding-Doubling Reference (iadpython)
- **Purpose**: Validation reference for Monte Carlo
- **Source**: https://github.com/scottprahl/iadpython
- **Citation**: Scott Prahl, "iadpython: Inverse Adding-Doubling"

## API Reference

### MaterialDatabase

```python
from mcrt import MaterialDatabase

db = MaterialDatabase()

# List all materials
materials = db.list_materials()

# Filter by shelf (main, organic, glass, other, 3d)
metals = db.list_materials(shelf="main")

# Search by name
results = db.list_materials(search="silver")

# Get optical constants
oc = db.get_optical_constants("main/Ag/Johnson")
# Returns: OpticalConstants with wavelength_um, n, k arrays

# Interpolate to new wavelength grid
oc_interp = oc.interpolate(np.linspace(0.4, 0.8, 100))

# Trim to wavelength range
oc_trim = oc.trim(0.4, 0.8)
```

### FOSWrapper

```python
from mcrt import FOSWrapper

fos = FOSWrapper()

# Simple simulation
result = fos.run_simple(
    particle_wavelength_um=...,
    particle_n=...,
    particle_k=...,
    matrix_wavelength_um=...,
    matrix_n=...,
    matrix_k=...,
    particle_diameter_um=1.0,
    particle_volume_fraction=10.0,
    layer_thickness_um=100.0,
    wavelength_start_um=0.4,
    wavelength_end_um=0.8,
    wavelength_interval_um=0.01,
    n_photons=10000,
    particle_std_dev=0.0,  # optional size distribution
)

# Batch simulation with progress callbacks
def on_progress(batch, total, accumulated_result):
    print(f"Batch {batch}/{total}: R={accumulated_result.reflectance.mean():.3f}")

result = fos.run_batch(
    ...,
    total_photons=100000,
    n_batches=10,
    progress_callback=on_progress,
)
```

### FOSResult

```python
result.wavelength_um   # Wavelength array
result.reflectance     # Hemispherical reflectance [0-1]
result.transmittance   # Hemispherical transmittance [0-1]
result.absorptance     # Absorptance [0-1]
result.n_photons       # Number of photons simulated

# Accumulate multiple results
combined = FOSResult.accumulate([result1, result2, result3])
```

## Troubleshooting

### "Database not found" error
```
FileNotFoundError: Database not found at .../refractiveindex-database/database
```
**Solution**: Initialize git submodules:
```bash
git submodule update --init --recursive
```

### "FOS Main3.py not found" error
```
FileNotFoundError: FOS Main3.py not found at .../fos/src
```
**Solution**: Initialize git submodules:
```bash
git submodule update --init --recursive
```

### "uvicorn not installed" error
```
Error: uvicorn not installed. Install with: pip install mcrt[web]
```
**Solution**: Install web dependencies:
```bash
pip install ".[web]"
```

### Frontend can't connect to backend
- Ensure backend is running: `mcrt serve --port 8000`
- Check CORS settings if using different ports
- Frontend expects backend at `http://localhost:8000` by default

## License

MIT

## References

1. van de Hulst, H.C. (1980). *Multiple Light Scattering*. Academic Press.
2. Prahl, S.A. (1995). "The Adding-Doubling Method." *Optical-Thermal Response of Laser-Irradiated Tissue*.
3. Polyanskiy, M.N. "Refractive index database." https://refractiveindex.info
