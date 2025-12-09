# MCRT - Monte Carlo Radiative Transfer

Interactive Monte Carlo radiative transfer simulator for particulate materials.

## Features

- **Monte Carlo Simulation**: Uses FOS (Fast Optical Spectrum) for accurate photon transport
- **Materials Database**: Access to 1000+ materials from refractiveindex.info
- **Progressive Updates**: Real-time results via Server-Sent Events
- **Save/Restore**: Pause and continue simulations
- **Web Interface**: React frontend with interactive controls and charts

## Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/eric/mcrt.git
cd mcrt

# Install Python package
pip install -e ".[all]"

# Install frontend dependencies
cd frontend
npm install
```

## Quick Start

```bash
# Start backend
python -m mcrt.api.app

# In another terminal, start frontend
cd frontend
npm run dev
```

Open http://localhost:3000 in your browser.

## Python API

```python
from mcrt import FOSWrapper, MaterialDatabase

# Load materials
db = MaterialDatabase()
silver = db.get_optical_constants("main/Ag/Johnson")

# Run simulation
fos = FOSWrapper()
result = fos.run_simple(
    particle_wavelength_um=silver.wavelength_um,
    particle_n=silver.n,
    particle_k=silver.k,
    matrix_wavelength_um=acrylic.wavelength_um,
    matrix_n=acrylic.n,
    matrix_k=acrylic.k,
    particle_diameter_um=0.5,
    particle_volume_fraction=10,
    layer_thickness_um=100,
    n_photons=100000,
)

print(f"Reflectance: {result.reflectance}")
```

## License

MIT
