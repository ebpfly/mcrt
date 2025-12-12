"""Validation tests comparing MCRT thin film implementation against tmm package.

The tmm package by Steven Byrnes is a well-tested reference implementation
of the transfer matrix method. These tests verify our implementation produces
matching results across various scenarios.

Reference: https://github.com/sbyrnes321/tmm
"""

import numpy as np
import pytest
from numpy.typing import NDArray

try:
    import tmm
    TMM_AVAILABLE = True
except ImportError:
    TMM_AVAILABLE = False

from mcrt.thinfilm import calculate_thin_film, ThinFilmStack, Layer


def run_tmm_reference(
    wavelengths_nm: NDArray,
    n_list: list[complex],
    d_list: list[float],
    angle_deg: float,
    polarization: str,
) -> tuple[NDArray, NDArray]:
    """Run tmm package calculation for comparison.

    Args:
        wavelengths_nm: Wavelengths in nanometers
        n_list: List of complex refractive indices [incident, layer1, ..., layerN, substrate]
        d_list: List of thicknesses in nm [inf, d1, ..., dN, inf]
        angle_deg: Angle of incidence in degrees
        polarization: 's', 'p', or 'unpolarized'

    Returns:
        Tuple of (reflectance, transmittance) arrays
    """
    angle_rad = np.deg2rad(angle_deg)
    R = np.zeros(len(wavelengths_nm))
    T = np.zeros(len(wavelengths_nm))

    for i, wl in enumerate(wavelengths_nm):
        if polarization == 'unpolarized':
            # tmm.unpolarized_RT returns dict with 'R' and 'T'
            result = tmm.unpolarized_RT(n_list, d_list, angle_rad, wl)
            R[i] = result['R']
            T[i] = result['T']
        else:
            result = tmm.coh_tmm(polarization, n_list, d_list, angle_rad, wl)
            R[i] = result['R']
            T[i] = result['T']

    return R, T


@pytest.mark.skipif(not TMM_AVAILABLE, reason="tmm package not installed")
class TestTMMValidation:
    """Validation tests against tmm reference implementation."""

    def test_single_layer_normal_incidence(self):
        """Test single dielectric layer at normal incidence."""
        wavelengths_nm = np.linspace(400, 800, 50)
        wavelengths_um = wavelengths_nm / 1000

        # Glass layer on silicon
        n_glass = 1.46
        n_silicon = 3.5 + 0.01j
        thickness_nm = 100

        # Our implementation
        result = calculate_thin_film(
            wavelength_um=wavelengths_um,
            layers=[{
                "thickness_nm": thickness_nm,
                "n": np.full(len(wavelengths_um), n_glass),
                "k": np.zeros(len(wavelengths_um)),
            }],
            substrate_n=np.full(len(wavelengths_um), n_silicon.real),
            substrate_k=np.full(len(wavelengths_um), n_silicon.imag),
        )

        # tmm reference
        n_list = [1.0, n_glass, n_silicon]
        d_list = [np.inf, thickness_nm, np.inf]
        R_ref, T_ref = run_tmm_reference(wavelengths_nm, n_list, d_list, 0, 'unpolarized')

        np.testing.assert_allclose(result.reflectance, R_ref, rtol=1e-6)
        np.testing.assert_allclose(result.transmittance, T_ref, rtol=1e-6)

    def test_single_layer_oblique_s_polarization(self):
        """Test single layer at 45 degrees, s-polarization."""
        wavelengths_nm = np.linspace(400, 800, 30)
        wavelengths_um = wavelengths_nm / 1000

        n_layer = 2.0
        n_sub = 1.5
        thickness_nm = 200
        angle_deg = 45

        # Our implementation
        result = calculate_thin_film(
            wavelength_um=wavelengths_um,
            layers=[{
                "thickness_nm": thickness_nm,
                "n": np.full(len(wavelengths_um), n_layer),
                "k": np.zeros(len(wavelengths_um)),
            }],
            substrate_n=np.full(len(wavelengths_um), n_sub),
            substrate_k=np.zeros(len(wavelengths_um)),
            angle_deg=angle_deg,
            polarization='s',
        )

        # tmm reference
        n_list = [1.0, n_layer, n_sub]
        d_list = [np.inf, thickness_nm, np.inf]
        R_ref, T_ref = run_tmm_reference(wavelengths_nm, n_list, d_list, angle_deg, 's')

        np.testing.assert_allclose(result.reflectance, R_ref, rtol=1e-6)
        np.testing.assert_allclose(result.transmittance, T_ref, rtol=1e-6)

    def test_single_layer_oblique_p_polarization(self):
        """Test single layer at 45 degrees, p-polarization."""
        wavelengths_nm = np.linspace(400, 800, 30)
        wavelengths_um = wavelengths_nm / 1000

        n_layer = 2.0
        n_sub = 1.5
        thickness_nm = 200
        angle_deg = 45

        # Our implementation
        result = calculate_thin_film(
            wavelength_um=wavelengths_um,
            layers=[{
                "thickness_nm": thickness_nm,
                "n": np.full(len(wavelengths_um), n_layer),
                "k": np.zeros(len(wavelengths_um)),
            }],
            substrate_n=np.full(len(wavelengths_um), n_sub),
            substrate_k=np.zeros(len(wavelengths_um)),
            angle_deg=angle_deg,
            polarization='p',
        )

        # tmm reference
        n_list = [1.0, n_layer, n_sub]
        d_list = [np.inf, thickness_nm, np.inf]
        R_ref, T_ref = run_tmm_reference(wavelengths_nm, n_list, d_list, angle_deg, 'p')

        np.testing.assert_allclose(result.reflectance, R_ref, rtol=1e-6)
        np.testing.assert_allclose(result.transmittance, T_ref, rtol=1e-6)

    def test_absorbing_layer(self):
        """Test layer with significant absorption (metal-like)."""
        wavelengths_nm = np.linspace(500, 700, 20)
        wavelengths_um = wavelengths_nm / 1000

        # Metal-like layer
        n_metal = 1.5 + 3.0j
        n_sub = 1.5
        thickness_nm = 50

        # Our implementation
        result = calculate_thin_film(
            wavelength_um=wavelengths_um,
            layers=[{
                "thickness_nm": thickness_nm,
                "n": np.full(len(wavelengths_um), n_metal.real),
                "k": np.full(len(wavelengths_um), n_metal.imag),
            }],
            substrate_n=np.full(len(wavelengths_um), n_sub),
            substrate_k=np.zeros(len(wavelengths_um)),
        )

        # tmm reference
        n_list = [1.0, n_metal, n_sub]
        d_list = [np.inf, thickness_nm, np.inf]
        R_ref, T_ref = run_tmm_reference(wavelengths_nm, n_list, d_list, 0, 'unpolarized')

        np.testing.assert_allclose(result.reflectance, R_ref, rtol=1e-5)
        np.testing.assert_allclose(result.transmittance, T_ref, rtol=1e-5)

        # Verify absorption is significant
        assert np.mean(result.absorptance) > 0.1

    def test_multilayer_stack(self):
        """Test multilayer dielectric stack."""
        wavelengths_nm = np.linspace(400, 800, 40)
        wavelengths_um = wavelengths_nm / 1000

        # Three-layer structure
        n1, d1 = 1.46, 100  # SiO2
        n2, d2 = 2.0, 80    # TiO2
        n3, d3 = 1.38, 120  # MgF2
        n_sub = 1.52        # glass

        n_wl = len(wavelengths_um)

        # Our implementation
        result = calculate_thin_film(
            wavelength_um=wavelengths_um,
            layers=[
                {"thickness_nm": d1, "n": np.full(n_wl, n1), "k": np.zeros(n_wl)},
                {"thickness_nm": d2, "n": np.full(n_wl, n2), "k": np.zeros(n_wl)},
                {"thickness_nm": d3, "n": np.full(n_wl, n3), "k": np.zeros(n_wl)},
            ],
            substrate_n=np.full(n_wl, n_sub),
            substrate_k=np.zeros(n_wl),
        )

        # tmm reference
        n_list = [1.0, n1, n2, n3, n_sub]
        d_list = [np.inf, d1, d2, d3, np.inf]
        R_ref, T_ref = run_tmm_reference(wavelengths_nm, n_list, d_list, 0, 'unpolarized')

        np.testing.assert_allclose(result.reflectance, R_ref, rtol=1e-6)
        np.testing.assert_allclose(result.transmittance, T_ref, rtol=1e-6)

    def test_bare_substrate(self):
        """Test bare substrate (no layers)."""
        wavelengths_nm = np.linspace(400, 800, 20)
        wavelengths_um = wavelengths_nm / 1000

        n_sub = 1.5

        # Our implementation
        result = calculate_thin_film(
            wavelength_um=wavelengths_um,
            layers=[],
            substrate_n=np.full(len(wavelengths_um), n_sub),
            substrate_k=np.zeros(len(wavelengths_um)),
        )

        # tmm reference (just incident and substrate)
        n_list = [1.0, n_sub]
        d_list = [np.inf, np.inf]
        R_ref, T_ref = run_tmm_reference(wavelengths_nm, n_list, d_list, 0, 'unpolarized')

        np.testing.assert_allclose(result.reflectance, R_ref, rtol=1e-10)
        np.testing.assert_allclose(result.transmittance, T_ref, rtol=1e-10)

    def test_quarter_wave_coating(self):
        """Test quarter-wave antireflection coating at design wavelength."""
        design_wavelength_nm = 550
        wavelengths_nm = np.array([design_wavelength_nm])
        wavelengths_um = wavelengths_nm / 1000

        n_sub = 1.52  # glass
        n_coating = np.sqrt(n_sub)  # optimal AR coating
        thickness_nm = design_wavelength_nm / (4 * n_coating)

        # Our implementation
        result = calculate_thin_film(
            wavelength_um=wavelengths_um,
            layers=[{
                "thickness_nm": thickness_nm,
                "n": np.array([n_coating]),
                "k": np.zeros(1),
            }],
            substrate_n=np.array([n_sub]),
            substrate_k=np.zeros(1),
        )

        # tmm reference
        n_list = [1.0, n_coating, n_sub]
        d_list = [np.inf, thickness_nm, np.inf]
        R_ref, T_ref = run_tmm_reference(wavelengths_nm, n_list, d_list, 0, 'unpolarized')

        # Both should show near-zero reflectance at design wavelength
        assert result.reflectance[0] < 0.001
        assert R_ref[0] < 0.001
        np.testing.assert_allclose(result.reflectance, R_ref, atol=1e-10)

    def test_grazing_incidence(self):
        """Test at grazing incidence (high angle)."""
        wavelengths_nm = np.linspace(500, 600, 10)
        wavelengths_um = wavelengths_nm / 1000

        n_layer = 1.5
        n_sub = 2.0
        thickness_nm = 100
        angle_deg = 80  # grazing incidence

        # Our implementation
        result = calculate_thin_film(
            wavelength_um=wavelengths_um,
            layers=[{
                "thickness_nm": thickness_nm,
                "n": np.full(len(wavelengths_um), n_layer),
                "k": np.zeros(len(wavelengths_um)),
            }],
            substrate_n=np.full(len(wavelengths_um), n_sub),
            substrate_k=np.zeros(len(wavelengths_um)),
            angle_deg=angle_deg,
            polarization='s',
        )

        # tmm reference
        n_list = [1.0, n_layer, n_sub]
        d_list = [np.inf, thickness_nm, np.inf]
        R_ref, T_ref = run_tmm_reference(wavelengths_nm, n_list, d_list, angle_deg, 's')

        np.testing.assert_allclose(result.reflectance, R_ref, rtol=1e-5)
        np.testing.assert_allclose(result.transmittance, T_ref, rtol=1e-5)

        # At grazing incidence, reflectance should be high
        assert np.all(result.reflectance > 0.3)

    def test_dispersive_material(self):
        """Test with wavelength-dependent refractive index."""
        wavelengths_nm = np.linspace(400, 700, 30)
        wavelengths_um = wavelengths_nm / 1000

        # Simple Cauchy dispersion: n = A + B/lambda^2
        A, B = 1.5, 0.005  # B in um^2
        n_layer = A + B / wavelengths_um**2
        n_sub = 1.52
        thickness_nm = 150

        n_wl = len(wavelengths_um)

        # Our implementation
        result = calculate_thin_film(
            wavelength_um=wavelengths_um,
            layers=[{
                "thickness_nm": thickness_nm,
                "n": n_layer,
                "k": np.zeros(n_wl),
            }],
            substrate_n=np.full(n_wl, n_sub),
            substrate_k=np.zeros(n_wl),
        )

        # tmm reference (per wavelength)
        R_ref = np.zeros(n_wl)
        T_ref = np.zeros(n_wl)
        for i, (wl_nm, n) in enumerate(zip(wavelengths_nm, n_layer)):
            n_list = [1.0, complex(n), n_sub]
            d_list = [np.inf, thickness_nm, np.inf]
            tmm_result = tmm.unpolarized_RT(n_list, d_list, 0, wl_nm)
            R_ref[i] = tmm_result['R']
            T_ref[i] = tmm_result['T']

        np.testing.assert_allclose(result.reflectance, R_ref, rtol=1e-6)
        np.testing.assert_allclose(result.transmittance, T_ref, rtol=1e-6)

    def test_lwir_sio2_on_znse(self):
        """Test LWIR validation case: SiO2 on ZnSe substrate.

        This is the default validation case in the frontend.
        SiO2 has a strong phonon absorption band around 9 um.
        """
        wavelengths_um = np.linspace(7, 14, 50)
        wavelengths_nm = wavelengths_um * 1000
        n_wl = len(wavelengths_um)

        # Approximate optical constants for LWIR
        # ZnSe is nearly transparent with n ~ 2.4
        n_znse = np.full(n_wl, 2.4)
        k_znse = np.zeros(n_wl)

        # SiO2 has strong absorption around 9 um (Si-O stretching mode)
        # Simplified model: absorption peak at 9 um
        center_um = 9.0
        width_um = 1.5
        k_sio2 = 2.0 * np.exp(-((wavelengths_um - center_um) / width_um)**2)
        n_sio2 = np.full(n_wl, 1.4) + 0.5 * np.exp(-((wavelengths_um - center_um) / width_um)**2)

        thickness_nm = 500

        # Our implementation
        result = calculate_thin_film(
            wavelength_um=wavelengths_um,
            layers=[{
                "thickness_nm": thickness_nm,
                "n": n_sio2,
                "k": k_sio2,
            }],
            substrate_n=n_znse,
            substrate_k=k_znse,
        )

        # tmm reference
        R_ref = np.zeros(n_wl)
        T_ref = np.zeros(n_wl)
        for i in range(n_wl):
            n_list = [1.0, complex(n_sio2[i], k_sio2[i]), complex(n_znse[i], k_znse[i])]
            d_list = [np.inf, thickness_nm, np.inf]
            tmm_result = tmm.unpolarized_RT(n_list, d_list, 0, wavelengths_nm[i])
            R_ref[i] = tmm_result['R']
            T_ref[i] = tmm_result['T']

        np.testing.assert_allclose(result.reflectance, R_ref, rtol=1e-5)
        np.testing.assert_allclose(result.transmittance, T_ref, rtol=1e-5)

        # Verify physics: absorption peak around 9 um
        idx_9um = np.argmin(np.abs(wavelengths_um - 9.0))
        assert result.absorptance[idx_9um] > result.absorptance[0]
        assert result.absorptance[idx_9um] > result.absorptance[-1]


@pytest.mark.skipif(not TMM_AVAILABLE, reason="tmm package not installed")
class TestEnergyConservation:
    """Tests for energy conservation R + T + A = 1."""

    @pytest.mark.parametrize("angle_deg", [0, 30, 45, 60, 75])
    def test_energy_conservation_angles(self, angle_deg):
        """Verify energy conservation at various angles."""
        wavelengths_um = np.linspace(0.5, 1.0, 20)
        n_wl = len(wavelengths_um)

        result = calculate_thin_film(
            wavelength_um=wavelengths_um,
            layers=[{
                "thickness_nm": 100,
                "n": np.full(n_wl, 1.5),
                "k": np.full(n_wl, 0.1),
            }],
            substrate_n=np.full(n_wl, 2.0),
            substrate_k=np.full(n_wl, 0.05),
            angle_deg=angle_deg,
        )

        total = result.reflectance + result.transmittance + result.absorptance
        np.testing.assert_allclose(total, 1.0, rtol=1e-10)

    @pytest.mark.parametrize("polarization", ['s', 'p', 'unpolarized'])
    def test_energy_conservation_polarizations(self, polarization):
        """Verify energy conservation for all polarizations."""
        wavelengths_um = np.linspace(0.5, 1.0, 20)
        n_wl = len(wavelengths_um)

        result = calculate_thin_film(
            wavelength_um=wavelengths_um,
            layers=[{
                "thickness_nm": 100,
                "n": np.full(n_wl, 1.5),
                "k": np.full(n_wl, 0.1),
            }],
            substrate_n=np.full(n_wl, 2.0),
            substrate_k=np.full(n_wl, 0.05),
            angle_deg=45,
            polarization=polarization,
        )

        total = result.reflectance + result.transmittance + result.absorptance
        np.testing.assert_allclose(total, 1.0, rtol=1e-10)
