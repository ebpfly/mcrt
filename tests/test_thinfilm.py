"""Tests for thin film transfer matrix method."""

import numpy as np
import pytest

from mcrt.thinfilm import Layer, ThinFilmStack, ThinFilmResult, calculate_thin_film


class TestLayer:
    """Tests for Layer dataclass."""

    def test_create_layer(self):
        """Test creating a layer."""
        n = np.array([1.5, 1.5, 1.5])
        k = np.array([0.0, 0.0, 0.0])
        layer = Layer(thickness_nm=100, n=n, k=k)

        assert layer.thickness_nm == 100
        np.testing.assert_array_equal(layer.n, n)
        np.testing.assert_array_equal(layer.k, k)

    def test_layer_converts_lists(self):
        """Test that layer converts lists to arrays."""
        layer = Layer(thickness_nm=50, n=[1.5, 1.6], k=[0.0, 0.01])

        assert isinstance(layer.n, np.ndarray)
        assert isinstance(layer.k, np.ndarray)

    def test_layer_mismatched_lengths(self):
        """Test that mismatched n and k lengths raise error."""
        with pytest.raises(ValueError, match="same length"):
            Layer(thickness_nm=100, n=[1.5, 1.6], k=[0.0])


class TestThinFilmStack:
    """Tests for ThinFilmStack."""

    @pytest.fixture
    def wavelengths(self):
        """Sample wavelength array."""
        return np.linspace(0.4, 0.8, 50)

    @pytest.fixture
    def glass_on_silicon(self, wavelengths):
        """Glass (SiO2) on silicon substrate."""
        n_wl = len(wavelengths)

        # Approximate values
        glass_n = np.full(n_wl, 1.46)
        glass_k = np.zeros(n_wl)
        si_n = np.full(n_wl, 3.5)
        si_k = np.full(n_wl, 0.01)

        stack = ThinFilmStack(
            incident_n=np.ones(n_wl),
            incident_k=np.zeros(n_wl),
            substrate_n=si_n,
            substrate_k=si_k,
            wavelength_um=wavelengths,
        )
        stack.add_layer(Layer(thickness_nm=100, n=glass_n, k=glass_k))

        return stack

    def test_create_stack(self, wavelengths):
        """Test creating a stack."""
        n_wl = len(wavelengths)
        stack = ThinFilmStack(
            incident_n=np.ones(n_wl),
            incident_k=np.zeros(n_wl),
            substrate_n=np.full(n_wl, 3.5),
            substrate_k=np.zeros(n_wl),
            wavelength_um=wavelengths,
        )

        assert len(stack.wavelength_um) == n_wl
        assert len(stack.layers) == 0

    def test_add_layer(self, glass_on_silicon):
        """Test adding layers."""
        assert len(glass_on_silicon.layers) == 1
        assert glass_on_silicon.layers[0].thickness_nm == 100

    def test_add_layer_chaining(self, wavelengths):
        """Test method chaining for add_layer."""
        n_wl = len(wavelengths)
        stack = ThinFilmStack(
            incident_n=np.ones(n_wl),
            incident_k=np.zeros(n_wl),
            substrate_n=np.full(n_wl, 3.5),
            substrate_k=np.zeros(n_wl),
            wavelength_um=wavelengths,
        )

        result = stack.add_layer(
            Layer(thickness_nm=100, n=np.full(n_wl, 1.46), k=np.zeros(n_wl))
        ).add_layer(
            Layer(thickness_nm=50, n=np.full(n_wl, 2.0), k=np.zeros(n_wl))
        )

        assert result is stack
        assert len(stack.layers) == 2

    def test_calculate_normal_incidence(self, glass_on_silicon):
        """Test calculation at normal incidence."""
        result = glass_on_silicon.calculate(angle_deg=0)

        assert isinstance(result, ThinFilmResult)
        assert len(result.wavelength_um) == len(glass_on_silicon.wavelength_um)
        assert len(result.reflectance) == len(result.wavelength_um)
        assert len(result.transmittance) == len(result.wavelength_um)
        assert len(result.absorptance) == len(result.wavelength_um)

        # Check physical bounds
        assert np.all(result.reflectance >= 0)
        assert np.all(result.reflectance <= 1)
        assert np.all(result.transmittance >= 0)
        assert np.all(result.transmittance <= 1)
        assert np.all(result.absorptance >= 0)
        assert np.all(result.absorptance <= 1)

        # R + T + A should equal 1
        total = result.reflectance + result.transmittance + result.absorptance
        np.testing.assert_allclose(total, 1.0, rtol=1e-10)

    def test_calculate_oblique_incidence(self, glass_on_silicon):
        """Test calculation at oblique incidence."""
        result = glass_on_silicon.calculate(angle_deg=45)

        assert result.angle_deg == 45
        # R + T + A should still equal 1
        total = result.reflectance + result.transmittance + result.absorptance
        np.testing.assert_allclose(total, 1.0, rtol=1e-10)

    def test_polarization_s(self, glass_on_silicon):
        """Test s-polarization."""
        result = glass_on_silicon.calculate(angle_deg=45, polarization="s")
        assert result.polarization == "s"

    def test_polarization_p(self, glass_on_silicon):
        """Test p-polarization."""
        result = glass_on_silicon.calculate(angle_deg=45, polarization="p")
        assert result.polarization == "p"

    def test_polarization_unpolarized(self, glass_on_silicon):
        """Test unpolarized light."""
        result = glass_on_silicon.calculate(angle_deg=45, polarization="unpolarized")
        assert result.polarization == "unpolarized"

    def test_invalid_polarization(self, glass_on_silicon):
        """Test invalid polarization raises error."""
        with pytest.raises(ValueError, match="polarization"):
            glass_on_silicon.calculate(polarization="invalid")

    def test_bare_substrate(self, wavelengths):
        """Test reflectance from bare substrate (no layers)."""
        n_wl = len(wavelengths)

        # Air-silicon interface
        stack = ThinFilmStack(
            incident_n=np.ones(n_wl),
            incident_k=np.zeros(n_wl),
            substrate_n=np.full(n_wl, 3.5),
            substrate_k=np.zeros(n_wl),
            wavelength_um=wavelengths,
        )

        result = stack.calculate(angle_deg=0)

        # Fresnel reflectance for air-silicon: R = ((n-1)/(n+1))^2
        expected_R = ((3.5 - 1) / (3.5 + 1)) ** 2
        np.testing.assert_allclose(result.reflectance, expected_R, rtol=1e-10)

    def test_quarter_wave_antireflection(self):
        """Test quarter-wave antireflection coating.

        For a quarter-wave coating (thickness = lambda/4n), minimum reflectance
        occurs when n_coating = sqrt(n_substrate * n_incident).
        """
        # Design for 550nm wavelength
        wavelength = np.array([0.55])  # um
        n_sub = 1.52  # glass
        n_opt = np.sqrt(n_sub)  # optimal coating index ~1.23

        # Quarter wave thickness: d = lambda / (4 * n)
        thickness_nm = 550 / (4 * n_opt)  # ~112 nm

        stack = ThinFilmStack(
            incident_n=np.array([1.0]),
            incident_k=np.array([0.0]),
            substrate_n=np.array([n_sub]),
            substrate_k=np.array([0.0]),
            wavelength_um=wavelength,
        )
        stack.add_layer(Layer(
            thickness_nm=thickness_nm,
            n=np.array([n_opt]),
            k=np.array([0.0]),
        ))

        result = stack.calculate(angle_deg=0)

        # At design wavelength, reflectance should be near zero
        assert result.reflectance[0] < 0.001


class TestConvenienceFunction:
    """Tests for calculate_thin_film convenience function."""

    def test_basic_calculation(self):
        """Test basic thin film calculation."""
        wavelengths = np.linspace(0.4, 0.8, 20)
        n_wl = len(wavelengths)

        result = calculate_thin_film(
            wavelength_um=wavelengths,
            layers=[{
                "thickness_nm": 100,
                "n": np.full(n_wl, 1.46),
                "k": np.zeros(n_wl),
            }],
            substrate_n=np.full(n_wl, 3.5),
            substrate_k=np.full(n_wl, 0.01),
        )

        assert isinstance(result, ThinFilmResult)
        assert len(result.wavelength_um) == n_wl

    def test_default_incident_medium(self):
        """Test that air is used as default incident medium."""
        wavelengths = np.linspace(0.4, 0.8, 10)
        n_wl = len(wavelengths)

        result = calculate_thin_film(
            wavelength_um=wavelengths,
            layers=[],
            substrate_n=np.full(n_wl, 1.5),
            substrate_k=np.zeros(n_wl),
        )

        # Air-glass Fresnel reflectance
        expected_R = ((1.5 - 1) / (1.5 + 1)) ** 2
        np.testing.assert_allclose(result.reflectance, expected_R, rtol=1e-10)

    def test_multilayer(self):
        """Test multilayer structure."""
        wavelengths = np.linspace(0.4, 0.8, 20)
        n_wl = len(wavelengths)

        result = calculate_thin_film(
            wavelength_um=wavelengths,
            layers=[
                {"thickness_nm": 100, "n": np.full(n_wl, 1.46), "k": np.zeros(n_wl)},
                {"thickness_nm": 50, "n": np.full(n_wl, 2.0), "k": np.zeros(n_wl)},
                {"thickness_nm": 75, "n": np.full(n_wl, 1.7), "k": np.zeros(n_wl)},
            ],
            substrate_n=np.full(n_wl, 3.5),
            substrate_k=np.zeros(n_wl),
        )

        # Should still satisfy energy conservation
        total = result.reflectance + result.transmittance + result.absorptance
        np.testing.assert_allclose(total, 1.0, rtol=1e-10)


class TestAbsorbingMedia:
    """Tests for absorbing thin films."""

    def test_absorbing_layer(self):
        """Test thin film with absorption."""
        wavelengths = np.linspace(0.4, 0.8, 20)
        n_wl = len(wavelengths)

        # Metal-like layer with significant k
        stack = ThinFilmStack(
            incident_n=np.ones(n_wl),
            incident_k=np.zeros(n_wl),
            substrate_n=np.full(n_wl, 1.5),
            substrate_k=np.zeros(n_wl),
            wavelength_um=wavelengths,
        )
        stack.add_layer(Layer(
            thickness_nm=20,
            n=np.full(n_wl, 1.5),
            k=np.full(n_wl, 2.0),  # Absorbing
        ))

        result = stack.calculate()

        # Should have significant absorptance
        assert np.mean(result.absorptance) > 0.1

        # Energy conservation
        total = result.reflectance + result.transmittance + result.absorptance
        np.testing.assert_allclose(total, 1.0, rtol=1e-10)

    def test_thick_absorbing_layer(self):
        """Test thick absorbing layer (should have low transmittance)."""
        wavelengths = np.array([0.5])

        stack = ThinFilmStack(
            incident_n=np.array([1.0]),
            incident_k=np.array([0.0]),
            substrate_n=np.array([1.5]),
            substrate_k=np.array([0.0]),
            wavelength_um=wavelengths,
        )
        stack.add_layer(Layer(
            thickness_nm=1000,  # 1 micron thick
            n=np.array([1.5]),
            k=np.array([1.0]),  # Strong absorption
        ))

        result = stack.calculate()

        # Thick absorber should have very low transmittance
        assert result.transmittance[0] < 0.01
