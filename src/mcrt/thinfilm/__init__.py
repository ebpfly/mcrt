"""Thin film optics using the Transfer Matrix Method (TMM).

This module computes reflectance, transmittance, and absorptance for
multilayer thin films on a substrate using the transfer matrix method.

Example:
    >>> from mcrt.thinfilm import ThinFilmStack, Layer
    >>> import numpy as np
    >>>
    >>> # Define wavelengths
    >>> wavelengths = np.linspace(0.4, 0.8, 100)  # 400-800 nm in um
    >>>
    >>> # Create a 100nm SiO2 film on Si substrate
    >>> stack = ThinFilmStack(
    ...     incident_n=np.ones_like(wavelengths),  # air
    ...     incident_k=np.zeros_like(wavelengths),
    ...     substrate_n=si_n,  # silicon n values
    ...     substrate_k=si_k,  # silicon k values
    ...     wavelength_um=wavelengths,
    ... )
    >>> stack.add_layer(Layer(thickness_nm=100, n=sio2_n, k=sio2_k))
    >>>
    >>> # Compute at normal incidence
    >>> result = stack.calculate(angle_deg=0)
    >>> print(result.reflectance)
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass
class Layer:
    """A single thin film layer.

    Attributes:
        thickness_nm: Layer thickness in nanometers.
        n: Real part of refractive index (array, one per wavelength).
        k: Imaginary part of refractive index (array, one per wavelength).
    """

    thickness_nm: float
    n: NDArray[np.floating]
    k: NDArray[np.floating]

    def __post_init__(self):
        self.n = np.asarray(self.n)
        self.k = np.asarray(self.k)
        if len(self.n) != len(self.k):
            raise ValueError("n and k arrays must have same length")


@dataclass
class ThinFilmResult:
    """Results from thin film calculation.

    Attributes:
        wavelength_um: Wavelength array in micrometers.
        reflectance: Hemispherical reflectance [0-1].
        transmittance: Hemispherical transmittance [0-1].
        absorptance: Absorptance [0-1], computed as 1 - R - T.
        angle_deg: Angle of incidence in degrees.
        polarization: Polarization state ('s', 'p', or 'unpolarized').
    """

    wavelength_um: NDArray[np.floating]
    reflectance: NDArray[np.floating]
    transmittance: NDArray[np.floating]
    absorptance: NDArray[np.floating]
    angle_deg: float
    polarization: str


@dataclass
class ThinFilmStack:
    """A stack of thin film layers on a substrate.

    The stack is defined from top to bottom:
    - Incident medium (semi-infinite, e.g., air)
    - Layer 0 (topmost thin film)
    - Layer 1
    - ...
    - Layer N-1 (bottommost thin film)
    - Substrate (semi-infinite)

    Attributes:
        incident_n: Refractive index (real part) of incident medium.
        incident_k: Refractive index (imag part) of incident medium.
        substrate_n: Refractive index (real part) of substrate.
        substrate_k: Refractive index (imag part) of substrate.
        wavelength_um: Wavelength array in micrometers.
        layers: List of Layer objects (added via add_layer).
    """

    incident_n: NDArray[np.floating]
    incident_k: NDArray[np.floating]
    substrate_n: NDArray[np.floating]
    substrate_k: NDArray[np.floating]
    wavelength_um: NDArray[np.floating]
    layers: list[Layer] = field(default_factory=list)

    def __post_init__(self):
        self.incident_n = np.asarray(self.incident_n)
        self.incident_k = np.asarray(self.incident_k)
        self.substrate_n = np.asarray(self.substrate_n)
        self.substrate_k = np.asarray(self.substrate_k)
        self.wavelength_um = np.asarray(self.wavelength_um)

        n_wl = len(self.wavelength_um)
        for name, arr in [
            ("incident_n", self.incident_n),
            ("incident_k", self.incident_k),
            ("substrate_n", self.substrate_n),
            ("substrate_k", self.substrate_k),
        ]:
            if len(arr) != n_wl:
                raise ValueError(f"{name} length {len(arr)} != wavelength length {n_wl}")

    def add_layer(self, layer: Layer) -> "ThinFilmStack":
        """Add a layer to the stack (topmost layer added first).

        Args:
            layer: Layer to add.

        Returns:
            self, for method chaining.
        """
        if len(layer.n) != len(self.wavelength_um):
            raise ValueError(
                f"Layer n length {len(layer.n)} != wavelength length {len(self.wavelength_um)}"
            )
        self.layers.append(layer)
        return self

    def calculate(
        self,
        angle_deg: float = 0.0,
        polarization: str = "unpolarized",
    ) -> ThinFilmResult:
        """Calculate reflectance and transmittance using transfer matrix method.

        Args:
            angle_deg: Angle of incidence in degrees (0 = normal incidence).
            polarization: 's', 'p', or 'unpolarized' (average of s and p).

        Returns:
            ThinFilmResult with reflectance, transmittance, absorptance.
        """
        if polarization not in ("s", "p", "unpolarized"):
            raise ValueError(f"polarization must be 's', 'p', or 'unpolarized', got {polarization}")

        angle_rad = np.deg2rad(angle_deg)
        wavelength_m = self.wavelength_um * 1e-6

        # Complex refractive indices
        n_inc = self.incident_n + 1j * self.incident_k
        n_sub = self.substrate_n + 1j * self.substrate_k

        # Snell's law for complex angles
        sin_theta_inc = np.sin(angle_rad)
        cos_theta_inc = np.cos(angle_rad)

        # Angle in substrate (complex for absorbing materials)
        sin_theta_sub = n_inc / n_sub * sin_theta_inc
        cos_theta_sub = np.sqrt(1 - sin_theta_sub**2)

        if polarization == "unpolarized":
            # Calculate both polarizations and average
            r_s, t_s = self._calculate_polarization(
                wavelength_m, n_inc, n_sub, sin_theta_inc, cos_theta_inc, cos_theta_sub, "s"
            )
            r_p, t_p = self._calculate_polarization(
                wavelength_m, n_inc, n_sub, sin_theta_inc, cos_theta_inc, cos_theta_sub, "p"
            )
            R = 0.5 * (np.abs(r_s) ** 2 + np.abs(r_p) ** 2)
            # Transmittance correction for different media
            factor = np.real(n_sub * cos_theta_sub) / np.real(n_inc * cos_theta_inc)
            T = 0.5 * factor * (np.abs(t_s) ** 2 + np.abs(t_p) ** 2)
        else:
            r, t = self._calculate_polarization(
                wavelength_m, n_inc, n_sub, sin_theta_inc, cos_theta_inc, cos_theta_sub, polarization
            )
            R = np.abs(r) ** 2
            factor = np.real(n_sub * cos_theta_sub) / np.real(n_inc * cos_theta_inc)
            T = factor * np.abs(t) ** 2

        # Ensure physical bounds
        R = np.clip(np.real(R), 0, 1)
        T = np.clip(np.real(T), 0, 1 - R)
        A = 1 - R - T

        return ThinFilmResult(
            wavelength_um=self.wavelength_um.copy(),
            reflectance=R,
            transmittance=T,
            absorptance=A,
            angle_deg=angle_deg,
            polarization=polarization,
        )

    def _calculate_polarization(
        self,
        wavelength_m: NDArray,
        n_inc: NDArray,
        n_sub: NDArray,
        sin_theta_inc: float,
        cos_theta_inc: float,
        cos_theta_sub: NDArray,
        polarization: str,
    ) -> tuple[NDArray, NDArray]:
        """Calculate r and t coefficients for a single polarization."""
        n_wl = len(wavelength_m)

        # Build transfer matrix for each wavelength
        # M = D_inc^-1 * P_1 * D_1 * D_1^-1 * P_2 * D_2 * ... * D_sub
        # where D = dynamical matrix, P = propagation matrix

        # Initialize with identity matrices for each wavelength
        M = np.zeros((n_wl, 2, 2), dtype=complex)
        M[:, 0, 0] = 1
        M[:, 1, 1] = 1

        # Current medium is incident medium
        n_curr = n_inc
        cos_theta_curr = np.full(n_wl, cos_theta_inc, dtype=complex)

        for layer in self.layers:
            n_layer = layer.n + 1j * layer.k
            thickness_m = layer.thickness_nm * 1e-9

            # Angle in layer (Snell's law)
            sin_theta_layer = n_inc / n_layer * sin_theta_inc
            cos_theta_layer = np.sqrt(1 - sin_theta_layer**2)

            # Interface matrix from current medium to layer
            D_int = self._interface_matrix(n_curr, cos_theta_curr, n_layer, cos_theta_layer, polarization)

            # Propagation matrix through layer
            P = self._propagation_matrix(n_layer, cos_theta_layer, thickness_m, wavelength_m)

            # Update total matrix: M = M @ D_int @ P
            for i in range(n_wl):
                M[i] = M[i] @ D_int[i] @ P[i]

            n_curr = n_layer
            cos_theta_curr = cos_theta_layer

        # Final interface to substrate
        D_final = self._interface_matrix(n_curr, cos_theta_curr, n_sub, cos_theta_sub, polarization)
        for i in range(n_wl):
            M[i] = M[i] @ D_final[i]

        # Extract r and t from transfer matrix
        # M relates [E_inc+, E_inc-] to [E_sub+, E_sub-]
        # With E_sub- = 0 (no backward wave in substrate):
        # r = M[1,0] / M[0,0], t = 1 / M[0,0]
        r = M[:, 1, 0] / M[:, 0, 0]
        t = 1.0 / M[:, 0, 0]

        return r, t

    def _interface_matrix(
        self,
        n1: NDArray,
        cos1: NDArray,
        n2: NDArray,
        cos2: NDArray,
        polarization: str,
    ) -> NDArray:
        """Compute interface matrix between two media."""
        n_wl = len(n1)
        D = np.zeros((n_wl, 2, 2), dtype=complex)

        if polarization == "s":
            # s-polarization (TE): E perpendicular to plane of incidence
            r = (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)
            t = 2 * n1 * cos1 / (n1 * cos1 + n2 * cos2)
        else:
            # p-polarization (TM): E in plane of incidence
            r = (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2)
            t = 2 * n1 * cos1 / (n2 * cos1 + n1 * cos2)

        # Interface matrix
        D[:, 0, 0] = 1
        D[:, 0, 1] = r
        D[:, 1, 0] = r
        D[:, 1, 1] = 1
        D = D / t[:, np.newaxis, np.newaxis]

        return D

    def _propagation_matrix(
        self,
        n: NDArray,
        cos_theta: NDArray,
        thickness_m: float,
        wavelength_m: NDArray,
    ) -> NDArray:
        """Compute propagation matrix through a layer."""
        n_wl = len(n)
        P = np.zeros((n_wl, 2, 2), dtype=complex)

        # Phase accumulated traversing the layer
        delta = 2 * np.pi * n * cos_theta * thickness_m / wavelength_m

        P[:, 0, 0] = np.exp(-1j * delta)
        P[:, 1, 1] = np.exp(1j * delta)

        return P


def calculate_thin_film(
    wavelength_um: Sequence[float],
    layers: list[dict],
    substrate_n: Sequence[float],
    substrate_k: Sequence[float],
    incident_n: Sequence[float] | None = None,
    incident_k: Sequence[float] | None = None,
    angle_deg: float = 0.0,
    polarization: str = "unpolarized",
) -> ThinFilmResult:
    """Convenience function for thin film calculations.

    Args:
        wavelength_um: Wavelength array in micrometers.
        layers: List of dicts with keys 'thickness_nm', 'n', 'k'.
        substrate_n: Substrate refractive index (real part).
        substrate_k: Substrate refractive index (imaginary part).
        incident_n: Incident medium n (default: 1.0 = air).
        incident_k: Incident medium k (default: 0.0 = transparent).
        angle_deg: Angle of incidence in degrees.
        polarization: 's', 'p', or 'unpolarized'.

    Returns:
        ThinFilmResult with R, T, A spectra.

    Example:
        >>> result = calculate_thin_film(
        ...     wavelength_um=np.linspace(0.4, 0.8, 100),
        ...     layers=[
        ...         {"thickness_nm": 100, "n": sio2_n, "k": sio2_k},
        ...         {"thickness_nm": 50, "n": tin_n, "k": tin_k},
        ...     ],
        ...     substrate_n=si_n,
        ...     substrate_k=si_k,
        ... )
    """
    wavelength_um = np.asarray(wavelength_um)
    n_wl = len(wavelength_um)

    if incident_n is None:
        incident_n = np.ones(n_wl)
    if incident_k is None:
        incident_k = np.zeros(n_wl)

    stack = ThinFilmStack(
        incident_n=np.asarray(incident_n),
        incident_k=np.asarray(incident_k),
        substrate_n=np.asarray(substrate_n),
        substrate_k=np.asarray(substrate_k),
        wavelength_um=wavelength_um,
    )

    for layer_dict in layers:
        stack.add_layer(
            Layer(
                thickness_nm=layer_dict["thickness_nm"],
                n=np.asarray(layer_dict["n"]),
                k=np.asarray(layer_dict["k"]),
            )
        )

    return stack.calculate(angle_deg=angle_deg, polarization=polarization)


__all__ = [
    "Layer",
    "ThinFilmResult",
    "ThinFilmStack",
    "calculate_thin_film",
]
