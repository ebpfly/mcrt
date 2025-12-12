"""Pydantic models for API request/response validation."""

from datetime import datetime
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


# === Enums ===


class SimulationStatusEnum(str, Enum):
    """Simulation status values."""

    idle = "idle"
    running = "running"
    paused = "paused"
    completed = "completed"
    error = "error"


# === Material Models ===


class MaterialInfoResponse(BaseModel):
    """Information about a material in the database."""

    material_id: str
    name: str
    shelf: str
    book: str
    page: str
    references: str = ""
    comments: str = ""
    wavelength_range_um: tuple[float, float] | None = None


class OpticalConstantsResponse(BaseModel):
    """Optical constants data."""

    wavelength_um: list[float]
    n: list[float]
    k: list[float]
    material_info: MaterialInfoResponse | None = None


class MaterialListResponse(BaseModel):
    """List of materials."""

    materials: list[MaterialInfoResponse]
    total: int


class CustomMaterialRequest(BaseModel):
    """Request to create a custom material."""

    name: str
    wavelength_um: list[float]
    n: list[float]
    k: list[float]
    description: str = ""


# === Layer/Particle Models ===


class ParticleConfigRequest(BaseModel):
    """Particle configuration within a layer."""

    material_id: int = Field(ge=1, description="Material ID reference")
    diameter_um: float = Field(gt=0, description="Particle diameter in micrometers")
    volume_fraction: float = Field(
        ge=0, le=100, description="Volume fraction as percentage (0-100)"
    )
    std_dev: float = Field(ge=0, default=0.0, description="Size distribution std dev as fraction of diameter (0-1)")


class LayerConfigRequest(BaseModel):
    """Layer configuration."""

    matrix_id: int = Field(ge=1, description="Matrix material ID reference")
    thickness_um: float = Field(gt=0, description="Layer thickness in micrometers")
    particles: list[ParticleConfigRequest]


# === Simulation Models ===


class SimulationConfigRequest(BaseModel):
    """Request to start a new simulation."""

    # Material definitions (inline optical constants)
    particle_materials: dict[str, OpticalConstantsResponse] = Field(
        description="Dict mapping material ID to optical constants"
    )
    matrix_materials: dict[str, OpticalConstantsResponse] = Field(
        description="Dict mapping material ID to optical constants"
    )

    # Layer configuration
    layers: list[LayerConfigRequest]

    # Wavelength range
    wavelength_start_um: float = Field(default=0.3, gt=0)
    wavelength_end_um: float = Field(default=2.5, gt=0)
    wavelength_interval_um: float = Field(default=0.01, gt=0)

    # Photon settings
    photons_target: int = Field(default=100000, gt=0)
    n_batches: int = Field(default=10, ge=1, le=100)


class SimulationResultsResponse(BaseModel):
    """Simulation results data."""

    wavelength_um: list[float]
    reflectance: list[float]
    absorptance: list[float]
    transmittance: list[float]


class SimulationSessionResponse(BaseModel):
    """Full simulation session state."""

    session_id: str
    status: SimulationStatusEnum
    batches_completed: int
    total_batches: int
    photons_completed: int
    photons_target: int
    progress_percent: float
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None
    results: SimulationResultsResponse | None = None


class SimulationListResponse(BaseModel):
    """List of simulation sessions."""

    sessions: list[SimulationSessionResponse]
    total: int


class BatchUpdateResponse(BaseModel):
    """SSE event data for batch completion."""

    event: Literal["batch_complete"] = "batch_complete"
    batch_number: int
    total_batches: int
    photons_completed: int
    photons_target: int
    progress_percent: float
    results: SimulationResultsResponse
    timestamp: datetime


class SimulationCompleteResponse(BaseModel):
    """SSE event data for simulation completion."""

    event: Literal["simulation_complete"] = "simulation_complete"
    session_id: str
    status: SimulationStatusEnum
    results: SimulationResultsResponse
    timestamp: datetime


class SimulationErrorResponse(BaseModel):
    """SSE event data for errors."""

    event: Literal["error"] = "error"
    session_id: str
    error: str
    timestamp: datetime


# === State Management ===


class StateExportRequest(BaseModel):
    """Request to export session state."""

    session_id: str
    format: Literal["json", "compact"] = "json"


class StateImportResponse(BaseModel):
    """Response after importing state."""

    session_id: str
    status: SimulationStatusEnum
    message: str


# === Health Check ===


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = "ok"
    version: str
    fos_available: bool
    database_available: bool
    active_sessions: int


# === Thin Film Models ===


class ThinFilmLayerRequest(BaseModel):
    """A single thin film layer configuration."""

    thickness_nm: float = Field(gt=0, description="Layer thickness in nanometers")
    material_id: str = Field(description="Material ID from database")


class ThinFilmLayerInlineRequest(BaseModel):
    """A single thin film layer with inline optical constants."""

    thickness_nm: float = Field(gt=0, description="Layer thickness in nanometers")
    n: list[float] = Field(description="Refractive index (real part)")
    k: list[float] = Field(description="Refractive index (imaginary part)")


class ThinFilmRequest(BaseModel):
    """Request to calculate thin film reflectance/transmittance."""

    wavelength_um: list[float] = Field(description="Wavelength array in micrometers")
    layers: list[ThinFilmLayerInlineRequest] = Field(
        description="List of thin film layers (top to bottom)"
    )
    substrate_n: list[float] = Field(description="Substrate refractive index (real part)")
    substrate_k: list[float] = Field(description="Substrate refractive index (imaginary part)")
    incident_n: list[float] | None = Field(
        default=None, description="Incident medium n (default: 1.0 = air)"
    )
    incident_k: list[float] | None = Field(
        default=None, description="Incident medium k (default: 0.0 = transparent)"
    )
    angle_deg: float = Field(default=0.0, ge=0, lt=90, description="Angle of incidence in degrees")
    polarization: Literal["s", "p", "unpolarized"] = Field(
        default="unpolarized", description="Polarization state"
    )


class ThinFilmResponse(BaseModel):
    """Response from thin film calculation."""

    wavelength_um: list[float]
    reflectance: list[float]
    transmittance: list[float]
    absorptance: list[float]
    angle_deg: float
    polarization: str
