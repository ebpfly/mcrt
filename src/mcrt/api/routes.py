"""FastAPI route definitions."""

import asyncio
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from mcrt.api.models import (
    BatchUpdateResponse,
    CustomMaterialRequest,
    HealthResponse,
    MaterialInfoResponse,
    MaterialListResponse,
    OpticalConstantsResponse,
    SimulationConfigRequest,
    SimulationListResponse,
    SimulationResultsResponse,
    SimulationSessionResponse,
    SimulationStatusEnum,
    StateImportResponse,
    ThinFilmRequest,
    ThinFilmResponse,
)
from mcrt.api.sse import SSEManager, create_sse_response
from mcrt.fos.wrapper import FOSWrapper
from mcrt.fos.output_parser import FOSResult
from mcrt.materials.database import MaterialDatabase, OpticalConstants
from mcrt.materials.custom import CustomMaterial, CustomMaterialLibrary
from mcrt.simulation.manager import SimulationManager, SimulationSession, SimulationStatus
from mcrt.simulation.progressive import ProgressiveSimulation
from mcrt.simulation.state import StatePersistence

import json
from pathlib import Path as PathLib

# Reference data directory
REFERENCE_DATA_DIR = PathLib(__file__).parent.parent / "reference_data"


# === Router Setup ===

router = APIRouter(prefix="/api/v1")


# === Dependencies ===

# Global instances (initialized by app.py)
_fos_wrapper: FOSWrapper | None = None
_material_db: MaterialDatabase | None = None
_simulation_manager: SimulationManager | None = None
_sse_manager: SSEManager | None = None
_custom_materials: CustomMaterialLibrary | None = None


def init_dependencies(
    fos_wrapper: FOSWrapper,
    material_db: MaterialDatabase,
    work_dir: Path,
):
    """Initialize global dependencies.

    Args:
        fos_wrapper: FOSWrapper instance
        material_db: MaterialDatabase instance
        work_dir: Working directory for files
    """
    global _fos_wrapper, _material_db, _simulation_manager, _sse_manager, _custom_materials

    _fos_wrapper = fos_wrapper
    _material_db = material_db
    _simulation_manager = SimulationManager(fos_wrapper)
    _sse_manager = SSEManager()
    _custom_materials = CustomMaterialLibrary()


def get_fos() -> FOSWrapper:
    if _fos_wrapper is None:
        raise HTTPException(status_code=503, detail="FOS not initialized")
    return _fos_wrapper


def get_db() -> MaterialDatabase:
    if _material_db is None:
        raise HTTPException(status_code=503, detail="Material database not initialized")
    return _material_db


def get_manager() -> SimulationManager:
    if _simulation_manager is None:
        raise HTTPException(status_code=503, detail="Simulation manager not initialized")
    return _simulation_manager


def get_sse() -> SSEManager:
    if _sse_manager is None:
        raise HTTPException(status_code=503, detail="SSE manager not initialized")
    return _sse_manager


def get_custom_materials() -> CustomMaterialLibrary:
    if _custom_materials is None:
        raise HTTPException(status_code=503, detail="Custom materials not initialized")
    return _custom_materials


# === Health Check ===


@router.get("/health", response_model=HealthResponse)
async def health_check(
    fos: Annotated[FOSWrapper | None, Depends(lambda: _fos_wrapper)],
    db: Annotated[MaterialDatabase | None, Depends(lambda: _material_db)],
    manager: Annotated[SimulationManager | None, Depends(lambda: _simulation_manager)],
):
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        fos_available=fos is not None,
        database_available=db is not None,
        active_sessions=len(manager.list_sessions()) if manager else 0,
    )


# === Material Endpoints ===


@router.get("/materials", response_model=MaterialListResponse)
async def list_materials(
    db: Annotated[MaterialDatabase, Depends(get_db)],
    shelf: str | None = None,
    search: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """List available materials from the database."""
    materials = db.list_materials(shelf=shelf, search=search)
    total = len(materials)
    materials = materials[offset : offset + limit]

    return MaterialListResponse(
        materials=[
            MaterialInfoResponse(
                material_id=m.material_id,
                name=m.name,
                shelf=m.shelf,
                book=m.book,
                page=m.page,
                wavelength_range_um=m.wavelength_range_um,
            )
            for m in materials
        ],
        total=total,
    )


@router.get("/materials/shelves")
async def list_shelves(db: Annotated[MaterialDatabase, Depends(get_db)]):
    """List available material shelves (categories)."""
    return {"shelves": db.list_shelves()}


# Custom materials routes must come BEFORE the {material_id:path} route
@router.post("/materials/custom")
async def create_custom_material(
    request: CustomMaterialRequest,
    custom: Annotated[CustomMaterialLibrary, Depends(get_custom_materials)],
):
    """Create a custom material with user-defined optical constants."""
    material = CustomMaterial(
        name=request.name,
        wavelength_um=np.array(request.wavelength_um),
        n=np.array(request.n),
        k=np.array(request.k),
        description=request.description,
    )
    custom.add(material)
    return {"status": "created", "name": request.name}


@router.get("/materials/custom/{name}", response_model=OpticalConstantsResponse)
async def get_custom_material(
    name: str,
    custom: Annotated[CustomMaterialLibrary, Depends(get_custom_materials)],
):
    """Get a custom material."""
    try:
        material = custom.get(name)
        oc = material.to_optical_constants()
        return OpticalConstantsResponse(
            wavelength_um=oc.wavelength_um.tolist(),
            n=oc.n.tolist(),
            k=oc.k.tolist(),
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Custom material not found: {name}")


# This route uses :path so it must come AFTER more specific routes
@router.get("/materials/{material_id:path}", response_model=OpticalConstantsResponse)
async def get_material(
    material_id: str,
    db: Annotated[MaterialDatabase, Depends(get_db)],
    wavelength_min_um: float | None = None,
    wavelength_max_um: float | None = None,
):
    """Get optical constants for a material."""
    try:
        wavelength_range = None
        if wavelength_min_um is not None and wavelength_max_um is not None:
            wavelength_range = (wavelength_min_um, wavelength_max_um)

        oc = db.get_optical_constants(material_id, wavelength_range)

        return OpticalConstantsResponse(
            wavelength_um=oc.wavelength_um.tolist(),
            n=oc.n.tolist(),
            k=oc.k.tolist(),
            material_info=MaterialInfoResponse(
                material_id=oc.material_info.material_id,
                name=oc.material_info.name,
                shelf=oc.material_info.shelf,
                book=oc.material_info.book,
                page=oc.material_info.page,
                references=oc.material_info.references,
                comments=oc.material_info.comments,
                wavelength_range_um=oc.material_info.wavelength_range_um,
            )
            if oc.material_info
            else None,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Material not found: {material_id}")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Material data not found: {material_id}")


# === Simulation Endpoints ===


def _session_to_response(session: SimulationSession) -> SimulationSessionResponse:
    """Convert session to response model."""
    results = None
    if session.current_result:
        results = SimulationResultsResponse(
            wavelength_um=session.current_result.wavelength_um.tolist(),
            reflectance=session.current_result.reflectance.tolist(),
            absorptance=session.current_result.absorptance.tolist(),
            transmittance=session.current_result.transmittance.tolist(),
        )

    return SimulationSessionResponse(
        session_id=session.session_id,
        status=SimulationStatusEnum(session.status.value),
        batches_completed=session.batches_completed,
        total_batches=session.total_batches,
        photons_completed=session.photons_completed,
        photons_target=session.photons_target,
        progress_percent=session.progress_percent,
        created_at=session.created_at,
        updated_at=session.updated_at,
        error_message=session.error_message,
        results=results,
    )


@router.post("/simulation/start", response_model=SimulationSessionResponse)
async def start_simulation(
    request: SimulationConfigRequest,
    background_tasks: BackgroundTasks,
    manager: Annotated[SimulationManager, Depends(get_manager)],
    sse: Annotated[SSEManager, Depends(get_sse)],
    fos: Annotated[FOSWrapper, Depends(get_fos)],
):
    """Start a new simulation."""
    # Convert materials
    particle_materials = {
        int(k): OpticalConstants(
            wavelength_um=np.array(v.wavelength_um),
            n=np.array(v.n),
            k=np.array(v.k),
        )
        for k, v in request.particle_materials.items()
    }
    matrix_materials = {
        int(k): OpticalConstants(
            wavelength_um=np.array(v.wavelength_um),
            n=np.array(v.n),
            k=np.array(v.k),
        )
        for k, v in request.matrix_materials.items()
    }

    # Convert layers
    # Note: std_dev from API is relative (0-1), but FOS expects absolute (μm)
    # Convert by multiplying by diameter
    layers = [
        {
            "matrix_id": layer.matrix_id,
            "thickness_um": layer.thickness_um,
            "particles": [
                {
                    "material_id": p.material_id,
                    "diameter_um": p.diameter_um,
                    "volume_fraction": p.volume_fraction,
                    "std_dev": p.std_dev * p.diameter_um,  # Convert relative to absolute
                }
                for p in layer.particles
            ],
        }
        for layer in request.layers
    ]

    # Create session
    session = manager.create_session(
        particle_materials=particle_materials,
        matrix_materials=matrix_materials,
        layers=layers,
        wavelength_start_um=request.wavelength_start_um,
        wavelength_end_um=request.wavelength_end_um,
        wavelength_interval_um=request.wavelength_interval_um,
        photons_target=request.photons_target,
        total_batches=request.n_batches,
    )

    # Run simulation in background
    async def run_simulation():
        import traceback
        print(f"[SIM] Starting simulation for session {session.session_id}")
        progressive = ProgressiveSimulation(fos)
        try:
            batch_count = 0
            async for update in progressive.run_session_progressive(session):
                batch_count += 1
                print(f"[SIM] Batch {batch_count} complete, sending update...")
                await sse.send_batch_update(session.session_id, update)

            print(f"[SIM] Simulation complete, sending completion event")
            await sse.send_completion(
                session.session_id,
                session.status.value,
                session.current_result.to_dict() if session.current_result else None,
            )
        except Exception as e:
            print(f"[SIM] ERROR: {e}")
            traceback.print_exc()
            session.status = SimulationStatus.ERROR
            session.error_message = str(e)
            await sse.send_error(session.session_id, str(e))

    background_tasks.add_task(run_simulation)

    return _session_to_response(session)


@router.get("/simulation/{session_id}", response_model=SimulationSessionResponse)
async def get_simulation(
    session_id: str,
    manager: Annotated[SimulationManager, Depends(get_manager)],
):
    """Get simulation session status and results."""
    try:
        session = manager.get_session(session_id)
        return _session_to_response(session)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")


@router.get("/simulation/{session_id}/stream")
async def stream_simulation(
    session_id: str,
    manager: Annotated[SimulationManager, Depends(get_manager)],
    sse: Annotated[SSEManager, Depends(get_sse)],
):
    """Stream simulation progress via SSE."""
    try:
        manager.get_session(session_id)  # Verify exists
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    queue = sse.subscribe(session_id)
    return create_sse_response(queue, session_id, sse)


@router.post("/simulation/{session_id}/continue", response_model=SimulationSessionResponse)
async def continue_simulation(
    session_id: str,
    background_tasks: BackgroundTasks,
    manager: Annotated[SimulationManager, Depends(get_manager)],
    sse: Annotated[SSEManager, Depends(get_sse)],
    fos: Annotated[FOSWrapper, Depends(get_fos)],
    n_batches: int | None = None,
):
    """Continue a paused simulation."""
    try:
        session = manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    if session.status == SimulationStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Simulation already running")

    if session.status == SimulationStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Simulation already completed")

    # Run continuation in background
    async def run_continuation():
        progressive = ProgressiveSimulation(fos)
        try:
            async for update in progressive.run_session_progressive(session, n_batches):
                await sse.send_batch_update(session.session_id, update)

            await sse.send_completion(
                session.session_id,
                session.status.value,
                session.current_result.to_dict() if session.current_result else None,
            )
        except Exception as e:
            session.status = SimulationStatus.ERROR
            session.error_message = str(e)
            await sse.send_error(session.session_id, str(e))

    background_tasks.add_task(run_continuation)

    return _session_to_response(session)


@router.post("/simulation/{session_id}/stop")
async def stop_simulation(
    session_id: str,
    manager: Annotated[SimulationManager, Depends(get_manager)],
):
    """Stop a running simulation."""
    try:
        manager.stop_session(session_id)
        return {"status": "stopping", "session_id": session_id}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")


@router.delete("/simulation/{session_id}")
async def delete_simulation(
    session_id: str,
    manager: Annotated[SimulationManager, Depends(get_manager)],
):
    """Delete a simulation session."""
    try:
        manager.delete_session(session_id)
        return {"status": "deleted", "session_id": session_id}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")


@router.get("/simulations", response_model=SimulationListResponse)
async def list_simulations(
    manager: Annotated[SimulationManager, Depends(get_manager)],
):
    """List all simulation sessions."""
    sessions = manager.list_sessions()
    return SimulationListResponse(
        sessions=[_session_to_response(s) for s in sessions],
        total=len(sessions),
    )


# === State Management Endpoints ===


@router.get("/simulation/{session_id}/state")
async def export_state(
    session_id: str,
    manager: Annotated[SimulationManager, Depends(get_manager)],
    format: str = "json",
):
    """Export simulation state for saving."""
    try:
        session = manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        filepath = Path(f.name)

    if format == "compact":
        StatePersistence.save_compact(session, filepath)
        # Return JSON metadata file
        return FileResponse(
            filepath,
            media_type="application/json",
            filename=f"mcrt_state_{session_id[:8]}.json",
        )
    else:
        StatePersistence.save_json(session, filepath)
        return FileResponse(
            filepath,
            media_type="application/json",
            filename=f"mcrt_state_{session_id[:8]}.json",
        )


@router.post("/simulation/restore", response_model=StateImportResponse)
async def restore_state(
    file: UploadFile = File(...),
    manager: Annotated[SimulationManager, Depends(get_manager)] = None,
):
    """Restore simulation from saved state file."""
    import json

    content = await file.read()
    data = json.loads(content)

    session = manager.restore_session(data["session"])

    return StateImportResponse(
        session_id=session.session_id,
        status=SimulationStatusEnum(session.status.value),
        message=f"Restored session with {session.batches_completed}/{session.total_batches} batches",
    )


@router.get("/simulation/{session_id}/results/csv")
async def export_results_csv(
    session_id: str,
    manager: Annotated[SimulationManager, Depends(get_manager)],
):
    """Export results as CSV."""
    try:
        session = manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    if session.current_result is None:
        raise HTTPException(status_code=400, detail="No results to export")

    with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        filepath = Path(f.name)

    StatePersistence.export_results_csv(session, filepath)

    return FileResponse(
        filepath,
        media_type="text/csv",
        filename=f"mcrt_results_{session_id[:8]}.csv",
    )


# === Reference Data Endpoints ===


@router.get("/reference")
async def list_reference_data():
    """List available reference reflectance datasets."""
    references = []
    if REFERENCE_DATA_DIR.exists():
        for f in REFERENCE_DATA_DIR.glob("*.json"):
            with open(f) as fp:
                data = json.load(fp)
                references.append({
                    "id": data.get("id"),
                    "name": data.get("name"),
                    "source": data.get("source"),
                    "description": data.get("description"),
                    "measurement_type": data.get("measurement_type"),
                    "particle_size": data.get("particle_size"),
                    "wavelength_range_um": data.get("wavelength_range_um"),
                })
    return {"references": references, "total": len(references)}


@router.get("/reference/{reference_id}")
async def get_reference_data(reference_id: str):
    """Get reference reflectance data by ID."""
    if not REFERENCE_DATA_DIR.exists():
        raise HTTPException(status_code=404, detail="Reference data not found")

    # Try to find the file
    filepath = REFERENCE_DATA_DIR / f"{reference_id}.json"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Reference not found: {reference_id}")

    with open(filepath) as f:
        data = json.load(f)

    return data


# === Thin Film Endpoints ===


@router.post("/thinfilm/calculate", response_model=ThinFilmResponse)
async def calculate_thin_film(request: ThinFilmRequest):
    """Calculate thin film reflectance/transmittance using Transfer Matrix Method.

    This endpoint computes the optical response of multilayer thin films on a substrate.
    Useful for modeling coatings, anti-reflection films, and layered structures.

    The stack is defined from top (incident side) to bottom (substrate):
    - Incident medium (air by default)
    - Layer 0 (topmost thin film)
    - Layer 1
    - ...
    - Substrate (semi-infinite)
    """
    from mcrt.thinfilm import calculate_thin_film as tmm_calculate

    try:
        # Convert request layers to TMM format
        layers = [
            {
                "thickness_nm": layer.thickness_nm,
                "n": np.array(layer.n),
                "k": np.array(layer.k),
            }
            for layer in request.layers
        ]

        result = tmm_calculate(
            wavelength_um=np.array(request.wavelength_um),
            layers=layers,
            substrate_n=np.array(request.substrate_n),
            substrate_k=np.array(request.substrate_k),
            incident_n=np.array(request.incident_n) if request.incident_n else None,
            incident_k=np.array(request.incident_k) if request.incident_k else None,
            angle_deg=request.angle_deg,
            polarization=request.polarization,
        )

        return ThinFilmResponse(
            wavelength_um=result.wavelength_um.tolist(),
            reflectance=result.reflectance.tolist(),
            transmittance=result.transmittance.tolist(),
            absorptance=result.absorptance.tolist(),
            angle_deg=result.angle_deg,
            polarization=result.polarization,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
