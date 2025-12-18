"""FastAPI application for MCRT Monte Carlo Radiative Transfer."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mcrt.api.routes import router, init_dependencies
from mcrt.fos.wrapper import FOSWrapper
from mcrt.materials.database import CombinedMaterialDatabase


def create_app(
    fos_path: str | Path | None = None,
    db_path: str | Path | None = None,
    jena_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        fos_path: Path to FOS source directory
        db_path: Path to refractiveindex.info database
        jena_path: Path to Jena/optool database
        work_dir: Working directory for simulation files
        cors_origins: Allowed CORS origins (default: localhost:3003)

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="MCRT API",
        description="Monte Carlo Radiative Transfer for Particulate Materials",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # Configure CORS
    if cors_origins is None:
        cors_origins = [
            "http://localhost:3003",
            "http://127.0.0.1:3003",
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Set up working directory
    if work_dir is None:
        work_dir = Path.cwd() / ".mcrt_work"
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Initialize FOS wrapper
    try:
        fos_wrapper = FOSWrapper(fos_path=fos_path, work_dir=work_dir)
    except FileNotFoundError as e:
        print(f"Warning: FOS not found: {e}")
        fos_wrapper = None

    # Initialize combined material database (refractiveindex.info + Jena)
    material_db = CombinedMaterialDatabase(
        refractiveindex_path=db_path,
        jena_path=jena_path,
    )

    # Initialize dependencies (material_db is always available, may have both or one database)
    if fos_wrapper:
        init_dependencies(fos_wrapper, material_db, work_dir)

    # Include routes
    app.include_router(router)

    @app.on_event("startup")
    async def startup():
        print("MCRT API starting up...")
        print(f"  FOS available: {fos_wrapper is not None}")
        print(f"  RefractiveIndex.info: {material_db._ri_db is not None}")
        print(f"  Jena database: {material_db._jena_db is not None}")
        print(f"  Work directory: {work_dir}")

    @app.on_event("shutdown")
    async def shutdown():
        print("MCRT API shutting down...")

    return app


# Default application instance
app = create_app()


def main():
    """Run the application with uvicorn."""
    import uvicorn

    host = os.environ.get("MCRT_HOST", "127.0.0.1")
    port = int(os.environ.get("MCRT_PORT", "8003"))

    uvicorn.run(
        "mcrt.api.app:app",
        host=host,
        port=port,
        reload=True,
    )


if __name__ == "__main__":
    main()
