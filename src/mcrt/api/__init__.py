"""FastAPI web API module.

Provides REST and SSE endpoints for the MCRT simulator.
"""

from mcrt.api.app import create_app, app
from mcrt.api.routes import router

__all__ = ["create_app", "app", "router"]
