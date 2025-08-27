"""Application entry point."""

from fastapi import FastAPI

from .api import api_router
from .core.config import settings
from .core.logging import RequestIDMiddleware, init_logging


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    init_logging(settings.LOG_LEVEL)

    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    app.include_router(api_router)
    return app


app = create_app()
