"""FastAPI application factory."""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, index, search

try:
    __version__ = version("clip-image-retrieval")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="CLIP Image Retrieval",
        version=__version__,
        description="Multimodal image retrieval service backed by CLIP, Qdrant and MinIO.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(index.router)

    return app
