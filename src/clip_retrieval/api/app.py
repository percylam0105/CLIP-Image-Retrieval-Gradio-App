"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from clip_retrieval import __version__
from clip_retrieval.api.routes import health, index, search

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
