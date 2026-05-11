"""Health endpoint — surface model + vector store + object store status."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from clip_retrieval.api.dependencies import (
    get_embedding_service,
    get_object_store,
    get_settings,
    get_vector_store,
)
from clip_retrieval.api.schemas import HealthResponse
from clip_retrieval.config import Settings
from clip_retrieval.core.embedding import EmbeddingService
from clip_retrieval.db.object_store import ObjectStore
from clip_retrieval.db.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
    object_store: ObjectStore = Depends(get_object_store),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    qdrant_info: dict | None
    try:
        qdrant_info = vector_store.get_collection_info()
    except Exception as exc:
        logger.warning("Qdrant health probe failed: %s", exc)
        qdrant_info = {"error": str(exc)}
    return HealthResponse(
        status="ok",
        model_loaded=embedding_service.is_loaded,
        qdrant=qdrant_info,
        minio_bucket=object_store.bucket,
    )
