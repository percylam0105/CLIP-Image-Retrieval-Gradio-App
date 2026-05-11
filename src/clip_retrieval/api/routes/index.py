"""Indexing endpoint — trigger end-to-end ingestion of a local directory."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from clip_retrieval.api.dependencies import get_indexing_service, get_vector_store
from clip_retrieval.api.schemas import IndexRequest, IndexResponse
from clip_retrieval.core.indexing import IndexingService
from clip_retrieval.db.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/index", tags=["index"])


@router.post("/", response_model=IndexResponse)
def index_directory(
    request: IndexRequest,
    indexing_service: IndexingService = Depends(get_indexing_service),
    vector_store: VectorStore = Depends(get_vector_store),
) -> IndexResponse:
    images_dir = Path(request.images_dir) if request.images_dir else None
    try:
        count = indexing_service.index_directory(images_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return IndexResponse(indexed_count=count, collection_info=vector_store.get_collection_info())
