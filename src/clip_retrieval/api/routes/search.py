"""Search endpoints — text and image queries."""

from __future__ import annotations

import io
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image

from clip_retrieval.api.dependencies import get_image_service, get_search_service
from clip_retrieval.api.schemas import (
    SearchResponse,
    SearchResultItem,
    TextSearchRequest,
)
from clip_retrieval.core.image_service import ImageService
from clip_retrieval.core.schemas import SearchResult
from clip_retrieval.core.search import SearchService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/search", tags=["search"])


def _to_response(
    results: list[SearchResult],
    image_service: ImageService,
    query: str | None,
) -> SearchResponse:
    items: list[SearchResultItem] = []
    for r in results:
        try:
            url = image_service.get_image_url(r.image_path)
        except Exception as exc:
            logger.warning("Failed to presign %s: %s", r.image_path, exc)
            url = None
        items.append(
            SearchResultItem(
                image_path=r.image_path,
                image_url=url,
                score=r.score,
                caption=r.caption,
                filename=r.filename,
            )
        )
    return SearchResponse(results=items, total=len(items), query=query)


@router.post("/text", response_model=SearchResponse)
def search_by_text(
    request: TextSearchRequest,
    search_service: SearchService = Depends(get_search_service),
    image_service: ImageService = Depends(get_image_service),
) -> SearchResponse:
    try:
        results = search_service.search_by_text(request.query, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_response(results, image_service, request.query)


@router.post("/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = 5,
    search_service: SearchService = Depends(get_search_service),
    image_service: ImageService = Depends(get_image_service),
) -> SearchResponse:
    if top_k < 1 or top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be in [1, 100]")
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
    results = search_service.search_by_image(image, top_k)
    return _to_response(results, image_service, None)
