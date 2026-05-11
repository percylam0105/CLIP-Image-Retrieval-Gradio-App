"""Qdrant-backed vector store wrapper."""

from __future__ import annotations

import logging
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    SearchParams,
    VectorParams,
)

from clip_retrieval.config import Settings

logger = logging.getLogger(__name__)

VECTOR_DIM = 512  # CLIP ViT-B/16 latent dim
BATCH_SIZE = 100


class VectorStore:
    """Thin wrapper around ``qdrant_client`` exposing collection + search primitives."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.collection_name = settings.qdrant_collection
        self.client = self._init_client()
        self._ensure_collection()

    def _init_client(self) -> QdrantClient:
        mode = self.settings.qdrant_mode
        if mode == "memory":
            return QdrantClient(":memory:")
        if mode == "local":
            return QdrantClient(path=self.settings.qdrant_path)
        if mode == "remote":
            return QdrantClient(
                url=self.settings.qdrant_url, api_key=self.settings.qdrant_api_key
            )
        raise ValueError(f"Unknown qdrant_mode: {mode!r}. Expected memory|local|remote.")

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in existing:
            return
        logger.info("Creating Qdrant collection %s", self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=32, ef_construct=200),
        )

    def upsert_batch(
        self,
        object_keys: list[str],
        embeddings: np.ndarray,
        captions: dict[str, str] | None = None,
    ) -> None:
        """Upsert ``(key, embedding)`` pairs in batches of ``BATCH_SIZE``."""
        captions = captions or {}
        points: list[PointStruct] = []
        for key, emb in zip(object_keys, embeddings, strict=False):
            filename = key.split("/")[-1]
            caption = captions.get(filename)
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, key)),
                    vector=np.asarray(emb).flatten().tolist(),
                    payload={
                        "image_path": key,
                        "caption": caption,
                        "filename": filename,
                    },
                )
            )

        for i in range(0, len(points), BATCH_SIZE):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + BATCH_SIZE],
            )

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=np.asarray(query_vector).flatten().tolist(),
            limit=top_k,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=128),
        )
        return [
            {
                "image_path": hit.payload["image_path"],
                "caption": hit.payload.get("caption"),
                "score": float(hit.score),
                "filename": hit.payload.get("filename"),
            }
            for hit in response.points
        ]

    def get_collection_info(self) -> dict:
        info = self.client.get_collection(self.collection_name)
        vectors_count = getattr(info, "vectors_count", None)
        if vectors_count is None:
            vectors_count = getattr(info, "indexed_vectors_count", 0) or 0
        points_count = getattr(info, "points_count", 0) or 0
        status = info.status
        return {
            "name": self.collection_name,
            "vectors_count": int(vectors_count),
            "points_count": int(points_count),
            "status": status.value if hasattr(status, "value") else str(status),
        }

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
