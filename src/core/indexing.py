"""Pipeline that scans local images, uploads them to MinIO, and indexes Qdrant."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import Settings
from core.embedding import EmbeddingService

if TYPE_CHECKING:
    from db.object_store import ObjectStore
    from db.vector_store import VectorStore

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


class IndexingService:
    """End-to-end indexing of a local image directory."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: "VectorStore",
        object_store: "ObjectStore",
        settings: Settings,
    ):
        self.embedding = embedding_service
        self.vector_store = vector_store
        self.object_store = object_store
        self.settings = settings

    def index_directory(self, images_dir: Path | None = None) -> int:
        """Scan ``images_dir``, push every image to MinIO, and upsert vectors."""
        images_dir = images_dir or self.settings.legacy_images_path
        if images_dir is None or not Path(images_dir).exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        images_dir = Path(images_dir)

        captions: dict[str, str] = {}
        if self.settings.captions_path and self.settings.captions_path.exists():
            with open(self.settings.captions_path) as f:
                captions = json.load(f)

        image_files = [p for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS]

        paths: list[str] = []
        embeddings: list[np.ndarray] = []
        for img_path in tqdm(image_files, desc="Encoding & uploading images"):
            try:
                with Image.open(img_path) as img:
                    emb = self.embedding.get_image_features(img)
                object_key = f"images/{img_path.name}"
                self.object_store.upload_file(str(img_path), object_key)
                paths.append(object_key)
                embeddings.append(emb.flatten())
            except Exception as exc:
                logger.warning("Failed to process %s: %s", img_path, exc)

        if not embeddings:
            logger.info("No images indexed from %s", images_dir)
            return 0

        self.vector_store.upsert_batch(paths, np.array(embeddings), captions)
        logger.info("Indexed %d images", len(paths))
        return len(paths)
