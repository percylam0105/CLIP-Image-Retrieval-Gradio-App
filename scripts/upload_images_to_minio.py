"""CLI entry point — upload every legacy image into the MinIO bucket."""

from __future__ import annotations

import logging

from clip_retrieval.config import Settings
from clip_retrieval.db.migration import MigrationService
from clip_retrieval.db.object_store import ObjectStore
from clip_retrieval.db.vector_store import VectorStore


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    settings = Settings()
    vector_store = VectorStore(settings)
    object_store = ObjectStore(settings)
    service = MigrationService(settings, vector_store, object_store)

    uploaded = service.upload_images()
    logging.info("Upload finished: %d new objects in bucket %s", uploaded, object_store.bucket)


if __name__ == "__main__":
    main()
