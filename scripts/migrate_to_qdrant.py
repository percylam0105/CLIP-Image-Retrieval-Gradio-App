"""CLI entry point — migrate legacy df.csv + .npy artefacts into Qdrant."""

from __future__ import annotations

import logging

from config import Settings
from db.migration import MigrationService
from db.object_store import ObjectStore
from db.vector_store import VectorStore


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    settings = Settings()
    vector_store = VectorStore(settings)
    object_store = ObjectStore(settings)
    service = MigrationService(settings, vector_store, object_store)

    upserted = service.migrate_embeddings_only()
    logging.info("Migration finished: %d embeddings upserted", upserted)


if __name__ == "__main__":
    main()
