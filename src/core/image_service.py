"""Thin facade over the object store for image URL/byte access."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from db.object_store import ObjectStore


class ImageService:
    """Translate object keys into presigned URLs and raw byte streams."""

    def __init__(self, object_store: "ObjectStore"):
        self.object_store = object_store

    def get_image_url(self, object_key: str, expires: int = 3600) -> str:
        return self.object_store.get_presigned_url(object_key, expires)

    def get_image_bytes(self, object_key: str) -> bytes:
        return self.object_store.get_object(object_key)
