"""Tests for ``SearchService``."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from clip_retrieval.core.search import SearchService


def _seed_collection(vector_store):
    rng = np.random.default_rng(0)
    vectors = rng.random((5, 512)).astype(np.float32)
    keys = [f"images/i_{i}.jpg" for i in range(5)]
    captions = {f"i_{i}.jpg": f"cap {i}" for i in range(5)}
    vector_store.upsert_batch(keys, vectors, captions)


def test_search_by_text_returns_results(vector_store, fake_embedding_service):
    _seed_collection(vector_store)
    svc = SearchService(fake_embedding_service, vector_store)
    results = svc.search_by_text("red dress", top_k=3)
    assert len(results) == 3
    for r in results:
        assert r.image_path.startswith("images/")
        assert isinstance(r.score, float)


def test_search_by_text_rejects_empty(vector_store, fake_embedding_service):
    svc = SearchService(fake_embedding_service, vector_store)
    with pytest.raises(ValueError):
        svc.search_by_text("")
    with pytest.raises(ValueError):
        svc.search_by_text("   ")


def test_search_by_image_returns_results(vector_store, fake_embedding_service):
    _seed_collection(vector_store)
    svc = SearchService(fake_embedding_service, vector_store)
    img = Image.new("RGB", (4, 4))
    results = svc.search_by_image(img, top_k=2)
    assert len(results) == 2
