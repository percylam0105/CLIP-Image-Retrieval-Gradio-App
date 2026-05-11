"""Tests for ``EmbeddingService`` lazy-loading semantics."""

from __future__ import annotations

from unittest.mock import patch

from config import Settings
from core.embedding import EmbeddingService


def test_embedding_service_not_loaded_at_init():
    service = EmbeddingService(Settings())
    assert service.is_loaded is False


def test_embedding_service_device_resolution_cpu():
    settings = Settings(device="cpu")
    service = EmbeddingService(settings)
    assert service.device == "cpu"


def test_ensure_loaded_calls_transformers_factories():
    settings = Settings(device="cpu")
    service = EmbeddingService(settings)
    with (
        patch("core.embedding.CLIPModel") as MockModel,
        patch("core.embedding.CLIPTokenizer") as MockTok,
        patch("core.embedding.CLIPProcessor") as MockProc,
    ):
        MockModel.from_pretrained.return_value.to.return_value = MockModel.from_pretrained.return_value
        service._ensure_loaded()

        MockModel.from_pretrained.assert_called_once_with(settings.model_id)
        MockTok.from_pretrained.assert_called_once_with(settings.model_id)
        MockProc.from_pretrained.assert_called_once_with(settings.model_id)

    assert service.is_loaded is True
