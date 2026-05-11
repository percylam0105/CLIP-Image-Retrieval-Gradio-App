"""CLIP embedding service with lazy model loading."""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

from clip_retrieval.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wrap a HuggingFace CLIP model behind a lazy, side-effect-free API."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: CLIPModel | None = None
        self._tokenizer: CLIPTokenizer | None = None
        self._processor: CLIPProcessor | None = None

    @property
    def device(self) -> str:
        return self.settings.device or ("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading CLIP model %s on %s", self.settings.model_id, self.device)
        model = CLIPModel.from_pretrained(self.settings.model_id).to(self.device)
        model.eval()
        self._model = model
        self._tokenizer = CLIPTokenizer.from_pretrained(self.settings.model_id)
        self._processor = CLIPProcessor.from_pretrained(self.settings.model_id)

    @torch.no_grad()
    def get_text_features(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
        return self._model.get_text_features(**inputs).cpu().numpy()

    @torch.no_grad()
    def get_image_features(self, image) -> np.ndarray:
        self._ensure_loaded()
        assert self._processor is not None and self._model is not None
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        return self._model.get_image_features(**inputs).cpu().numpy()
