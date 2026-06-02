"""Generacion de embeddings con sentence-transformers."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.config import get_section
from src.logger import get_logger

logger = get_logger(__name__)


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...

    @property
    def dimension(self) -> int: ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        normalize: bool = True,
    ):
        cfg = get_section("embeddings")
        self._model_name = model_name or cfg["modelo"]
        self._device = device or cfg.get("dispositivo", "cpu")
        self._batch_size = batch_size or cfg.get("batch_size", 32)
        self._normalize = normalize if normalize is not None else cfg.get("normalizar", True)
        self._model = None
        self._dimension = cfg.get("dimension", 384)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "cargando_modelo_embeddings",
                modelo=self._model_name,
                dispositivo=self._device,
            )
            self._model = SentenceTransformer(
                self._model_name, device=self._device
            )
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        logger.info("generando_embeddings", num_textos=len(texts))
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return np.array(embeddings)


def get_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()
