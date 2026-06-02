"""Reranking con cross-encoder para mejorar la precision del retrieval."""

from __future__ import annotations

from src.config import get_section
from src.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None):
        cfg = get_section("retrieval")
        self._model_name = model_name or cfg.get(
            "modelo_reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("cargando_reranker", modelo=self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, results: list[dict], top_k: int = 5) -> list[dict]:
        if not results:
            return []

        model = self._load_model()
        pairs = [(query, r["texto"]) for r in results]
        scores = model.predict(pairs)

        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        logger.info(
            "reranking_completado",
            input_docs=len(results),
            output_docs=min(top_k, len(reranked)),
        )
        return reranked[:top_k]
