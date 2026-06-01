"""Busqueda hibrida: combina dense retrieval (embeddings) con sparse (BM25)."""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import get_section
from src.embeddings.embedder import Embedder
from src.logger import get_logger
from src.retrieval.vector_store import ChromaVectorStore

logger = get_logger(__name__)


class HybridSearcher:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedder: Embedder,
        alpha: float | None = None,
        rrf_k: int | None = None,
    ):
        cfg = get_section("retrieval")
        self._store = vector_store
        self._embedder = embedder
        self._alpha = alpha if alpha is not None else cfg.get("hybrid_alpha", 0.7)
        self._rrf_k = rrf_k if rrf_k is not None else cfg.get("rrf_k", 60)
        self._bm25: BM25Okapi | None = None
        self._corpus_chunks: list[dict] | None = None

    def build_bm25_index(self) -> None:
        collection = self._store._collection
        all_data = collection.get(include=["documents", "metadatas"])

        if not all_data["documents"]:
            logger.warning("bm25_sin_documentos")
            return

        tokenized = [doc.lower().split() for doc in all_data["documents"]]
        self._bm25 = BM25Okapi(tokenized)
        self._corpus_chunks = [
            {
                "id": all_data["ids"][i],
                "texto": all_data["documents"][i],
                "metadata": all_data["metadatas"][i] if all_data["metadatas"] else {},
            }
            for i in range(len(all_data["ids"]))
        ]
        logger.info("bm25_index_construido", num_docs=len(tokenized))

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_emb = self._embedder.encode([query])[0]
        dense_results = self._store.search(query_emb, top_k=top_k * 2)

        if self._bm25 is None or self._alpha >= 1.0:
            return dense_results[:top_k]

        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)
        bm25_ranked = sorted(
            enumerate(bm25_scores), key=lambda x: x[1], reverse=True
        )[:top_k * 2]

        sparse_results = []
        for idx, score in bm25_ranked:
            if score > 0 and self._corpus_chunks:
                chunk = self._corpus_chunks[idx]
                sparse_results.append({**chunk, "bm25_score": float(score)})

        return self._reciprocal_rank_fusion(dense_results, sparse_results, top_k)

    def _reciprocal_rank_fusion(
        self,
        dense: list[dict],
        sparse: list[dict],
        top_k: int,
    ) -> list[dict]:
        k = self._rrf_k
        alpha = self._alpha
        scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        for rank, hit in enumerate(dense):
            doc_id = hit["id"]
            scores[doc_id] = scores.get(doc_id, 0) + alpha / (k + rank + 1)
            doc_map[doc_id] = hit

        for rank, hit in enumerate(sparse):
            doc_id = hit["id"]
            scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = hit

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, rrf_score in ranked:
            hit = doc_map[doc_id]
            hit["rrf_score"] = rrf_score
            results.append(hit)

        logger.info(
            "busqueda_hibrida",
            alpha=self._alpha,
            dense_hits=len(dense),
            sparse_hits=len(sparse),
            resultado_final=len(results),
        )
        return results
