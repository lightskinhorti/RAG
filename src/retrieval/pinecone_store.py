"""Implementacion Pinecone del vector store para despliegue cloud."""

from __future__ import annotations

import os
import uuid

import numpy as np

from src.config import get_section
from src.ingestion.chunker import Chunk
from src.logger import get_logger
from src.retrieval.vector_store import VectorStore

logger = get_logger(__name__)


class PineconeVectorStore(VectorStore):
    """Vector store backed by Pinecone for cloud-scale deployments."""

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        namespace: str | None = None,
        dimension: int | None = None,
    ):
        cfg = get_section("vector_store")
        pinecone_cfg = cfg.get("pinecone", {})

        self._api_key = api_key or os.getenv(
            "PINECONE_API_KEY", pinecone_cfg.get("api_key", "")
        )
        self._index_name = index_name or pinecone_cfg.get(
            "index_name", "boe-legal-docs"
        )
        self._namespace = namespace or pinecone_cfg.get("namespace", "default")
        self._dimension = dimension or cfg.get("dimension", 384)

        if not self._api_key:
            raise ValueError(
                "Pinecone API key requerida. Configura PINECONE_API_KEY en .env "
                "o pinecone.api_key en configs/default.yaml"
            )

        try:
            from pinecone import Pinecone

            self._pc = Pinecone(api_key=self._api_key)

            # Create index if it doesn't exist
            existing = [idx.name for idx in self._pc.list_indexes()]
            if self._index_name not in existing:
                from pinecone import ServerlessSpec

                self._pc.create_index(
                    name=self._index_name,
                    dimension=self._dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                logger.info("pinecone_index_creado", nombre=self._index_name)

            self._index = self._pc.Index(self._index_name)
            stats = self._index.describe_index_stats()
            logger.info(
                "pinecone_store_inicializado",
                index=self._index_name,
                namespace=self._namespace,
                vectores=stats.total_vector_count,
            )
        except ImportError:
            raise ImportError("Instala pinecone: pip install pinecone")
        except Exception as e:
            raise ConnectionError(f"Error conectando a Pinecone: {e}") from e

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        batch_size = 100
        vectors = []
        for chunk, emb in zip(chunks, embeddings):
            vec_id = str(uuid.uuid4())
            metadata = {**chunk.metadata, "texto": chunk.texto[:1000]}
            vectors.append((vec_id, emb.tolist(), metadata))

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._index.upsert(vectors=batch, namespace=self._namespace)

        logger.info("pinecone_chunks_indexados", total=len(chunks))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        query_params: dict = {
            "vector": query_embedding.tolist(),
            "top_k": top_k,
            "include_metadata": True,
            "namespace": self._namespace,
        }

        # Convert where filter to Pinecone filter format
        if where:
            query_params["filter"] = self._convert_filter(where)

        results = self._index.query(**query_params)

        hits = []
        for match in results.matches:
            metadata = dict(match.metadata) if match.metadata else {}
            texto = metadata.pop("texto", "")
            hits.append(
                {
                    "id": match.id,
                    "texto": texto,
                    "metadata": metadata,
                    "score": float(match.score),
                }
            )
        return hits

    def count(self) -> int:
        stats = self._index.describe_index_stats()
        ns_stats = stats.namespaces.get(self._namespace)
        return ns_stats.vector_count if ns_stats else 0

    def reset(self) -> None:
        self._index.delete(delete_all=True, namespace=self._namespace)
        logger.info("pinecone_store_reseteado", namespace=self._namespace)

    def get_stats(self) -> dict:
        stats = self._index.describe_index_stats()
        return {
            "coleccion": self._index_name,
            "namespace": self._namespace,
            "total_chunks": self.count(),
            "directorio": f"pinecone://{self._index_name}",
            "dimension": stats.dimension,
        }

    @staticmethod
    def _convert_filter(where: dict) -> dict:
        """Convert ChromaDB-style where filter to Pinecone filter format."""
        if "$and" in where:
            conditions = where["$and"]
            return {
                k: {"$eq": v} for cond in conditions for k, v in cond.items()
            }
        return {k: {"$eq": v} for k, v in where.items()}
