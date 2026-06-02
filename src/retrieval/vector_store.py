"""Interfaz abstracta y implementacion ChromaDB para el vector store."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import chromadb
import numpy as np

from src.config import get_section
from src.ingestion.chunker import Chunk
from src.logger import get_logger

logger = get_logger(__name__)


class VectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def reset(self) -> None: ...


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ):
        cfg = get_section("vector_store")
        self._persist_dir = persist_dir or cfg["directorio_persistencia"]
        self._collection_name = collection_name or cfg["nombre_coleccion"]

        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": cfg.get("metrica_distancia", "cosine")},
        )
        logger.info(
            "vector_store_inicializado",
            tipo="chromadb",
            coleccion=self._collection_name,
            documentos_existentes=self._collection.count(),
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [c.texto for c in chunks]
        metadatas = [c.metadata for c in chunks]
        embeddings_list = embeddings.tolist()

        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            self._collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                embeddings=embeddings_list[i:end],
            )

        logger.info("chunks_indexados", total=len(chunks))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Búsqueda por similitud con filtrado opcional por metadata.

        Args:
            query_embedding: Vector de la consulta.
            top_k: Número máximo de resultados.
            where: Filtro de metadata ChromaDB (ej: {"seccion": "I"}).
        """
        kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "texto": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distancia": results["distances"][0][i],
                "score": 1 - results["distances"][0][i],
            })
        return hits

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("vector_store_reseteado")

    def get_stats(self) -> dict:
        return {
            "coleccion": self._collection_name,
            "total_chunks": self._collection.count(),
            "directorio": self._persist_dir,
        }


def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore()
