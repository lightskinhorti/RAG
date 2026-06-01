"""Pipeline de ingesta: carga documentos, aplica chunking, devuelve chunks listos."""

from __future__ import annotations

from pathlib import Path

from src.config import get_section
from src.ingestion.chunker import Chunk, chunk_document
from src.ingestion.loader import Document, load_directory, load_document
from src.logger import get_logger

logger = get_logger(__name__)


def ingest_document(
    path: Path,
    config: dict | None = None,
) -> list[Chunk]:
    cfg = config or get_section("ingestion")
    doc = load_document(path)
    return _chunk_doc(doc, cfg)


def ingest_directory(
    directory: Path,
    config: dict | None = None,
) -> list[Chunk]:
    cfg = config or get_section("ingestion")
    docs = load_directory(directory, cfg.get("formatos_soportados"))

    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(_chunk_doc(doc, cfg))

    logger.info(
        "ingesta_completada",
        total_documentos=len(docs),
        total_chunks=len(all_chunks),
        avg_chunk_size=round(
            sum(len(c.texto) for c in all_chunks) / max(len(all_chunks), 1)
        ),
    )
    return all_chunks


def _chunk_doc(doc: Document, cfg: dict) -> list[Chunk]:
    return chunk_document(
        text=doc.contenido,
        strategy=cfg.get("estrategia_chunking", "recursive"),
        chunk_size=cfg.get("chunk_size", 512),
        chunk_overlap=cfg.get("chunk_overlap", 64),
        min_chunk_length=cfg.get("min_chunk_length", 50),
        metadata=doc.metadata,
    )
