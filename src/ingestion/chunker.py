"""Estrategias de chunking: fixed, semantic, recursive."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    texto: str
    metadata: dict = field(default_factory=dict)


def fixed_chunking(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    metadata: dict | None = None,
) -> list[Chunk]:
    meta = metadata or {}
    overlap = min(chunk_overlap, chunk_size - 1) if chunk_size > 1 else 0
    step = max(chunk_size - overlap, 1)
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        if chunk_text.strip():
            chunks.append(Chunk(
                texto=chunk_text.strip(),
                metadata={**meta, "chunk_index": idx, "estrategia": "fixed"},
            ))
            idx += 1
        start += step
    return chunks


def semantic_chunking(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    metadata: dict | None = None,
) -> list[Chunk]:
    meta = metadata or {}
    sentences = []
    for sep in [".\n", ". ", "\n\n", "\n"]:
        if sep in text:
            sentences = [s.strip() for s in text.split(sep) if s.strip()]
            break
    if not sentences:
        sentences = [text]

    chunks = []
    current = ""
    idx = 0
    for sentence in sentences:
        if len(current) + len(sentence) + 1 > chunk_size and current:
            chunks.append(Chunk(
                texto=current.strip(),
                metadata={**meta, "chunk_index": idx, "estrategia": "semantic"},
            ))
            idx += 1
            overlap_text = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            current = overlap_text + " " + sentence
        else:
            current = current + " " + sentence if current else sentence

    if current.strip():
        chunks.append(Chunk(
            texto=current.strip(),
            metadata={**meta, "chunk_index": idx, "estrategia": "semantic"},
        ))

    return chunks


def recursive_chunking(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    metadata: dict | None = None,
) -> list[Chunk]:
    meta = metadata or {}
    separators = ["\n\n", "\n", ". ", " "]

    def _split(txt: str, seps: list[str]) -> list[str]:
        if not seps or len(txt) <= chunk_size:
            return [txt] if txt.strip() else []

        sep = seps[0]
        parts = txt.split(sep)
        result = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.extend(_split(current, seps[1:]) if len(current) > chunk_size else [current])
                current = part
        if current:
            result.extend(_split(current, seps[1:]) if len(current) > chunk_size else [current])
        return result

    raw_chunks = _split(text, separators)

    chunks = []
    for idx, chunk_text in enumerate(raw_chunks):
        if chunk_text.strip():
            chunks.append(Chunk(
                texto=chunk_text.strip(),
                metadata={**meta, "chunk_index": idx, "estrategia": "recursive"},
            ))

    return chunks


_STRATEGIES = {
    "fixed": fixed_chunking,
    "semantic": semantic_chunking,
    "recursive": recursive_chunking,
}


def chunk_document(
    text: str,
    strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    min_chunk_length: int = 50,
    metadata: dict | None = None,
) -> list[Chunk]:
    fn = _STRATEGIES.get(strategy)
    if fn is None:
        raise ValueError(f"Estrategia desconocida: {strategy}. Opciones: {list(_STRATEGIES)}")

    chunks = fn(text, chunk_size, chunk_overlap, metadata)
    chunks = [c for c in chunks if len(c.texto) >= min_chunk_length]

    logger.info(
        "chunking_completado",
        estrategia=strategy,
        num_chunks=len(chunks),
        avg_length=round(sum(len(c.texto) for c in chunks) / max(len(chunks), 1)),
    )
    return chunks
