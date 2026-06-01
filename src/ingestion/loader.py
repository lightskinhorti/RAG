"""Carga de documentos desde diferentes formatos (PDF, TXT, Markdown)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    contenido: str
    metadata: dict = field(default_factory=dict)


def load_txt(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    return Document(
        contenido=text,
        metadata={"fuente": path.name, "formato": "txt", "ruta": str(path)},
    )


def load_markdown(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    return Document(
        contenido=text,
        metadata={"fuente": path.name, "formato": "markdown", "ruta": str(path)},
    )


def load_pdf(path: Path) -> Document:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)

    return Document(
        contenido="\n\n".join(pages),
        metadata={
            "fuente": path.name,
            "formato": "pdf",
            "ruta": str(path),
            "num_paginas": len(reader.pages),
        },
    )


_LOADERS = {
    ".txt": load_txt,
    ".md": load_markdown,
    ".pdf": load_pdf,
}


def load_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"Formato no soportado: {suffix}")
    logger.info("cargando_documento", fichero=path.name, formato=suffix)
    return loader(path)


def load_directory(directory: Path, extensions: list[str] | None = None) -> list[Document]:
    exts = set(extensions or _LOADERS.keys())
    docs = []
    for file_path in sorted(directory.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in exts:
            try:
                docs.append(load_document(file_path))
            except Exception as e:
                logger.error("error_cargando", fichero=file_path.name, error=str(e))
    logger.info("directorio_cargado", total_documentos=len(docs))
    return docs
