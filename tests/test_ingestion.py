"""Tests del pipeline de ingesta: carga de documentos y chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.chunker import Chunk, chunk_document
from src.ingestion.loader import Document, load_document

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def texto_corto() -> str:
    return "Esta es una ley española. Regula los derechos fundamentales. Tiene varios artículos."


@pytest.fixture
def texto_largo() -> str:
    parrafo = "Artículo de legislación española con contenido normativo relevante. " * 20
    return "\n\n".join([parrafo] * 5)


@pytest.fixture
def fichero_txt(tmp_path: Path) -> Path:
    f = tmp_path / "ley_test.txt"
    f.write_text("Contenido de prueba para el test de carga de documentos TXT.", encoding="utf-8")
    return f


@pytest.fixture
def fichero_md(tmp_path: Path) -> Path:
    f = tmp_path / "doc_test.md"
    f.write_text("# Título\n\nContenido en Markdown para el test de carga.", encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# Tests: loader
# ---------------------------------------------------------------------------


def test_load_txt_devuelve_documento(fichero_txt: Path):
    doc = load_document(fichero_txt)
    assert isinstance(doc, Document)
    assert "Contenido de prueba" in doc.contenido
    assert doc.metadata["formato"] == "txt"
    assert doc.metadata["fuente"] == fichero_txt.name


def test_load_markdown_devuelve_documento(fichero_md: Path):
    doc = load_document(fichero_md)
    assert isinstance(doc, Document)
    assert "Markdown" in doc.contenido
    assert doc.metadata["formato"] == "markdown"


def test_load_formato_no_soportado_lanza_excepcion(tmp_path: Path):
    f = tmp_path / "fichero.xyz"
    f.write_text("contenido")
    with pytest.raises(ValueError, match="Formato no soportado"):
        load_document(f)


def test_load_directorio_carga_multiples(tmp_path: Path):
    from src.ingestion.loader import load_directory

    for i in range(3):
        (tmp_path / f"doc_{i}.txt").write_text(f"Documento {i} con contenido legislativo.")

    docs = load_directory(tmp_path, extensions=[".txt"])
    assert len(docs) == 3
    assert all(isinstance(d, Document) for d in docs)


# ---------------------------------------------------------------------------
# Tests: chunker — estrategia fixed
# ---------------------------------------------------------------------------


def test_fixed_chunking_genera_chunks(texto_largo: str):
    chunks = chunk_document(texto_largo, strategy="fixed", chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_fixed_chunking_respeta_chunk_size(texto_largo: str):
    chunk_size = 150
    chunks = chunk_document(texto_largo, strategy="fixed", chunk_size=chunk_size, chunk_overlap=0)
    # Los chunks no deberían exceder el tamaño configurado
    assert all(len(c.texto) <= chunk_size + 10 for c in chunks)


def test_fixed_chunking_preserva_metadata(texto_corto: str):
    meta = {"fuente": "test.txt", "seccion": "I"}
    chunks = chunk_document(texto_corto, strategy="fixed", chunk_size=50, metadata=meta)
    assert all(c.metadata.get("fuente") == "test.txt" for c in chunks)
    assert all(c.metadata.get("estrategia") == "fixed" for c in chunks)
    assert all("chunk_index" in c.metadata for c in chunks)


# ---------------------------------------------------------------------------
# Tests: chunker — estrategia recursive
# ---------------------------------------------------------------------------


def test_recursive_chunking_genera_chunks(texto_largo: str):
    chunks = chunk_document(texto_largo, strategy="recursive", chunk_size=300, chunk_overlap=30)
    assert len(chunks) > 0


def test_recursive_chunking_metadata_correcta(texto_corto: str):
    chunks = chunk_document(texto_corto, strategy="recursive", metadata={"fuente": "ley.txt"})
    assert all(c.metadata.get("estrategia") == "recursive" for c in chunks)


# ---------------------------------------------------------------------------
# Tests: chunker — estrategia semantic
# ---------------------------------------------------------------------------


def test_semantic_chunking_genera_chunks(texto_largo: str):
    chunks = chunk_document(texto_largo, strategy="semantic", chunk_size=300)
    assert len(chunks) > 0


def test_semantic_chunking_metadato_estrategia(texto_largo: str):
    chunks = chunk_document(texto_largo, strategy="semantic")
    assert all(c.metadata.get("estrategia") == "semantic" for c in chunks)


# ---------------------------------------------------------------------------
# Tests: chunker — casos borde
# ---------------------------------------------------------------------------


def test_chunk_document_estrategia_invalida():
    with pytest.raises(ValueError, match="Estrategia desconocida"):
        chunk_document("texto de prueba", strategy="magica")


def test_chunk_document_filtra_chunks_cortos():
    texto = "a b c"  # Muy corto, debería filtrarse
    chunks = chunk_document(texto, strategy="fixed", min_chunk_length=20)
    assert len(chunks) == 0


def test_chunk_document_texto_vacio():
    chunks = chunk_document("", strategy="recursive")
    assert len(chunks) == 0


# ---------------------------------------------------------------------------
# Tests: pipeline de ingesta
# ---------------------------------------------------------------------------


def test_ingest_document_genera_chunks(tmp_path: Path):
    from src.ingestion.pipeline import ingest_document

    f = tmp_path / "decreto.txt"
    f.write_text(
        "Real Decreto sobre regulación de servicios digitales. " * 30,
        encoding="utf-8",
    )
    chunks = ingest_document(f)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_ingest_directory_carga_multiples(tmp_path: Path):
    from src.ingestion.pipeline import ingest_directory

    for i in range(3):
        (tmp_path / f"ley_{i}.txt").write_text(
            f"Ley {i}: contenido normativo español relevante para el test. " * 20
        )

    chunks = ingest_directory(tmp_path)
    assert len(chunks) > 0
