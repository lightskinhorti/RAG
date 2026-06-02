"""Tests del módulo de embeddings."""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Tests: SentenceTransformerEmbedder
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedder():
    """Fixture del embedder (carga el modelo una vez por módulo)."""
    from src.embeddings.embedder import SentenceTransformerEmbedder

    # Usamos el modelo más ligero para los tests
    return SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")


def test_encode_devuelve_ndarray(embedder):
    textos = ["Ley de protección de datos", "Real Decreto sobre empleo"]
    embeddings = embedder.encode(textos)
    assert isinstance(embeddings, np.ndarray)


def test_encode_dimensiones_correctas(embedder):
    textos = ["texto uno", "texto dos", "texto tres"]
    embeddings = embedder.encode(textos)
    assert embeddings.shape[0] == len(textos)
    assert embeddings.shape[1] == embedder.dimension


def test_encode_lista_vacia(embedder):
    embeddings = embedder.encode([])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 0


def test_encode_texto_unico(embedder):
    embeddings = embedder.encode(["Artículo único sobre derechos laborales"])
    assert embeddings.shape == (1, embedder.dimension)


def test_embeddings_normalizados(embedder):
    textos = ["Ley Orgánica de Educación"]
    embeddings = embedder.encode(textos)
    norma = np.linalg.norm(embeddings[0])
    assert abs(norma - 1.0) < 1e-5, "Los embeddings deberían estar normalizados"


def test_similitud_semantica(embedder):
    """Textos similares deberían tener mayor similitud que textos distintos."""
    textos = [
        "jornada laboral máxima en España",
        "horas de trabajo permitidas por semana",
        "regulación del tráfico de vehículos",
    ]
    embs = embedder.encode(textos)

    # Similitud coseno entre los dos primeros (similares)
    sim_similar = float(np.dot(embs[0], embs[1]))
    # Similitud con el tercero (diferente)
    sim_distinto = float(np.dot(embs[0], embs[2]))

    assert sim_similar > sim_distinto, (
        f"Textos similares deberían tener mayor similitud: {sim_similar:.3f} vs {sim_distinto:.3f}"
    )


def test_dimension_property(embedder):
    assert isinstance(embedder.dimension, int)
    assert embedder.dimension > 0


def test_get_embedder_devuelve_instancia():
    from src.embeddings.embedder import get_embedder

    emb = get_embedder()
    assert emb is not None
    assert hasattr(emb, "encode")
    assert hasattr(emb, "dimension")
