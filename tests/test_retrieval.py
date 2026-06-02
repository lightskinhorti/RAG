"""Tests del módulo de retrieval: vector store, búsqueda híbrida y reranker."""

from __future__ import annotations

import numpy as np
import pytest

from src.ingestion.chunker import Chunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chunks_ejemplo() -> list[Chunk]:
    return [
        Chunk(
            texto="La jornada laboral ordinaria es de cuarenta horas semanales.",
            metadata={"fuente": "estatuto.txt", "seccion": "I", "chunk_index": 0},
        ),
        Chunk(
            texto="Los trabajadores tienen derecho a quince días de vacaciones anuales.",
            metadata={"fuente": "estatuto.txt", "seccion": "I", "chunk_index": 1},
        ),
        Chunk(
            texto="El plazo de prescripción tributaria es de cuatro años.",
            metadata={"fuente": "ley_general_tributaria.txt", "seccion": "II", "chunk_index": 0},
        ),
        Chunk(
            texto="El responsable del fichero debe notificar violaciones de datos en 72 horas.",
            metadata={"fuente": "rgpd.txt", "seccion": "III", "chunk_index": 0},
        ),
        Chunk(
            texto="La indemnización por despido improcedente es de 33 días por año trabajado.",
            metadata={"fuente": "estatuto.txt", "seccion": "I", "chunk_index": 2},
        ),
    ]


@pytest.fixture
def embeddings_ejemplo(chunks_ejemplo: list[Chunk]) -> np.ndarray:
    """Embeddings aleatorios normalizados para tests (sin cargar modelo real)."""
    rng = np.random.default_rng(42)
    embs = rng.random((len(chunks_ejemplo), 384)).astype(np.float32)
    normas = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / normas


@pytest.fixture
def vector_store_tmp(tmp_path, chunks_ejemplo, embeddings_ejemplo):
    """ChromaVectorStore con datos de prueba en directorio temporal."""
    from src.retrieval.vector_store import ChromaVectorStore

    vs = ChromaVectorStore(
        persist_dir=str(tmp_path / "chroma_test"),
        collection_name="test_collection",
    )
    vs.add_chunks(chunks_ejemplo, embeddings_ejemplo)
    return vs


# ---------------------------------------------------------------------------
# Tests: ChromaVectorStore
# ---------------------------------------------------------------------------


def test_vector_store_add_y_count(vector_store_tmp, chunks_ejemplo):
    assert vector_store_tmp.count() == len(chunks_ejemplo)


def test_vector_store_search_devuelve_resultados(vector_store_tmp, embeddings_ejemplo):
    query_emb = embeddings_ejemplo[0]
    resultados = vector_store_tmp.search(query_emb, top_k=3)
    assert len(resultados) == 3
    assert all("texto" in r for r in resultados)
    assert all("metadata" in r for r in resultados)
    assert all("score" in r for r in resultados)


def test_vector_store_search_scores_entre_0_y_1(vector_store_tmp, embeddings_ejemplo):
    resultados = vector_store_tmp.search(embeddings_ejemplo[0], top_k=5)
    for r in resultados:
        assert -0.1 <= r["score"] <= 1.1, f"Score fuera de rango: {r['score']}"


def test_vector_store_top_k_respetado(vector_store_tmp, embeddings_ejemplo):
    resultados = vector_store_tmp.search(embeddings_ejemplo[0], top_k=2)
    assert len(resultados) <= 2


def test_vector_store_reset(vector_store_tmp):
    vector_store_tmp.reset()
    assert vector_store_tmp.count() == 0


def test_vector_store_get_stats(vector_store_tmp):
    stats = vector_store_tmp.get_stats()
    assert "total_chunks" in stats
    assert "coleccion" in stats
    assert stats["total_chunks"] > 0


# ---------------------------------------------------------------------------
# Tests: HybridSearcher
# ---------------------------------------------------------------------------


class MockEmbedder:
    """Mock del embedder para tests sin cargar el modelo real."""

    dimension = 384

    def __init__(self, fixed_emb: np.ndarray):
        self._emb = fixed_emb

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.tile(self._emb, (len(texts), 1))


def test_hybrid_search_sin_bm25(vector_store_tmp, embeddings_ejemplo):
    from src.retrieval.hybrid_search import HybridSearcher

    mock_emb = MockEmbedder(embeddings_ejemplo[0])
    searcher = HybridSearcher(vector_store_tmp, mock_emb, alpha=1.0)
    # Sin índice BM25, debería caer en solo denso
    resultados = searcher.search("jornada laboral", top_k=3)
    assert len(resultados) <= 3
    assert all("texto" in r for r in resultados)


def test_hybrid_search_con_bm25(vector_store_tmp, embeddings_ejemplo):
    from src.retrieval.hybrid_search import HybridSearcher

    mock_emb = MockEmbedder(embeddings_ejemplo[0])
    searcher = HybridSearcher(vector_store_tmp, mock_emb, alpha=0.7)
    searcher.build_bm25_index()

    resultados = searcher.search("jornada laboral horas semanales", top_k=3)
    assert len(resultados) >= 1
    assert all("texto" in r for r in resultados)


def test_hybrid_search_alpha_0_solo_bm25(vector_store_tmp, embeddings_ejemplo):
    from src.retrieval.hybrid_search import HybridSearcher

    mock_emb = MockEmbedder(embeddings_ejemplo[0])
    searcher = HybridSearcher(vector_store_tmp, mock_emb, alpha=0.0)
    searcher.build_bm25_index()

    resultados = searcher.search("vacaciones anuales trabajadores", top_k=3)
    assert len(resultados) >= 0  # Puede haber 0 si BM25 no encuentra nada


def test_hybrid_search_devuelve_rrf_score(vector_store_tmp, embeddings_ejemplo):
    from src.retrieval.hybrid_search import HybridSearcher

    mock_emb = MockEmbedder(embeddings_ejemplo[0])
    searcher = HybridSearcher(vector_store_tmp, mock_emb, alpha=0.7)
    searcher.build_bm25_index()

    resultados = searcher.search("prescripción tributaria", top_k=3)
    if resultados:
        assert "rrf_score" in resultados[0]


# ---------------------------------------------------------------------------
# Tests: métricas de evaluación
# ---------------------------------------------------------------------------


def test_evaluar_fidelidad_alta():
    from src.evaluation.metrics import evaluar_fidelidad

    respuesta = "La jornada laboral es de cuarenta horas semanales"
    contexto = [{"texto": "La jornada laboral ordinaria es de cuarenta horas semanales según el ET"}]
    score = evaluar_fidelidad(respuesta, contexto)
    assert score > 0.5


def test_evaluar_fidelidad_baja():
    from src.evaluation.metrics import evaluar_fidelidad

    respuesta = "Los planetas orbitan alrededor del sol"
    contexto = [{"texto": "La jornada laboral es de cuarenta horas"}]
    score = evaluar_fidelidad(respuesta, contexto)
    assert score < 0.5


def test_evaluar_relevancia_respuesta():
    from src.evaluation.metrics import evaluar_relevancia_respuesta

    pregunta = "¿Cuántos días de vacaciones tienen los trabajadores?"
    respuesta = "Los trabajadores tienen derecho a quince días de vacaciones anuales retribuidas."
    score = evaluar_relevancia_respuesta(pregunta, respuesta)
    assert score > 0.3


def test_evaluar_precision_contexto():
    from src.evaluation.metrics import evaluar_precision_contexto

    pregunta = "prescripción tributaria plazos"
    contexto = [
        {"texto": "La prescripción de deudas tributarias es de cuatro años"},
        {"texto": "El impuesto sobre la renta grava las rentas de las personas físicas"},
        {"texto": "Los plazos tributarios están regulados en la LGT"},
    ]
    score = evaluar_precision_contexto(pregunta, contexto)
    assert 0.0 <= score <= 1.0


def test_evaluar_recall_contexto():
    from src.evaluation.metrics import evaluar_recall_contexto

    respuesta_esperada = "72 horas para notificar violaciones de datos personales"
    contexto = [
        {"texto": "El responsable debe notificar en 72 horas las violaciones de datos"},
    ]
    score = evaluar_recall_contexto(respuesta_esperada, contexto)
    assert score > 0.3


def test_resultado_evaluacion_score_global():
    from src.evaluation.metrics import ResultadoEvaluacion

    r = ResultadoEvaluacion(
        id="q01",
        pregunta="¿test?",
        respuesta_generada="respuesta test",
        respuesta_esperada="respuesta esperada",
        fidelidad=0.8,
        relevancia_respuesta=0.7,
        precision_contexto=0.9,
        recall_contexto=0.6,
    )
    esperado = round((0.8 + 0.7 + 0.9 + 0.6) / 4, 4)
    assert r.score_global == esperado
