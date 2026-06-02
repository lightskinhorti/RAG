"""Rutas de la API FastAPI para el sistema RAG."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from src.api.models import (
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    FuenteDocumento,
    StatsResponse,
)
from src.config import get_section
from src.embeddings.embedder import get_embedder
from src.evaluation.metrics import EvaluadorRAG
from src.generation.llm import get_generador
from src.ingestion.pipeline import ingest_directory, ingest_document
from src.logger import get_logger
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import get_vector_store

logger = get_logger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Inicialización lazy de componentes pesados
# ---------------------------------------------------------------------------

_embedder = None
_vector_store = None
_searcher = None
_generador = None
_reranker = None


def _get_components():
    global _embedder, _vector_store, _searcher, _generador, _reranker

    if _embedder is None:
        _embedder = get_embedder()
    if _vector_store is None:
        _vector_store = get_vector_store()
    if _searcher is None:
        _searcher = HybridSearcher(_vector_store, _embedder)
        _searcher.build_bm25_index()
    if _generador is None:
        _generador = get_generador()
    if _reranker is None:
        cfg = get_section("retrieval")
        if cfg.get("reranking_habilitado", True):
            try:
                _reranker = CrossEncoderReranker()
            except Exception as e:
                logger.warning("reranker_no_disponible", error=str(e))

    return _embedder, _vector_store, _searcher, _generador, _reranker


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/query", response_model=QueryResponse, summary="Consulta al sistema RAG")
async def query(request: QueryRequest):
    """Endpoint principal: recibe una pregunta y devuelve respuesta con fuentes.

    - Recupera los fragmentos más relevantes del vector store
    - Opcionalmente aplica reranking con cross-encoder
    - Genera respuesta con el LLM citando las fuentes
    """
    inicio = time.time()

    try:
        embedder, vector_store, searcher, generador, reranker = _get_components()
    except Exception as e:
        logger.error("error_inicializando_componentes", error=str(e))
        raise HTTPException(status_code=503, detail=f"Sistema no disponible: {e}") from e

    if vector_store.count() == 0:
        raise HTTPException(
            status_code=404,
            detail="No hay documentos indexados. Ejecuta la ingesta primero.",
        )

    # Búsqueda híbrida
    searcher._alpha = request.alpha
    resultados = searcher.search(request.pregunta, top_k=request.top_k)

    # Reranking opcional
    if request.reranking and reranker and resultados:
        try:
            resultados = reranker.rerank(request.pregunta, resultados, top_k=request.top_k)
        except Exception as e:
            logger.warning("reranking_fallido", error=str(e))

    # Generación
    respuesta_data = generador.generar(request.pregunta, resultados)

    # Construir fuentes para la respuesta
    fuentes = []
    for r in resultados:
        meta = r.get("metadata", {})
        fuentes.append(FuenteDocumento(
            titulo=str(meta.get("titulo", meta.get("fuente", "Documento BOE")))[:200],
            fecha=str(meta.get("fecha", meta.get("fecha_publicacion", "—"))),
            seccion=str(meta.get("seccion", "—")),
            departamento=str(meta.get("departamento", "—")),
            texto_fragmento=r.get("texto", "")[:500],
            score=float(r.get("rrf_score", r.get("score", 0.0))),
            tipo_busqueda=str(meta.get("estrategia", "hybrid")),
        ))

    latencia_total = (time.time() - inicio) * 1000

    return QueryResponse(
        pregunta=request.pregunta,
        respuesta=respuesta_data["respuesta"],
        fuentes=fuentes,
        num_fuentes=len(fuentes),
        latencia_ms=round(latencia_total, 1),
        modo_mock=respuesta_data["modo_mock"],
    )


@router.post(
    "/query/stream",
    summary="Consulta con respuesta en streaming",
    response_class=StreamingResponse,
)
async def query_stream(request: QueryRequest):
    """Endpoint de consulta con streaming SSE (Server-Sent Events)."""
    try:
        embedder, vector_store, searcher, generador, reranker = _get_components()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    if vector_store.count() == 0:
        raise HTTPException(status_code=404, detail="No hay documentos indexados.")

    searcher._alpha = request.alpha
    resultados = searcher.search(request.pregunta, top_k=request.top_k)

    if request.reranking and reranker and resultados:
        try:
            resultados = reranker.rerank(request.pregunta, resultados, top_k=request.top_k)
        except Exception as e:
            logger.warning("reranking_stream_fallido", error=str(e))

    def generar():
        for chunk in generador.generar_streaming(request.pregunta, resultados):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generar(), media_type="text/event-stream")


@router.post("/ingest", response_model=IngestResponse, summary="Ingesta de documentos")
async def ingest(request: IngestRequest):
    """Procesa e indexa documentos desde un directorio local."""
    inicio = time.time()
    directorio = Path(request.directorio)

    if not directorio.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Directorio no encontrado: {request.directorio}",
        )

    try:
        embedder, vector_store, searcher, generador, reranker = _get_components()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    if request.resetear_coleccion:
        vector_store.reset()
        logger.info("coleccion_reseteada")

    config_override = {
        "estrategia_chunking": request.estrategia,
        "chunk_size": request.chunk_size,
        "chunk_overlap": request.chunk_overlap,
        "min_chunk_length": 50,
    }

    chunks = ingest_directory(directorio, config=config_override)
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail=f"No se encontraron documentos válidos en {request.directorio}",
        )

    textos = [c.texto for c in chunks]
    embeddings = embedder.encode(textos)
    vector_store.add_chunks(chunks, embeddings)

    # Reconstruir índice BM25 con nuevos documentos
    global _searcher
    _searcher = None  # Forzar reconstrucción en la próxima consulta

    duracion = time.time() - inicio
    total_docs = len({c.metadata.get("fuente", "") for c in chunks})

    return IngestResponse(
        total_documentos=total_docs,
        total_chunks=len(chunks),
        chunks_por_documento=round(len(chunks) / max(total_docs, 1), 1),
        estrategia_usada=request.estrategia,
        directorio=str(directorio),
        duracion_segundos=round(duracion, 2),
    )


@router.post(
    "/ingest/upload",
    response_model=IngestResponse,
    summary="Subir y procesar un documento",
)
async def ingest_upload(
    archivo: UploadFile = File(..., description="Archivo PDF, TXT o MD a indexar"),
    estrategia: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
):
    """Sube un documento y lo indexa directamente."""
    formatos_permitidos = {".pdf", ".txt", ".md"}
    sufijo = Path(archivo.filename or "").suffix.lower()

    if sufijo not in formatos_permitidos:
        raise HTTPException(
            status_code=415,
            detail=f"Formato no soportado: {sufijo}. Permitidos: {formatos_permitidos}",
        )

    inicio = time.time()
    tmp_path = Path("data/raw") / (archivo.filename or "upload_tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        contenido = await archivo.read()
        tmp_path.write_bytes(contenido)

        embedder, vector_store, _, _, _ = _get_components()
        config_override = {
            "estrategia_chunking": estrategia,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "min_chunk_length": 50,
        }
        chunks = ingest_document(tmp_path, config=config_override)
        if not chunks:
            raise HTTPException(status_code=422, detail="El documento no generó ningún chunk válido.")

        embeddings = embedder.encode([c.texto for c in chunks])
        vector_store.add_chunks(chunks, embeddings)

        global _searcher
        _searcher = None

    except HTTPException:
        raise
    except Exception as e:
        logger.error("error_ingesta_upload", fichero=archivo.filename, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

    return IngestResponse(
        total_documentos=1,
        total_chunks=len(chunks),
        chunks_por_documento=float(len(chunks)),
        estrategia_usada=estrategia,
        directorio=str(tmp_path),
        duracion_segundos=round(time.time() - inicio, 2),
    )


@router.get("/health", response_model=HealthResponse, summary="Estado del servicio")
async def health():
    """Health check con estado de los componentes críticos."""
    import os

    try:
        vs = get_vector_store()
        vs_estado = f"ok ({vs.count()} chunks)"
    except Exception as e:
        vs_estado = f"error: {e}"

    return HealthResponse(
        estado="ok",
        vector_store=vs_estado,
        llm_configurado=bool(os.getenv("ANTHROPIC_API_KEY")),
    )


@router.get("/stats", response_model=StatsResponse, summary="Estadísticas de la colección")
async def stats():
    """Devuelve estadísticas de la colección vectorial y estado de los índices."""
    try:
        vs = get_vector_store()
        vs_stats = vs.get_stats()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error obteniendo estadísticas: {e}") from e

    return StatsResponse(
        total_chunks=vs_stats.get("total_chunks", 0),
        coleccion=vs_stats.get("coleccion", "—"),
        directorio_persistencia=vs_stats.get("directorio", "—"),
        bm25_indexado=_searcher is not None,
        reranker_disponible=_reranker is not None,
    )


@router.get(
    "/evaluation",
    summary="Ejecutar evaluación del sistema RAG",
)
async def run_evaluation():
    """Ejecuta la evaluación completa del sistema RAG con el dataset de referencia."""
    try:
        embedder, vector_store, searcher, generador, reranker = _get_components()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    def pipeline(pregunta: str) -> dict:
        resultados = searcher.search(pregunta, top_k=5)
        if reranker:
            resultados = reranker.rerank(pregunta, resultados, top_k=5)
        gen = generador.generar(pregunta, resultados)
        return {"respuesta": gen["respuesta"], "contexto": resultados}

    evaluador = EvaluadorRAG(rag_pipeline=pipeline)
    informe = evaluador.evaluar_dataset()

    return informe.to_dict()
