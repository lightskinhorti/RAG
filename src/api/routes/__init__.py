"""Rutas de la API FastAPI para el sistema RAG."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.api.metrics import (
    RAG_CACHE_SIZE,
    RAG_COLLECTION_SIZE,
    RAG_INGESTION_CHUNKS_TOTAL,
    RAG_INGESTION_DURATION_SECONDS,
    RAG_QUERY_LATENCY_SECONDS,
    RAG_QUERY_TOTAL,
)
from src.api.models import (
    FuenteDocumento,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)
from src.cache.redis_cache import SemanticCache
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
# Caché semántico (Redis + fallback in-memory)
# ---------------------------------------------------------------------------

_semantic_cache: SemanticCache | None = None


# ---------------------------------------------------------------------------
# Inicialización lazy de componentes pesados
# ---------------------------------------------------------------------------

_embedder = None
_vector_store = None
_searcher = None
_generador = None
_reranker = None


def _get_semantic_cache() -> SemanticCache:
    """Return the shared :class:`SemanticCache` instance (lazy init)."""
    global _semantic_cache
    if _semantic_cache is None:
        # Read optional cache config from default.yaml
        try:
            cache_cfg = get_section("cache")
        except KeyError:
            cache_cfg = {}

        # The embedder may already be initialised by _get_components; if not
        # we fetch it here so the cache can do semantic lookups.
        embedder = _embedder or get_embedder()

        _semantic_cache = SemanticCache(
            embedder=embedder,
            redis_url=cache_cfg.get("redis_url") or None,
            ttl=int(cache_cfg.get("ttl", 3600)),
            max_memory_entries=int(cache_cfg.get("max_entries", 128)),
            similarity_threshold=float(cache_cfg.get("similarity_threshold", 0.95)),
        )
    return _semantic_cache


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

    # Ensure the semantic cache is ready (uses the embedder)
    _get_semantic_cache()

    return _embedder, _vector_store, _searcher, _generador, _reranker


def _build_fuentes(resultados: list[dict]) -> list[FuenteDocumento]:
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
    return fuentes


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/query", response_model=QueryResponse, summary="Consulta al sistema RAG")
async def query(request: QueryRequest):
    """Endpoint principal: recibe una pregunta y devuelve respuesta con fuentes.

    - Recupera los fragmentos más relevantes del vector store
    - Opcionalmente aplica reranking con cross-encoder
    - Genera respuesta con el LLM citando las fuentes
    - Usa caché semántico (Redis + in-memory) para consultas repetidas
    """
    inicio = time.perf_counter()

    # Comprobar caché (exact + semantic)
    cache = _get_semantic_cache()
    ckey = SemanticCache.cache_key(request.pregunta, request.top_k, request.alpha, request.reranking)
    cached = cache.get(ckey, query_text=request.pregunta)
    if cached:
        latencia = (time.perf_counter() - inicio) * 1000
        latencia_s = latencia / 1000
        RAG_QUERY_TOTAL.labels(cache_hit="true", reranking=str(request.reranking).lower()).inc()
        RAG_QUERY_LATENCY_SECONDS.observe(latencia_s)
        RAG_CACHE_SIZE.set(cache.size)
        logger.info("query_cache_hit", latencia_ms=round(latencia, 1))
        return QueryResponse(**{**cached, "latencia_ms": round(latencia, 1)})

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

    # Construir filtro de metadata si se proporcionaron parámetros
    where_filter = None
    conditions = {}
    if request.filtro_seccion:
        conditions["seccion"] = request.filtro_seccion
    if request.filtro_departamento:
        conditions["departamento"] = request.filtro_departamento
    if conditions:
        where_filter = conditions if len(conditions) == 1 else {"$and": [
            {k: v} for k, v in conditions.items()
        ]}

    # Búsqueda híbrida — operaciones CPU-bound delegadas al thread pool
    searcher._alpha = request.alpha
    resultados = await asyncio.to_thread(
        searcher.search, request.pregunta, request.top_k, where_filter
    )

    # Reranking opcional
    if request.reranking and reranker and resultados:
        try:
            resultados = await asyncio.to_thread(
                reranker.rerank, request.pregunta, resultados, request.top_k
            )
        except Exception as e:
            logger.warning("reranking_fallido", error=str(e))

    # Generación — CPU/IO-bound
    respuesta_data = await asyncio.to_thread(
        generador.generar, request.pregunta, resultados
    )

    fuentes = _build_fuentes(resultados)
    latencia_total = (time.perf_counter() - inicio) * 1000

    response_dict = {
        "pregunta": request.pregunta,
        "respuesta": respuesta_data["respuesta"],
        "fuentes": fuentes,
        "num_fuentes": len(fuentes),
        "latencia_ms": round(latencia_total, 1),
        "modo_mock": respuesta_data["modo_mock"],
    }

    # Guardar en caché
    cache.put(ckey, response_dict, query_text=request.pregunta)

    # Record Prometheus metrics
    latencia_s = latencia_total / 1000
    RAG_QUERY_TOTAL.labels(cache_hit="false", reranking=str(request.reranking).lower()).inc()
    RAG_QUERY_LATENCY_SECONDS.observe(latencia_s)
    RAG_CACHE_SIZE.set(cache.size)

    return QueryResponse(**response_dict)


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
    resultados = await asyncio.to_thread(
        searcher.search, request.pregunta, request.top_k
    )

    if request.reranking and reranker and resultados:
        try:
            resultados = await asyncio.to_thread(
                reranker.rerank, request.pregunta, resultados, request.top_k
            )
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
    inicio = time.perf_counter()
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

    chunks = await asyncio.to_thread(ingest_directory, directorio, config_override)
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail=f"No se encontraron documentos válidos en {request.directorio}",
        )

    textos = [c.texto for c in chunks]
    embeddings = await asyncio.to_thread(embedder.encode, textos)
    vector_store.add_chunks(chunks, embeddings)

    # Invalidar caché y forzar reconstrucción del índice BM25
    global _searcher
    _searcher = None
    _get_semantic_cache().invalidate()

    duracion = time.perf_counter() - inicio
    total_docs = len({c.metadata.get("fuente", "") for c in chunks})

    # Record Prometheus metrics
    RAG_INGESTION_CHUNKS_TOTAL.inc(len(chunks))
    RAG_INGESTION_DURATION_SECONDS.observe(duracion)

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

    inicio = time.perf_counter()
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

        embeddings = await asyncio.to_thread(embedder.encode, [c.texto for c in chunks])
        vector_store.add_chunks(chunks, embeddings)

        global _searcher
        _searcher = None
        _get_semantic_cache().invalidate()

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
        duracion_segundos=round(time.perf_counter() - inicio, 2),
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

    # Update Prometheus gauge
    RAG_COLLECTION_SIZE.set(vs_stats.get("total_chunks", 0))

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
async def run_evaluation(use_llm: bool = False):
    """Ejecuta la evaluación completa del sistema RAG.

    - `use_llm=false` (por defecto): métricas léxicas rápidas, sin consumo de API.
    - `use_llm=true`: métricas LLM-judge estilo RAGAS (requiere ANTHROPIC_API_KEY).
    """
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

    evaluador = EvaluadorRAG(rag_pipeline=pipeline, use_llm_judge=use_llm)
    informe = await asyncio.to_thread(evaluador.evaluar_dataset)

    return informe.to_dict()


@router.post(
    "/agent/query",
    summary="Agente de investigación legal multi-paso (LangGraph)",
)
async def agent_query(request: QueryRequest):
    """Agente legal que descompone preguntas complejas en sub-consultas,
    ejecuta múltiples búsquedas y sintetiza una respuesta unificada con
    referencias cruzadas entre documentos."""
    try:
        embedder, vector_store, searcher, generador, reranker = _get_components()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    if vector_store.count() == 0:
        raise HTTPException(status_code=404, detail="No hay documentos indexados.")

    from src.agents.legal_agent import AgentConfig, LegalResearchAgent

    config = AgentConfig(
        top_k_per_query=request.top_k,
        reranking=request.reranking,
        alpha=request.alpha,
    )
    agent = LegalResearchAgent(searcher, reranker, generador, config)
    resultado = await asyncio.to_thread(agent.run, request.pregunta)

    fuentes = _build_fuentes(resultado.get("fuentes", []))
    return {
        "pregunta": resultado["pregunta"],
        "respuesta": resultado["respuesta"],
        "fuentes": [f.model_dump() for f in fuentes],
        "sub_preguntas": resultado.get("sub_preguntas", []),
        "pasos": resultado.get("pasos", []),
        "es_compleja": resultado.get("es_compleja", False),
        "num_fuentes": len(fuentes),
        "latencia_ms": resultado.get("latencia_ms", 0),
    }
