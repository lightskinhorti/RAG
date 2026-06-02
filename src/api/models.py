"""Modelos Pydantic para la API FastAPI del sistema RAG."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Modelos de entrada
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Solicitud de consulta al sistema RAG."""

    pregunta: str = Field(..., min_length=3, max_length=1000, description="Pregunta del usuario")
    top_k: int = Field(default=5, ge=1, le=20, description="Número de fragmentos a recuperar")
    alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Peso de búsqueda densa vs BM25 (1.0=solo denso, 0.0=solo BM25)",
    )
    reranking: bool = Field(default=True, description="Activar reranking con cross-encoder")
    streaming: bool = Field(default=False, description="Activar respuesta en streaming")
    filtro_seccion: str | None = Field(
        default=None,
        description="Filtrar por sección del BOE (ej: 'I. Disposiciones generales')",
    )
    filtro_departamento: str | None = Field(
        default=None,
        description="Filtrar por departamento (ej: 'Ministerio de Trabajo')",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "pregunta": "¿Cuál es la jornada laboral máxima en España?",
            "top_k": 5,
            "alpha": 0.7,
            "reranking": True,
            "streaming": False,
            "filtro_seccion": None,
            "filtro_departamento": None,
        }
    }}


class IngestRequest(BaseModel):
    """Solicitud de ingesta de documentos desde un directorio."""

    directorio: str = Field(
        default="data/raw",
        description="Ruta al directorio con documentos a ingestar",
    )
    estrategia: str = Field(
        default="recursive",
        pattern="^(fixed|recursive|semantic)$",
        description="Estrategia de chunking: fixed, recursive o semantic",
    )
    chunk_size: int = Field(default=512, ge=100, le=2000)
    chunk_overlap: int = Field(default=64, ge=0, le=200)
    resetear_coleccion: bool = Field(
        default=False,
        description="Si True, elimina los documentos existentes antes de ingestar",
    )


# ---------------------------------------------------------------------------
# Modelos de salida
# ---------------------------------------------------------------------------


class FuenteDocumento(BaseModel):
    """Fragmento de documento fuente recuperado."""

    titulo: str
    fecha: str
    seccion: str
    departamento: str
    texto_fragmento: str
    score: float
    tipo_busqueda: str


class QueryResponse(BaseModel):
    """Respuesta del sistema RAG a una consulta."""

    pregunta: str
    respuesta: str
    fuentes: list[FuenteDocumento]
    num_fuentes: int
    latencia_ms: float
    modo_mock: bool = False

    model_config = {"json_schema_extra": {
        "example": {
            "pregunta": "¿Cuál es la jornada laboral máxima?",
            "respuesta": "La jornada máxima es de 40 horas semanales [Fuente 1].",
            "fuentes": [],
            "num_fuentes": 3,
            "latencia_ms": 1250.5,
            "modo_mock": False,
        }
    }}


class IngestResponse(BaseModel):
    """Respuesta tras la ingesta de documentos."""

    total_documentos: int
    total_chunks: int
    chunks_por_documento: float
    estrategia_usada: str
    directorio: str
    duracion_segundos: float


class StatsResponse(BaseModel):
    """Estadísticas actuales de la colección vectorial."""

    total_chunks: int
    coleccion: str
    directorio_persistencia: str
    bm25_indexado: bool
    reranker_disponible: bool


class HealthResponse(BaseModel):
    """Estado de salud del servicio."""

    estado: str
    vector_store: str
    llm_configurado: bool
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Respuesta de error estándar."""

    error: str
    detalle: str
    codigo: int
