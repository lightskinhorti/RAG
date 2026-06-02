"""Aplicación FastAPI principal del sistema RAG."""

from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.config import get_section
from src.logger import get_logger, setup_logging

cfg_api = get_section("api")
setup_logging(cfg_api.get("nivel_log", "info"))
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Document Intelligence",
    description=(
        "Sistema de Recuperación Aumentada por Generación sobre documentos legislativos "
        "del BOE (Boletín Oficial del Estado). Búsqueda híbrida densa+sparse con reranking."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS: permite que la UI Streamlit se comunique con la API
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg_api.get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Middleware de métricas de latencia
# ---------------------------------------------------------------------------


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    inicio = time.time()
    response = await call_next(request)
    duracion_ms = round((time.time() - inicio) * 1000, 1)
    response.headers["X-Process-Time-Ms"] = str(duracion_ms)
    logger.debug(
        "request_completado",
        metodo=request.method,
        ruta=request.url.path,
        status=response.status_code,
        latencia_ms=duracion_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Manejadores de errores globales
# ---------------------------------------------------------------------------


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.warning("value_error", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=422,
        content={"error": "Parámetro inválido", "detalle": str(exc), "codigo": 422},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(
        "error_inesperado",
        path=request.url.path,
        tipo=type(exc).__name__,
        error=str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Error interno del servidor", "detalle": str(exc), "codigo": 500},
    )


# ---------------------------------------------------------------------------
# Inclusión de rutas
# ---------------------------------------------------------------------------

app.include_router(router, prefix="", tags=["RAG"])


@app.on_event("startup")
async def startup_event():
    logger.info(
        "servidor_iniciado",
        host=cfg_api.get("host", "0.0.0.0"),
        puerto=cfg_api.get("puerto", 8000),
        version="1.0.0",
    )


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("servidor_detenido")
