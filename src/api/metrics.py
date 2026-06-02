"""Prometheus metrics for the RAG system."""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.requests import Request
from starlette.responses import Response

REGISTRY = CollectorRegistry()

# ---------------------------------------------------------------------------
# Query metrics
# ---------------------------------------------------------------------------

RAG_QUERY_TOTAL = Counter(
    "rag_query_total",
    "Total number of queries processed",
    labelnames=["cache_hit", "reranking"],
    registry=REGISTRY,
)

RAG_QUERY_LATENCY_SECONDS = Histogram(
    "rag_query_latency_seconds",
    "Query latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Ingestion metrics
# ---------------------------------------------------------------------------

RAG_INGESTION_CHUNKS_TOTAL = Counter(
    "rag_ingestion_chunks_total",
    "Total number of chunks ingested",
    registry=REGISTRY,
)

RAG_INGESTION_DURATION_SECONDS = Histogram(
    "rag_ingestion_duration_seconds",
    "Ingestion duration in seconds",
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Gauges
# ---------------------------------------------------------------------------

RAG_CACHE_SIZE = Gauge(
    "rag_cache_size",
    "Current number of entries in the query cache",
    registry=REGISTRY,
)

RAG_COLLECTION_SIZE = Gauge(
    "rag_collection_size",
    "Current number of chunks in the vector store collection",
    registry=REGISTRY,
)

RAG_ACTIVE_REQUESTS = Gauge(
    "rag_active_requests",
    "Number of concurrent requests currently in flight",
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Setup helper
# ---------------------------------------------------------------------------


def setup_metrics(app):  # noqa: ANN001
    """Add a ``/metrics`` endpoint and active-request tracking to *app*."""

    @app.middleware("http")
    async def _track_active_requests(request: Request, call_next):  # noqa: ANN001
        RAG_ACTIVE_REQUESTS.inc()
        try:
            response = await call_next(request)
        finally:
            RAG_ACTIVE_REQUESTS.dec()
        return response

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint():
        return Response(
            content=generate_latest(REGISTRY),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
