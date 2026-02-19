"""Instrumentação de métricas Prometheus para FastAPI."""
import os
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge


# Métricas custom para SSE streams
sse_streams_active = Gauge(
    "sse_streams_active",
    "Número de streams SSE ativos",
    ["service"],
)
sse_stream_duration_seconds = Histogram(
    "sse_stream_duration_seconds",
    "Duração dos streams SSE em segundos",
    ["service"],
    buckets=[1, 5, 10, 30, 60, 120, 300],
)
sse_stream_chunks_total = Counter(
    "sse_stream_chunks_total",
    "Total de chunks enviados via SSE",
    ["service"],
)
sse_client_disconnects_total = Counter(
    "sse_client_disconnects_total",
    "Total de desconexões de clientes SSE",
    ["service"],
)


def setup_metrics(app, service_name: str = "unknown"):
    """Configura métricas Prometheus no app FastAPI.

    Expõe /metrics e instrumenta automaticamente todos os endpoints HTTP.
    Gera métricas: http_requests_total, http_request_duration_seconds,
    http_requests_in_progress.
    """
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        excluded_handlers=["/metrics", "/health", "/", "/api/v1/health"],
        inprogress_name="http_requests_in_progress",
        inprogress_labels=True,
    )
    instrumentator.instrument(app)
    instrumentator.expose(app, include_in_schema=False, tags=["Observability"])
