"""Módulo de observabilidade: métricas Prometheus e tracing OpenTelemetry."""
from shared.observability.metrics import setup_metrics
from shared.observability.tracing import setup_tracing, instrument_fastapi, instrument_sqlalchemy
from shared.observability.celery_metrics import setup_celery_observability

__all__ = [
    "setup_metrics",
    "setup_tracing",
    "instrument_fastapi",
    "instrument_sqlalchemy",
    "setup_celery_observability",
]
