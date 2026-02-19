"""Métricas Prometheus e tracing OpenTelemetry para Celery workers."""
import time
from prometheus_client import Counter, Histogram, Gauge
from celery.signals import (
    task_prerun, task_postrun, task_failure, task_retry,
    worker_ready, worker_shutdown,
)
from opentelemetry.instrumentation.celery import CeleryInstrumentor


# Métricas Celery
celery_tasks_total = Counter(
    "celery_tasks_total",
    "Total de tasks Celery executadas",
    ["task_name", "queue", "status"],
)
celery_task_duration_seconds = Histogram(
    "celery_task_duration_seconds",
    "Duração de execução de tasks Celery em segundos",
    ["task_name", "queue"],
    buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 120, 300, 600],
)
celery_workers_active = Gauge(
    "celery_workers_active",
    "Número de workers Celery ativos",
)

_task_start_times: dict[str, float] = {}


def setup_celery_observability(celery_app):
    """Configura métricas e tracing para Celery workers.

    Registra signal handlers para capturar lifecycle de tasks.
    Inicializa auto-instrumentação OTel para propagação de contexto.
    """
    CeleryInstrumentor().instrument()

    @task_prerun.connect
    def on_task_prerun(sender=None, task_id=None, **kwargs):
        _task_start_times[task_id] = time.monotonic()

    @task_postrun.connect
    def on_task_postrun(sender=None, task_id=None, **kwargs):
        start = _task_start_times.pop(task_id, None)
        if start is not None:
            duration = time.monotonic() - start
            queue = getattr(sender, "queue", None) or "default"
            task_name = sender.name if sender else "unknown"
            celery_task_duration_seconds.labels(
                task_name=task_name, queue=queue
            ).observe(duration)
            celery_tasks_total.labels(
                task_name=task_name, queue=queue, status="success"
            ).inc()

    @task_failure.connect
    def on_task_failure(sender=None, task_id=None, **kwargs):
        _task_start_times.pop(task_id, None)
        queue = getattr(sender, "queue", None) or "default"
        task_name = sender.name if sender else "unknown"
        celery_tasks_total.labels(
            task_name=task_name, queue=queue, status="failure"
        ).inc()

    @task_retry.connect
    def on_task_retry(sender=None, **kwargs):
        queue = getattr(sender, "queue", None) or "default"
        task_name = sender.name if sender else "unknown"
        celery_tasks_total.labels(
            task_name=task_name, queue=queue, status="retry"
        ).inc()

    @worker_ready.connect
    def on_worker_ready(**kwargs):
        celery_workers_active.inc()

    @worker_shutdown.connect
    def on_worker_shutdown(**kwargs):
        celery_workers_active.dec()
