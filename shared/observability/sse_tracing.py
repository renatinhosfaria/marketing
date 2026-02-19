"""Tracing e métricas para Server-Sent Events (SSE).

Cria spans para o lifecycle COMPLETO do stream, NÃO por token/chunk individual.
Isso evita alta cardinalidade. Registra métricas resumidas ao fechar o span.
"""
import time
from contextlib import asynccontextmanager
from opentelemetry import trace
from shared.observability.metrics import (
    sse_streams_active,
    sse_stream_duration_seconds,
    sse_stream_chunks_total,
    sse_client_disconnects_total,
)

tracer = trace.get_tracer("sse.streaming")


class SSEStreamTracker:
    """Acumula métricas do stream sem criar spans adicionais por chunk."""

    __slots__ = ("service_name", "total_chunks", "total_bytes", "client_disconnected")

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.total_chunks = 0
        self.total_bytes = 0
        self.client_disconnected = False

    def record_chunk(self, byte_size: int = 0):
        """Registra envio de um chunk SSE."""
        self.total_chunks += 1
        self.total_bytes += byte_size


@asynccontextmanager
async def trace_sse_stream(service_name: str, thread_id: str = ""):
    """Context manager para tracing de um stream SSE completo.

    Cria um ÚNICO span para todo o stream. Ao finalizar,
    registra métricas resumidas (chunks, bytes, duração).

    Uso:
        async with trace_sse_stream("fb-ads-api", thread_id) as tracker:
            for event in stream:
                data = json.dumps(event)
                tracker.record_chunk(len(data))
                yield f"data: {data}\\n\\n"
    """
    tracker = SSEStreamTracker(service_name)
    sse_streams_active.labels(service=service_name).inc()
    start = time.monotonic()

    with tracer.start_as_current_span(
        "sse.stream",
        attributes={
            "sse.service": service_name,
            "sse.thread_id": thread_id,
        },
    ) as span:
        try:
            yield tracker
        except Exception as e:
            span.set_attribute("sse.error", True)
            span.set_attribute("sse.error_type", type(e).__name__)
            err_msg = str(e).lower()
            if "disconnect" in err_msg or "closed" in err_msg:
                tracker.client_disconnected = True
                sse_client_disconnects_total.labels(service=service_name).inc()
            raise
        finally:
            duration = time.monotonic() - start
            sse_streams_active.labels(service=service_name).dec()
            sse_stream_duration_seconds.labels(service=service_name).observe(duration)
            sse_stream_chunks_total.labels(service=service_name).inc(tracker.total_chunks)

            span.set_attribute("sse.total_chunks", tracker.total_chunks)
            span.set_attribute("sse.total_bytes", tracker.total_bytes)
            span.set_attribute("sse.duration_seconds", round(duration, 3))
            span.set_attribute("sse.client_disconnected", tracker.client_disconnected)
