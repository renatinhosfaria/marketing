"""Configuração de tracing distribuído com OpenTelemetry."""
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator


def setup_tracing(
    service_name: str,
    service_version: str = "1.0.0",
    otlp_endpoint: str | None = None,
):
    """Inicializa OpenTelemetry tracing com OTLP gRPC exporter.

    Args:
        service_name: Nome do serviço (ex: ml-api, fb-ads-api)
        service_version: Versão do serviço
        otlp_endpoint: Endpoint do OTel Collector (default: env OTEL_EXPORTER_OTLP_ENDPOINT)
    """
    endpoint = otlp_endpoint or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317"
    )

    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        "deployment.environment": os.getenv("ENVIRONMENT", "production"),
    })

    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Propagação W3C TraceContext + Baggage
    set_global_textmap(CompositePropagator([
        TraceContextTextMapPropagator(),
        W3CBaggagePropagator(),
    ]))

    # Auto-instrumentações globais (sem referência ao app específico)
    RedisInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()

    return provider


def instrument_fastapi(app):
    """Instrumenta um app FastAPI com OpenTelemetry ASGI."""
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="metrics,health",
    )


def instrument_sqlalchemy(engine):
    """Instrumenta engine SQLAlchemy com OpenTelemetry.

    Args:
        engine: AsyncEngine do SQLAlchemy. Usa .sync_engine internamente.
    """
    SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)
