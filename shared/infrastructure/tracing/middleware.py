"""
Middleware para capturar/gerar trace_id e injetar no contexto.
"""
import time
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from shared.infrastructure.tracing.context import (
    generate_trace_id,
    generate_span_id,
    set_trace_context,
    get_trace_context
)

logger = structlog.get_logger("tracing.middleware")


class TraceMiddleware(BaseHTTPMiddleware):
    """
    Middleware que:
    1. Captura X-Trace-ID do header ou gera novo
    2. Injeta trace_id no contexto
    3. Loga request received e response sent
    """

    EXCLUDED_PATHS = frozenset({"/health", "/api/health"})

    async def dispatch(self, request: Request, call_next):
        # Skip tracing para health checks (evita poluir logs)
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)

        # 1. Captura ou gera trace_id
        trace_id = request.headers.get("X-Trace-ID") or generate_trace_id()
        root_span_id = generate_span_id()

        # 2. Injeta no contexto
        set_trace_context(
            trace_id=trace_id,
            span_id=root_span_id,
            parent_span_id=None
        )

        # 3. Loga request received
        start_time = time.time()
        logger.info(
            "request_received",
            **get_trace_context(),
            path=str(request.url.path),
            method=request.method,
            ip=request.client.host if request.client else None,
        )

        try:
            # 4. Processa request
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000

            # 5. Loga response sent
            logger.info(
                "response_sent",
                **get_trace_context(),
                status_code=response.status_code,
                duration_ms=duration_ms,
            )

            # 6. Adiciona trace_id no header da resposta
            response.headers["X-Trace-ID"] = trace_id

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            # 7. Loga request failed
            logger.error(
                "request_failed",
                **get_trace_context(),
                error_type=type(e).__name__,
                error_message=str(e),
                duration_ms=duration_ms,
            )
            raise
