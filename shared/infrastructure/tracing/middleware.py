"""
Middleware para capturar/gerar trace_id e injetar no contexto.
"""
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from shared.infrastructure.tracing.context import (
    generate_trace_id,
    generate_span_id,
    set_trace_context,
    get_trace_context
)
from shared.infrastructure.tracing.events import (
    log_orchestrator_request_received,
    log_orchestrator_response_sent,
    log_orchestrator_request_failed
)


class TraceMiddleware(BaseHTTPMiddleware):
    """
    Middleware que:
    1. Captura X-Trace-ID do header ou gera novo
    2. Injeta trace_id no contexto
    3. Loga request received e response sent
    """

    EXCLUDED_PATHS = frozenset({"/health", "/api/health", "/api/v1/agent/health"})

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

        # 3. Extrai dados da request
        user_id = getattr(request.state, "user_id", None)
        config_id = getattr(request.state, "config_id", None)

        # 4. Loga request received
        start_time = time.time()
        log_orchestrator_request_received(
            user_id=user_id,
            config_id=config_id,
            path=str(request.url.path),
            method=request.method,
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )

        try:
            # 5. Processa request
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000

            # 6. Loga response sent
            log_orchestrator_response_sent(
                status_code=response.status_code,
                duration_ms=duration_ms
            )

            # 7. Adiciona trace_id no header da resposta
            response.headers["X-Trace-ID"] = trace_id

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            # 8. Loga request failed
            log_orchestrator_request_failed(
                error_type=type(e).__name__,
                error_message=str(e),
                duration_ms=duration_ms
            )
            raise
