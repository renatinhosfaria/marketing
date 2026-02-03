# Observabilidade

## Logs

Logging estruturado com `structlog` em `shared/infrastructure/logging/structlog_config.py`.
Nivel controlado por `LOG_LEVEL`.

## Trace

Middleware `TraceMiddleware` injeta `X-Trace-ID` e registra request/response em `shared/infrastructure/tracing/middleware.py`.
