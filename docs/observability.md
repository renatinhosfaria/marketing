# Observability

## Logs

Padrao de logs estruturados com `structlog`.

Boas praticas:

- incluir contexto de request e service;
- evitar dados sensiveis em payload de log;
- manter nivel `INFO` em producao e `DEBUG` apenas quando necessario.

## Metricas

Metricas expostas e coletadas por Prometheus para:

- saude de endpoints;
- latencia de requisicoes;
- throughput de workers/fila;
- disponibilidade de servicos criticos.

## Trace

Tracing distribuido via OpenTelemetry com exportacao OTLP.

Componentes:

- instrumentacao FastAPI;
- instrumentacao SQLAlchemy, Redis, Celery e HTTPX;
- backend de traces em Tempo;
- visualizacao consolidada em Grafana.
