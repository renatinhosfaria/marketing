# Arquitetura

## Componentes

O sistema e dividido em blocos claros:

- `marketing-frontend` (Next.js): UI e roteamento de chamadas para APIs internas.
- `marketing-api` (FastAPI): dominio ML com endpoints de predicao, forecast, recomendacao e anomalias.
- `marketing-fb-ads` (FastAPI): dominio Facebook Ads para OAuth, sync e insights.
- `marketing-worker` e `marketing-fb-ads-worker` (Celery): processamento assincrono por fila.
- `marketing-beat` (Celery Beat): agenda jobs recorrentes.
- `marketing-redis`: broker Celery e suporte a cache/rate-limit.
- PostgreSQL externo: armazenamento transacional e historico analitico.

## Fluxo de dados

1. O frontend envia requisicoes para `/api/*`.
2. Rewrites do Next.js direcionam chamadas para a API correta (ML ou Facebook Ads).
3. APIs validam autenticacao, acessam banco e podem publicar tarefas Celery.
4. Workers processam tarefas de treino, sync e consolidacao.
5. Resultados persistem em PostgreSQL e ficam disponiveis para consulta via API.
6. Instrumentacao de metricas e traces alimenta stack de observabilidade.

## Portas

| Componente | Porta interna | Porta externa |
| --- | --- | --- |
| Frontend | 3001 | 8000 |
| ML API | 8000 | 8001 |
| Facebook Ads API | 8002 | 8003 |
| Redis | 6379 | 8007 |
| Flower | 5555 | 5555 |
| Grafana | 3000 | 3000 |
| Prometheus | 9090 | 9090 |
| Tempo | 3200 | 3200 |
