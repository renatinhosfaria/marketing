# FamaChat ML - Documentacao

Documentacao oficial do microservico de Machine Learning, Agente IA e integracao com Facebook Ads.

## Comece aqui

1. Configure o `.env` (ver `docs/configuration.md`).
2. Suba o stack: `docker compose up -d`.
3. Verifique health:
   - ML: `GET http://localhost:8001/api/v1/health`
   - Agent: `GET http://localhost:8002/api/v1/health`
   - Facebook Ads: `GET http://localhost:8003/api/v1/facebook-ads/health/simple`

## Mapa dos servicos

| Servico | Container | Porta host -> container | Funcao |
| --- | --- | --- | --- |
| Frontend | marketing-frontend | 8000 -> 3001 | UI Next.js |
| ML API | marketing-api | 8001 -> 8000 | Previsoes/ML |
| Agent API | marketing-agent | 8002 -> 8001 | Agente IA |
| Facebook Ads API | marketing-fb-ads | 8003 -> 8002 | OAuth/Sync/Insights |
| Worker ML | marketing-worker | 8004 -> 8004 | Celery filas training/ml |
| Worker FB Ads | marketing-fb-ads-worker | 8005 -> 8005 | Celery fila default |
| Celery Beat | marketing-beat | 8006 -> 8006 | Scheduler |
| Redis | marketing-redis | 8007 -> 6379 | Broker/Cache |
| Flower | marketing-flower | 5555 -> 5555 | Monitoramento Celery |

## Documentos

- `docs/overview.md`
- `docs/architecture.md`
- `docs/backend.md`
- `docs/frontend.md`
- `docs/apis.md`
- `docs/ml.md`
- `docs/agent.md`
- `docs/facebook-ads.md`
- `docs/infra-deploy.md`
- `docs/configuration.md`
- `docs/observability.md`
- `docs/runbooks.md`
- `docs/testing.md`
- `docs/contributing.md`
