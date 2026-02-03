# Runbooks

## Operacao diaria

- Subir stack: `docker compose up -d`
- Verificar health:
  - ML: `curl http://localhost:8001/api/v1/health`
  - Agent: `curl http://localhost:8002/api/v1/health`
  - FB Ads: `curl http://localhost:8003/api/v1/facebook-ads/health/simple`
- Ver logs: `docker compose logs -f <servico>`

## Incidentes comuns

- Banco indisponivel: validar `DATABASE_URL` e conectividade do Postgres.
- Redis indisponivel: validar `REDIS_URL` e container `marketing-redis`.
- Celery sem tasks: verificar queues e Flower em `http://localhost:5555/flower`.
- OAuth falhando: validar `FACEBOOK_APP_ID/SECRET` e URLs de callback.
