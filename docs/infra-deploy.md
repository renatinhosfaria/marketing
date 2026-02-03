# Infraestrutura e Deploy

## Docker Compose

Stack principal em `docker-compose.yml`, com imagens para:
- API ML, Agent, Facebook Ads
- Celery workers e beat
- Redis
- Flower
- Frontend

## Healthchecks

Cada servico possui healthcheck configurado (API, Agent, FB Ads, Redis, Flower).
Use `docker compose ps` e `docker compose logs <servico>` para diagnostico.

## Volumes

- `models_storage` para modelos ML
- `logs` para logs
- `redis_ml_data` e `celerybeat_schedule`
