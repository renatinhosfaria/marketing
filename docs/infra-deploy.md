# Infra e Deploy

## Docker Compose

Ambiente recomendado para desenvolvimento:

```bash
docker compose up -d --build
docker compose exec marketing-api alembic upgrade head
```

Servicos centrais:

- frontend;
- ML API;
- Facebook Ads API;
- workers/beat;
- redis;
- flower.

## Healthchecks

Validacoes basicas:

```bash
curl -sf http://localhost:8001/api/v1/health
curl -sf http://localhost:8003/api/v1/facebook-ads/health/simple
bash scripts/healthcheck.sh
```

## Deploy em Swarm

Para producao:

```bash
docker stack deploy -c marketing-stack.yml marketing
```

Redeploy utilitario:

```bash
bash scripts/redeploy.sh all
```
