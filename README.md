# Marketing

Plataforma de operacao para Facebook Ads com duas APIs FastAPI (ML e Facebook Ads), frontend Next.js e processamento assincrono com Celery.

## Quickstart

```bash
# 1) configurar variaveis de ambiente
# crie ou atualize o arquivo .env na raiz

# 2) subir stack local
docker compose up -d --build

# 3) aplicar migracoes
docker compose exec marketing-api alembic upgrade head

# 4) validar saude
curl -sf http://localhost:8001/api/v1/health
curl -sf http://localhost:8003/api/v1/facebook-ads/health/simple
```

## Documentacao

A documentacao completa esta em [docs/README.md](docs/README.md).

Leitura recomendada:

1. [docs/overview.md](docs/overview.md)
2. [docs/architecture.md](docs/architecture.md)
3. [docs/configuration.md](docs/configuration.md)
4. [docs/infra-deploy.md](docs/infra-deploy.md)
5. [docs/apis.md](docs/apis.md)

## Stack

- Backend: Python 3.11, FastAPI, SQLAlchemy, Alembic
- Async: Celery, Redis, Flower
- Frontend: Next.js 16, React 19, TypeScript
- ML: scikit-learn, XGBoost, LightGBM, Prophet
- Observabilidade: OpenTelemetry, Prometheus, Grafana, Tempo

## Licenca

Projeto proprietario FamaChat. Todos os direitos reservados.
