# Backend

## Entrypoints

- `app/main.py`: API ML (previsoes, classificacoes, recomendacoes, anomalias)
- `app/agent_main.py`: API do Agente IA
- `app/fb_ads_main.py`: API Facebook Ads

## Rotas

- ML: `app/router.py` agrega rotas de `projects/ml/api/*`
- Agent: `app/agent_router.py`
- Facebook Ads: `app/fb_ads_router.py`

## Celery

Configuracao central em `app/celery.py` com filas `ml`, `training` e `default`.
Jobs agendados incluem pipeline diario, retraining, recomendacoes, anomalias e sync do Facebook Ads.

## Persistencia e config

- Sessao async e engine: `shared/infrastructure/persistence/database.py`
- Config global: `shared/infrastructure/config/settings.py`
- Logging estruturado: `shared/infrastructure/logging/structlog_config.py`
