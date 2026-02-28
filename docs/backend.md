# Backend

## Entrypoints

Entrypoints principais:

- `app/main.py`: inicializa ML API.
- `app/fb_ads_main.py`: inicializa Facebook Ads API.
- `app/celery.py`: inicializa app Celery, filas e beat schedule.
- `app/router.py` e `app/fb_ads_router.py`: agregam roteadores por contexto.

## Rotas

Estrutura de alto nivel:

- ML API (`/api/v1`):
- `/health`
- `/predictions`
- `/forecasts`
- `/impact`
- `/classifications`
- `/recommendations`
- `/anomalies`
- `/models`

- Facebook Ads API (`/api/v1/facebook-ads`):
- `/health`
- `/oauth`
- `/config`
- `/sync`
- `/campaigns` e `/adsets` e `/ads`
- `/insights`

Observacao: documentacao OpenAPI (`/docs`) e habilitada apenas quando `DEBUG=true`.

## Celery

Filas utilizadas:

- `training, ml`: workloads de treino e tarefas analiticas ML.
- `default`: tarefas de Facebook Ads.

Jobs agendados incluem:

- pipeline diario de ML;
- retreinamento de modelos;
- sincronizacao incremental e full de Facebook Ads;
- consolidacao periodica de insights.

## Persistencia e migracoes

- ORM: SQLAlchemy (async + sync where needed).
- Migracoes: Alembic (`alembic upgrade head`).
- URL de banco controlada por `DATABASE_URL`.
