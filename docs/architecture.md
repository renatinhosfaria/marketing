# Arquitetura

## Componentes

- FastAPI ML: `app/main.py` (API de previsoes e modelos)
- FastAPI Agent: `app/agent_main.py` (Agente IA)
- FastAPI Facebook Ads: `app/fb_ads_main.py` (OAuth, sync, insights)
- Celery: `app/celery.py` (workers e scheduler)
- Redis: broker e cache
- PostgreSQL: banco principal
- Frontend Next.js: `frontend/`

## Fluxo de dados

1. Facebook Ads sincroniza campanhas e insights para o banco.
2. ML consome insights para treinos, previsoes e recomendacoes.
3. Agente IA consulta dados e sintetiza respostas.
4. Frontend consome APIs via HTTP.

## Portas

- Frontend: 8000 -> 3001
- ML API: 8001 -> 8000
- Agent API: 8002 -> 8001
- Facebook Ads API: 8003 -> 8002
- Worker ML: 8004 -> 8004
- Worker FB Ads: 8005 -> 8005
- Celery Beat: 8006 -> 8006
- Redis: 8007 -> 6379
- Flower: 5555 -> 5555
