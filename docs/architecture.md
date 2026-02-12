# Arquitetura

## Componentes

- **Traefik**: proxy reverso e TLS (portas 80/443)
- **FastAPI ML**: `app/main.py` (API de previsoes e modelos)
- **FastAPI Facebook Ads**: `app/fb_ads_main.py` (OAuth, sync, insights)
- **FastAPI Agent**: `app/agent_main.py` (AI Agent multi-agente)
- **Celery**: `app/celery.py` (workers e scheduler)
- **Redis**: broker e cache
- **PostgreSQL**: banco principal
- **Frontend Next.js**: `frontend/`

## Fluxo de dados

1. Facebook Ads sincroniza campanhas e insights para o banco.
2. ML consome insights para treinos, previsoes e recomendacoes.
3. Frontend consome APIs via Traefik (HTTPS).
4. Agent orquestra subagentes que consultam ML API e banco.

## Roteamento (Traefik)

Todas as requisicoes passam pelo Traefik, que roteia por path e prioridade:

| Rota | Servico | Porta interna |
|------|---------|---------------|
| `/api/v1/agent` | Agent API | 8001 |
| `/api/v1/facebook-ads` | FB Ads API | 8002 |
| `/api/` | ML API | 8000 |
| `/flower/` | Flower | 5555 |
| `/` | Frontend | 3001 |

## Portas internas dos containers

| Servico | Porta |
|---------|-------|
| Frontend | 3001 |
| ML API | 8000 |
| FB Ads API | 8002 |
| Agent API | 8001 |
| Redis | 6379 |
| Flower | 5555 |
