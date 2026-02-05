"""
Router da API v1 exclusivo para o Agente IA.
Importa apenas os routers necessarios para o microservico do Agent.
"""

from fastapi import APIRouter

from projects.agent.api.health import router as agent_health_router
from projects.agent.api.router import router as agent_router

# Router principal (sem autenticacao para health)
agent_api_router = APIRouter()

# Health endpoints (sem autenticacao)
agent_api_router.include_router(
    agent_health_router,
    prefix="/health",
    tags=["Health"],
)

# Endpoints autenticados do Agent
authenticated_router = APIRouter()

authenticated_router.include_router(
    agent_router,
    tags=["Agente IA"],
)

# Incluir rotas autenticadas no router principal
agent_api_router.include_router(authenticated_router)
