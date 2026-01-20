"""
Router principal da API v1.
Agrega todos os endpoints do microserviço ML.
"""

from fastapi import APIRouter, Depends

from app.core.security import verify_api_key
from app.api.v1.endpoints import (
    health,
    predictions,
    forecasts,
    classifications,
    recommendations,
    anomalies,
    models,
)
from app.api.v1.agent import router as agent_router

# Router principal (sem autenticação para health)
api_router = APIRouter()

# Health endpoints (sem autenticação)
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health"]
)

# Endpoints autenticados
authenticated_router = APIRouter(dependencies=[Depends(verify_api_key)])

authenticated_router.include_router(
    predictions.router,
    prefix="/predictions",
    tags=["Previsões"]
)

authenticated_router.include_router(
    forecasts.router,
    prefix="/forecasts",
    tags=["Forecasts"]
)

authenticated_router.include_router(
    classifications.router,
    prefix="/classifications",
    tags=["Classificações"]
)

authenticated_router.include_router(
    recommendations.router,
    prefix="/recommendations",
    tags=["Recomendações"]
)

authenticated_router.include_router(
    anomalies.router,
    prefix="/anomalies",
    tags=["Anomalias"]
)

authenticated_router.include_router(
    models.router,
    prefix="/models",
    tags=["Modelos"]
)

# Agent router (usa autenticação JWT do usuário)
authenticated_router.include_router(
    agent_router,
    tags=["Agente IA"]
)

# Incluir rotas autenticadas no router principal
api_router.include_router(authenticated_router)
