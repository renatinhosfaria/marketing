"""
Router agregador da API v1 - exclusivo para Machine Learning.
Facebook Ads roda em processo separado (porta 8002).
"""

from fastapi import APIRouter


# Importar routers do projeto ML
from projects.ml.api.health import router as health_router
from projects.ml.api.predictions import router as predictions_router
from projects.ml.api.forecasts import router as forecasts_router
from projects.ml.api.impact import router as impact_router
from projects.ml.api.classifications import router as classifications_router
from projects.ml.api.recommendations import router as recommendations_router
from projects.ml.api.anomalies import router as anomalies_router
from projects.ml.api.models import router as models_router

# Router principal (sem autenticação para health)
api_router = APIRouter()

# Health endpoints (sem autenticação)
api_router.include_router(
    health_router,
    prefix="/health",
    tags=["Health"]
)


# Endpoints autenticados
authenticated_router = APIRouter()

authenticated_router.include_router(
    predictions_router,
    prefix="/predictions",
    tags=["Previsões"]
)

authenticated_router.include_router(
    forecasts_router,
    prefix="/forecasts",
    tags=["Forecasts"]
)

authenticated_router.include_router(
    impact_router,
    prefix="/impact",
    tags=["Impacto Causal"]
)

authenticated_router.include_router(
    classifications_router,
    prefix="/classifications",
    tags=["Classificações"]
)

authenticated_router.include_router(
    recommendations_router,
    prefix="/recommendations",
    tags=["Recomendações"]
)

authenticated_router.include_router(
    anomalies_router,
    prefix="/anomalies",
    tags=["Anomalias"]
)

authenticated_router.include_router(
    models_router,
    prefix="/models",
    tags=["Modelos"]
)

# Incluir rotas autenticadas no router principal
api_router.include_router(authenticated_router)
