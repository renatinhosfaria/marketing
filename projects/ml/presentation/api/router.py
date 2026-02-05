"""Router principal do módulo ML (Clean Architecture)."""

from fastapi import APIRouter

from projects.ml.presentation.api.routes import (
    health,
    predictions,
    forecasts,
    classifications,
    recommendations,
    anomalies,
    models,
)

# Router público (health checks)
public_router = APIRouter()
public_router.include_router(health.router, prefix="/health", tags=["ML - Health"])

# Router autenticado
authenticated_router = APIRouter()
authenticated_router.include_router(predictions.router, prefix="/predictions", tags=["ML - Predictions"])
authenticated_router.include_router(forecasts.router, prefix="/forecasts", tags=["ML - Forecasts"])
authenticated_router.include_router(classifications.router, prefix="/classifications", tags=["ML - Classifications"])
authenticated_router.include_router(recommendations.router, prefix="/recommendations", tags=["ML - Recommendations"])
authenticated_router.include_router(anomalies.router, prefix="/anomalies", tags=["ML - Anomalies"])
authenticated_router.include_router(models.router, prefix="/models", tags=["ML - Models"])

# Router principal
ml_router = APIRouter()
ml_router.include_router(public_router)
ml_router.include_router(authenticated_router)
