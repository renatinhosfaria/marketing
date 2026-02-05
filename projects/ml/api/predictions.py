"""
Endpoints de previsões (forecasting).
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.session import get_db
from projects.ml.db.repositories.ml_repo import MLRepository
from projects.ml.db.repositories.insights_repo import InsightsRepository
from projects.ml.db.models import PredictionType
from shared.core.logging import get_logger
from projects.ml.algorithms.models.timeseries.forecaster import forecaster

logger = get_logger(__name__)
router = APIRouter()


# ==================== SCHEMAS ====================

class PredictionRequest(BaseModel):
    """Request para gerar previsão."""
    config_id: int = Field(..., description="ID da configuração FB Ads")
    entity_type: str = Field("campaign", description="Tipo de entidade")
    entity_id: str = Field(..., description="ID da entidade")
    horizon_days: int = Field(7, ge=1, le=30, description="Dias de previsão")


class PredictionResponse(BaseModel):
    """Resposta com previsão."""
    id: int
    entity_type: str
    entity_id: str
    prediction_type: str
    forecast_date: datetime
    predicted_value: float
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    created_at: datetime


class PredictionSeriesResponse(BaseModel):
    """Série de previsões."""
    entity_type: str
    entity_id: str
    predictions: list[PredictionResponse]


class BatchPredictionRequest(BaseModel):
    """Request para previsões em batch."""
    config_id: int
    entity_type: str = "campaign"
    entity_ids: list[str]
    horizon_days: int = 7


# ==================== ENDPOINTS ====================

@router.post("/cpl", response_model=list[PredictionResponse])
async def predict_cpl(
    request: PredictionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Gera previsão de CPL (Custo por Lead).

    - **config_id**: ID da configuração do Facebook Ads
    - **entity_type**: Tipo da entidade (campaign, adset, ad)
    - **entity_id**: ID da entidade no Facebook
    - **horizon_days**: Número de dias para prever (1-30)
    """
    ml_repo = MLRepository(db)
    logger.info(
        "Solicitação de previsão CPL",
        config_id=request.config_id,
        entity_id=request.entity_id
    )
    insights_repo = InsightsRepository(db)
    start_date = datetime.utcnow() - timedelta(days=30)
    end_date = datetime.utcnow()
    df = await insights_repo.get_insights_as_dataframe(
        request.config_id,
        start_date,
        end_date,
        entity_type=request.entity_type,
        entity_id=request.entity_id,
    )

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dados insuficientes para previsão",
        )

    daily = df.groupby("date").agg({"spend": "sum", "leads": "sum"}).reset_index()
    forecasts = forecaster.forecast_cpl(
        daily,
        request.entity_type,
        request.entity_id,
        horizon_days=request.horizon_days,
    )

    responses = []
    for item in forecasts.forecasts:
        prediction = await ml_repo.create_prediction(
            model_id=None,
            config_id=request.config_id,
            entity_type=request.entity_type,
            entity_id=request.entity_id,
            prediction_type=PredictionType.CPL_FORECAST,
            forecast_date=item.forecast_date,
            predicted_value=item.predicted_value,
            horizon_days=request.horizon_days,
            confidence_lower=item.confidence_lower,
            confidence_upper=item.confidence_upper,
        )
        responses.append(
            PredictionResponse(
                id=prediction.id,
                entity_type=prediction.entity_type,
                entity_id=prediction.entity_id,
                prediction_type=prediction.prediction_type.value,
                forecast_date=prediction.forecast_date,
                predicted_value=prediction.predicted_value,
                confidence_lower=prediction.confidence_lower,
                confidence_upper=prediction.confidence_upper,
                created_at=prediction.created_at,
            )
        )

    return responses


@router.post("/leads", response_model=list[PredictionResponse])
async def predict_leads(
    request: PredictionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Gera previsão de número de leads.

    - **config_id**: ID da configuração do Facebook Ads
    - **entity_type**: Tipo da entidade (campaign, adset, ad)
    - **entity_id**: ID da entidade no Facebook
    - **horizon_days**: Número de dias para prever (1-30)
    """
    logger.info(
        "Solicitação de previsão de leads",
        config_id=request.config_id,
        entity_id=request.entity_id
    )

    ml_repo = MLRepository(db)
    insights_repo = InsightsRepository(db)
    start_date = datetime.utcnow() - timedelta(days=30)
    end_date = datetime.utcnow()
    df = await insights_repo.get_insights_as_dataframe(
        request.config_id,
        start_date,
        end_date,
        entity_type=request.entity_type,
        entity_id=request.entity_id,
    )

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dados insuficientes para previsão",
        )

    daily = df.groupby("date").agg({"leads": "sum"}).reset_index()
    forecasts = forecaster.forecast_leads(
        daily,
        request.entity_type,
        request.entity_id,
        horizon_days=request.horizon_days,
    )

    responses = []
    for item in forecasts.forecasts:
        prediction = await ml_repo.create_prediction(
            model_id=None,
            config_id=request.config_id,
            entity_type=request.entity_type,
            entity_id=request.entity_id,
            prediction_type=PredictionType.LEADS_FORECAST,
            forecast_date=item.forecast_date,
            predicted_value=item.predicted_value,
            horizon_days=request.horizon_days,
            confidence_lower=item.confidence_lower,
            confidence_upper=item.confidence_upper,
        )
        responses.append(
            PredictionResponse(
                id=prediction.id,
                entity_type=prediction.entity_type,
                entity_id=prediction.entity_id,
                prediction_type=prediction.prediction_type.value,
                forecast_date=prediction.forecast_date,
                predicted_value=prediction.predicted_value,
                confidence_lower=prediction.confidence_lower,
                confidence_upper=prediction.confidence_upper,
                created_at=prediction.created_at,
            )
        )

    return responses


@router.post("/batch", response_model=list[PredictionResponse])
async def batch_predictions(
    request: BatchPredictionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Gera previsões em batch para múltiplas entidades.
    """
    logger.info(
        "Solicitação de previsões em batch",
        config_id=request.config_id,
        entity_count=len(request.entity_ids)
    )

    results = []
    for entity_id in request.entity_ids:
        single_req = PredictionRequest(
            config_id=request.config_id,
            entity_type=request.entity_type,
            entity_id=entity_id,
            horizon_days=request.horizon_days,
        )
        try:
            predictions = await predict_cpl(single_req, db)
            results.extend(predictions)
        except HTTPException:
            continue

    return results


@router.get("/series/{entity_type}/{entity_id}", response_model=PredictionSeriesResponse)
async def get_prediction_series(
    entity_type: str,
    entity_id: str,
    config_id: int = Query(..., description="ID da configuração"),
    prediction_type: Optional[str] = Query(None, description="Tipo de previsão"),
    limit: int = Query(30, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """
    Obtém série histórica de previsões de uma entidade.
    """
    ml_repo = MLRepository(db)

    pred_type = None
    if prediction_type:
        try:
            pred_type = PredictionType(prediction_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tipo de previsão inválido: {prediction_type}"
            )

    predictions = await ml_repo.get_predictions(
        config_id=config_id,
        entity_type=entity_type,
        entity_id=entity_id,
        prediction_type=pred_type,
        limit=limit
    )

    return PredictionSeriesResponse(
        entity_type=entity_type,
        entity_id=entity_id,
        predictions=[
            PredictionResponse(
                id=p.id,
                entity_type=p.entity_type,
                entity_id=p.entity_id,
                prediction_type=p.prediction_type.value,
                forecast_date=p.forecast_date,
                predicted_value=p.predicted_value,
                confidence_lower=p.confidence_lower,
                confidence_upper=p.confidence_upper,
                created_at=p.created_at
            )
            for p in predictions
        ]
    )
