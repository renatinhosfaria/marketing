"""
Endpoints de forecasts (previsoes agregadas).
"""

from datetime import datetime, timedelta, date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.session import get_db
from projects.ml.db.repositories.ml_repo import MLRepository
from projects.ml.db.repositories.insights_repo import InsightsRepository

router = APIRouter()


class ForecastItem(BaseModel):
    id: int
    config_id: int
    entity_type: str
    entity_id: str
    target_metric: str
    horizon_days: int
    method: str
    model_version: Optional[str] = None
    window_days: Optional[int] = None
    forecast_date: date
    predictions: list[dict]
    insufficient_data: bool
    created_at: datetime


class ForecastListResponse(BaseModel):
    config_id: int
    total: int
    page: int
    page_size: int
    data: list[ForecastItem]
    forecasts: list[ForecastItem]


class ForecastSummaryResponse(BaseModel):
    config_id: int
    window_days: int
    date_from: datetime
    date_to: datetime
    total: int
    by_metric: dict[str, int]
    trend: dict[str, str]


@router.get("", response_model=ForecastListResponse)
async def list_forecasts(
    config_id: Optional[int] = Query(None, description="ID da configuração"),
    ad_account_id: Optional[str] = Query(None, description="Ad account ID"),
    entity_type: Optional[str] = Query(None, description="Tipo de entidade"),
    entity_id: Optional[str] = Query(None, description="ID da entidade"),
    campaign_id: Optional[str] = Query(None, description="Alias de entity_id"),
    target_metric: Optional[str] = Query(None, description="Metrica alvo"),
    metric: Optional[str] = Query(None, description="Alias para target_metric"),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    page: Optional[int] = Query(None, ge=1),
    page_size: Optional[int] = Query(None, ge=1, le=200),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    ml_repo = MLRepository(db)

    if config_id is None and ad_account_id:
        insights_repo = InsightsRepository(db)
        config = await insights_repo.get_config_by_ad_account_id(ad_account_id)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Configuração não encontrada para ad_account_id",
            )
        config_id = config.id

    if config_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="config_id é obrigatório",
        )

    if campaign_id and not entity_id:
        entity_id = campaign_id
        if not entity_type:
            entity_type = "campaign"

    metric_filter = target_metric or metric

    if page and page_size:
        offset = (page - 1) * page_size
        limit = page_size

    forecasts = await ml_repo.get_forecasts(
        config_id=config_id,
        entity_type=entity_type,
        entity_id=entity_id,
        target_metric=metric_filter,
        start_date=date_from,
        end_date=date_to,
        limit=limit,
        offset=offset,
    )
    total = await ml_repo.count_forecasts(
        config_id=config_id,
        entity_type=entity_type,
        entity_id=entity_id,
        target_metric=metric_filter,
        start_date=date_from,
        end_date=date_to,
    )

    data = [
        ForecastItem(
            id=f.id,
            config_id=f.config_id,
            entity_type=f.entity_type,
            entity_id=f.entity_id,
            target_metric=f.target_metric,
            horizon_days=f.horizon_days,
            method=f.method,
            model_version=f.model_version,
            window_days=f.window_days,
            forecast_date=f.forecast_date,
            predictions=f.predictions or [],
            insufficient_data=f.insufficient_data,
            created_at=f.created_at,
        )
        for f in forecasts
    ]

    page_value = page or 1
    page_size_value = page_size or limit

    return ForecastListResponse(
        config_id=config_id,
        total=total,
        page=page_value,
        page_size=page_size_value,
        data=data,
        forecasts=data,
    )


@router.get("/summary", response_model=ForecastSummaryResponse)
async def forecast_summary(
    config_id: Optional[int] = Query(None, description="ID da configuração"),
    ad_account_id: Optional[str] = Query(None, description="Ad account ID"),
    window_days: int = Query(7, ge=1, le=30),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    ml_repo = MLRepository(db)

    if config_id is None and ad_account_id:
        insights_repo = InsightsRepository(db)
        config = await insights_repo.get_config_by_ad_account_id(ad_account_id)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Configuração não encontrada para ad_account_id",
            )
        config_id = config.id

    if config_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="config_id é obrigatório",
        )

    now = datetime.utcnow()
    if date_from or date_to:
        start = date_from or (now - timedelta(days=window_days))
        end = date_to or now
        window_days = (end - start).days or window_days
    else:
        end = now
        start = now - timedelta(days=window_days)

    forecasts = await ml_repo.get_forecasts(
        config_id=config_id,
        start_date=start,
        end_date=end,
        limit=500,
        offset=0,
    )

    by_metric: dict[str, int] = {}
    trend: dict[str, str] = {}
    trend_scores: dict[str, list[float]] = {}

    for f in forecasts:
        metric = f.target_metric
        by_metric[metric] = by_metric.get(metric, 0) + 1
        preds = f.predictions or []
        if len(preds) >= 2:
            first = preds[0].get("predicted_value", preds[0].get("value", 0))
            last = preds[-1].get("predicted_value", preds[-1].get("value", 0))
            trend_scores.setdefault(metric, []).append(last - first)

    for metric, deltas in trend_scores.items():
        avg_delta = sum(deltas) / len(deltas) if deltas else 0
        if avg_delta > 0:
            trend[metric] = "up"
        elif avg_delta < 0:
            trend[metric] = "down"
        else:
            trend[metric] = "flat"

    return ForecastSummaryResponse(
        config_id=config_id,
        window_days=window_days,
        date_from=start,
        date_to=end,
        total=len(forecasts),
        by_metric=by_metric,
        trend=trend,
    )
