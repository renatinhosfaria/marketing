"""
Endpoints de detecção de anomalias.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.repositories.ml_repo import MLRepository
from app.db.models.ml_models import AnomalySeverity
from app.core.logging import get_logger
from app.services.anomaly_service import AnomalyService

logger = get_logger(__name__)
router = APIRouter()


# ==================== SCHEMAS ====================

class AnomalyResponse(BaseModel):
    """Resposta de anomalia."""
    id: int
    config_id: int
    entity_type: str
    entity_id: str
    anomaly_type: str
    type: Optional[str] = None
    metric_name: str
    observed_value: float
    expected_value: float
    current_value: Optional[float] = None
    baseline_value: Optional[float] = None
    baseline_window_days: Optional[int] = None
    deviation_score: float
    severity: str
    is_acknowledged: bool
    status: Optional[str] = None
    acknowledged_by: Optional[int] = None
    resolution_notes: Optional[str] = None
    anomaly_date: datetime
    detected_at: datetime
    recommendation_id: Optional[int] = None


class AnomalyListResponse(BaseModel):
    """Lista de anomalias."""
    config_id: int
    total: int
    page: int
    page_size: int
    by_severity: dict[str, int]
    unacknowledged_count: int
    data: list[AnomalyResponse]
    anomalies: list[AnomalyResponse]


class AnomalySummaryResponse(BaseModel):
    """Resumo de anomalias."""
    config_id: int
    window_days: int
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    total: int
    by_severity: dict[str, int]
    by_type: dict[str, int]
    unacknowledged: int
    critical_count: int
    last_detected_at: Optional[datetime] = None


class DetectRequest(BaseModel):
    """Request para detectar anomalias."""
    config_id: int = Field(..., description="ID da configuração FB Ads")
    days: int = Field(1, ge=1, le=7, description="Dias para analisar")
    entity_type: Optional[str] = Field(None, description="Tipo de entidade")
    entity_id: Optional[str] = Field(None, description="ID da entidade")


class DetectResponse(BaseModel):
    """Resposta de detecção."""
    config_id: int
    detected_count: int
    anomalies: list[AnomalyResponse]


class AcknowledgeRequest(BaseModel):
    """Request para acknowledgar anomalia."""
    user_id: int = Field(..., description="ID do usuário")
    notes: Optional[str] = Field(None, description="Notas de resolução")


# ==================== ENDPOINTS ====================

@router.get("", response_model=AnomalyListResponse)
async def list_anomalies(
    config_id: Optional[int] = Query(None, description="ID da configuração"),
    ad_account_id: Optional[str] = Query(None, description="Ad account ID"),
    severity: Optional[str] = Query(None, description="Filtrar por severidade"),
    acknowledged: Optional[bool] = Query(None, description="Filtrar por status"),
    is_acknowledged: Optional[bool] = Query(None, description="Alias para acknowledged"),
    status: Optional[str] = Query(None, description="open|closed"),
    anomaly_type: Optional[str] = Query(None, description="Filtrar por tipo"),
    type_filter: Optional[str] = Query(None, alias="type", description="Alias para anomaly_type"),
    entity_id: Optional[str] = Query(None, description="ID da entidade"),
    campaign_id: Optional[str] = Query(None, description="Alias de entity_id"),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    start_date: Optional[datetime] = Query(None, description="Alias para date_from"),
    end_date: Optional[datetime] = Query(None, description="Alias para date_to"),
    days: int = Query(7, ge=1, le=30, description="Período em dias"),
    page: Optional[int] = Query(None, ge=1),
    page_size: Optional[int] = Query(None, ge=1, le=200),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    Lista anomalias detectadas.

    - **config_id**: ID da configuração do Facebook Ads
    - **severity**: Filtrar por severidade (LOW, MEDIUM, HIGH, CRITICAL)
    - **acknowledged**: Filtrar por status de acknowledgment
    - **days**: Período de busca em dias
    """
    ml_repo = MLRepository(db)

    if config_id is None and ad_account_id:
        from app.db.repositories.insights_repo import InsightsRepository
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

    if is_acknowledged is not None:
        acknowledged = is_acknowledged
    if status:
        if status not in {"open", "closed"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Status inválido: {status}",
            )
        acknowledged = status == "closed"

    if type_filter and not anomaly_type:
        anomaly_type = type_filter

    if start_date and not date_from:
        date_from = start_date
    if end_date and not date_to:
        date_to = end_date

    if campaign_id and not entity_id:
        entity_id = campaign_id

    sev_filter = None
    if severity:
        try:
            sev_filter = AnomalySeverity(severity)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Severidade inválida: {severity}"
            )

    if page and page_size:
        offset = (page - 1) * page_size
        limit = page_size

    anomalies = await ml_repo.get_anomalies(
        config_id=config_id,
        severity=sev_filter,
        acknowledged=acknowledged,
        days=days,
        start_date=date_from,
        end_date=date_to,
        anomaly_type=anomaly_type,
        entity_id=entity_id,
        limit=limit,
        offset=offset,
    )
    total = await ml_repo.count_anomalies(
        config_id=config_id,
        severity=sev_filter,
        acknowledged=acknowledged,
        days=days,
        start_date=date_from,
        end_date=date_to,
        anomaly_type=anomaly_type,
        entity_id=entity_id,
    )

    # Estatísticas
    by_severity = {}
    unack_count = 0
    for a in anomalies:
        sev_val = a.severity.value
        by_severity[sev_val] = by_severity.get(sev_val, 0) + 1
        if not a.is_acknowledged:
            unack_count += 1

    data = [
        AnomalyResponse(
            id=a.id,
            config_id=a.config_id,
            entity_type=a.entity_type,
            entity_id=a.entity_id,
            anomaly_type=a.anomaly_type,
            type=a.anomaly_type,
            metric_name=a.metric_name,
            observed_value=a.observed_value,
            expected_value=a.expected_value,
            current_value=a.observed_value,
            baseline_value=a.expected_value,
            baseline_window_days=30,
            deviation_score=a.deviation_score,
            severity=a.severity.value,
            is_acknowledged=a.is_acknowledged,
            status="closed" if a.is_acknowledged else "open",
            acknowledged_by=a.acknowledged_by,
            resolution_notes=a.resolution_notes,
            anomaly_date=a.anomaly_date,
            detected_at=a.detected_at,
            recommendation_id=a.recommendation_id
        )
        for a in anomalies
    ]

    page_value = page or 1
    page_size_value = page_size or limit

    return AnomalyListResponse(
        config_id=config_id,
        total=total,
        page=page_value,
        page_size=page_size_value,
        by_severity=by_severity,
        unacknowledged_count=unack_count,
        data=data,
        anomalies=data,
    )


@router.get("/summary", response_model=AnomalySummaryResponse)
async def get_anomaly_summary(
    config_id: Optional[int] = Query(None, description="ID da configuração"),
    ad_account_id: Optional[str] = Query(None, description="Ad account ID"),
    days: int = Query(7, ge=1, le=30, description="Período em dias"),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Obtém resumo de anomalias do período.
    """
    ml_repo = MLRepository(db)

    if config_id is None and ad_account_id:
        from app.db.repositories.insights_repo import InsightsRepository
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

    if date_from or date_to:
        start = date_from
        end = date_to
        window_days = (
            (end - start).days if start and end else days
        )
    else:
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        window_days = days

    anomalies = await ml_repo.get_anomalies(
        config_id=config_id,
        days=days,
        start_date=start,
        end_date=end,
    )

    by_severity = {}
    by_type = {}
    unack_count = 0
    critical_count = 0

    for a in anomalies:
        sev_val = a.severity.value
        by_severity[sev_val] = by_severity.get(sev_val, 0) + 1

        by_type[a.anomaly_type] = by_type.get(a.anomaly_type, 0) + 1

        if not a.is_acknowledged:
            unack_count += 1
        if a.severity == AnomalySeverity.CRITICAL:
            critical_count += 1

    for sev_key in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        by_severity.setdefault(sev_key, 0)

    last_detected_at = max(
        (a.detected_at for a in anomalies),
        default=None,
    )

    return AnomalySummaryResponse(
        config_id=config_id,
        window_days=window_days,
        date_from=start,
        date_to=end,
        total=len(anomalies),
        by_severity=by_severity,
        by_type=by_type,
        unacknowledged=unack_count,
        critical_count=critical_count,
        last_detected_at=last_detected_at,
    )


@router.post("/detect", response_model=DetectResponse)
async def detect_anomalies(
    request: DetectRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Executa detecção de anomalias.

    Analisa métricas recentes e identifica comportamentos atípicos.
    """
    logger.info(
        "Solicitação de detecção de anomalias",
        config_id=request.config_id,
        days=request.days
    )

    service = AnomalyService(db)
    campaign_ids = None
    if request.entity_type == "campaign" and request.entity_id:
        campaign_ids = [request.entity_id]
    result = await service.detect_anomalies(
        config_id=request.config_id,
        campaign_ids=campaign_ids,
        days_to_analyze=request.days,
    )

    return DetectResponse(
        config_id=request.config_id,
        detected_count=result.anomalies_detected,
        anomalies=[
            AnomalyResponse(
                id=0,
                config_id=request.config_id,
                entity_type=a["entity_type"],
                entity_id=a["entity_id"],
                anomaly_type=a["anomaly_type"],
                metric_name=a["metric_name"],
                observed_value=a["observed_value"],
                expected_value=a["expected_value"],
                deviation_score=a["deviation_score"],
                severity=a["severity"],
                is_acknowledged=False,
                acknowledged_by=None,
                resolution_notes=None,
                anomaly_date=datetime.fromisoformat(a["anomaly_date"])
                if isinstance(a["anomaly_date"], str)
                else a["anomaly_date"],
                detected_at=datetime.utcnow(),
                recommendation_id=None,
            )
            for a in result.anomalies
        ],
    )


@router.post("/{anomaly_id}/acknowledge")
async def acknowledge_anomaly(
    anomaly_id: int,
    request: AcknowledgeRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Marca anomalia como reconhecida.
    """
    ml_repo = MLRepository(db)

    await ml_repo.acknowledge_anomaly(
        anomaly_id=anomaly_id,
        user_id=request.user_id,
        notes=request.notes
    )

    logger.info(
        "Anomalia reconhecida",
        anomaly_id=anomaly_id,
        user_id=request.user_id
    )

    return {"status": "acknowledged", "anomaly_id": anomaly_id}
