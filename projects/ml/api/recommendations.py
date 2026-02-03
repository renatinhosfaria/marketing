"""
Endpoints de recomendações de otimização.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from shared.db.session import get_db
from projects.ml.db.repositories.ml_repo import MLRepository
from projects.ml.db.models import RecommendationType, MLRecommendation
from shared.core.logging import get_logger
from projects.ml.services.recommendation_service import RecommendationService

logger = get_logger(__name__)
router = APIRouter()


# ==================== SCHEMAS ====================

class RecommendationResponse(BaseModel):
    """Resposta de recomendação."""
    id: int
    config_id: int
    entity_type: str
    entity_id: str
    recommendation_type: str
    action_type: Optional[str] = None
    priority: int
    priority_level: Optional[str] = None
    title: str
    description: str
    suggested_action: Optional[dict] = None
    confidence_score: float
    reasoning: Optional[dict] = None
    reason_code: Optional[str] = None
    explain: Optional[dict] = None
    is_active: bool
    was_applied: bool
    applied_at: Optional[datetime] = None
    dismissed: bool
    dismissed_reason: Optional[str] = None
    status: Optional[str] = None
    created_at: datetime
    expires_at: Optional[datetime] = None


class RecommendationListResponse(BaseModel):
    """Lista de recomendações."""
    config_id: int
    total: int
    page: int
    page_size: int
    by_type: dict[str, int]
    data: list[RecommendationResponse]
    recommendations: list[RecommendationResponse]


class RecommendationSummaryResponse(BaseModel):
    """Resumo de recomendacoes."""
    config_id: int
    window_days: int
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    total: int
    total_active: int
    by_type: dict[str, int]
    by_action: dict[str, int]
    by_priority: dict[str, int]
    high_priority_count: int
    by_status: dict[str, int]
    applied_last_7_days: int
    dismissed_last_7_days: int
    last_generated_at: Optional[datetime] = None
    top_items: list[RecommendationResponse] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    """Request para gerar recomendações."""
    config_id: int = Field(..., description="ID da configuração FB Ads")
    entity_type: str = Field("campaign", description="Tipo de entidade (campaign, adset, ad)")
    entity_ids: Optional[list[str]] = Field(None, description="IDs específicos (opcional)")
    force_refresh: bool = Field(False, description="Forçar regeneração")


# Valid entity types
VALID_ENTITY_TYPES = {"campaign", "adset", "ad"}


class GenerateResponse(BaseModel):
    """Resposta de geração de recomendações."""
    config_id: int
    generated_count: int
    recommendations: list[RecommendationResponse]


class DismissRequest(BaseModel):
    """Request para descartar recomendação."""
    user_id: int = Field(..., description="ID do usuário")
    reason: Optional[str] = Field(None, description="Motivo do descarte")


class ApplyRequest(BaseModel):
    """Request para marcar recomendação como aplicada."""
    user_id: int = Field(..., description="ID do usuário")


# ==================== ENDPOINTS ====================

@router.get("", response_model=RecommendationListResponse)
async def list_recommendations(
    config_id: Optional[int] = Query(None, description="ID da configuração"),
    ad_account_id: Optional[str] = Query(None, description="Ad account ID"),
    entity_type: Optional[str] = Query(None, description="Tipo de entidade"),
    recommendation_type: Optional[str] = Query(None, description="Tipo de recomendação"),
    type: Optional[str] = Query(None, description="Alias para recommendation_type"),
    status_filter: Optional[str] = Query(
        None, alias="status", description="active|expired|dismissed|applied"
    ),
    campaign_id: Optional[str] = Query(None, description="Filtro por campanha"),
    priority_min: Optional[int] = Query(None, ge=1, le=10),
    is_active: Optional[bool] = Query(None),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    page: Optional[int] = Query(None, ge=1),
    page_size: Optional[int] = Query(None, ge=1, le=200),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    Lista recomendações ativas.

    - **config_id**: ID da configuração do Facebook Ads
    - **entity_type**: Filtrar por tipo de entidade (campaign, adset, ad)
    - **recommendation_type**: Filtrar por tipo de recomendação
    """
    ml_repo = MLRepository(db)

    if config_id is None and ad_account_id:
        from projects.ml.db.repositories.insights_repo import InsightsRepository
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

    rec_type = None
    rec_type_param = recommendation_type or type
    if rec_type_param:
        try:
            rec_type = RecommendationType(rec_type_param)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tipo de recomendação inválido: {rec_type_param}"
            )

    entity_id = campaign_id if campaign_id else None
    if status_filter and status_filter.lower() not in {"active", "expired", "dismissed", "applied"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Status inválido: {status_filter}",
        )

    if campaign_id and not entity_type:
        entity_type = "campaign"

    if page and page_size:
        offset = (page - 1) * page_size
        limit = page_size

    recommendations = await ml_repo.get_recommendations(
        config_id=config_id,
        entity_type=entity_type,
        entity_id=entity_id,
        recommendation_type=rec_type,
        status=status_filter,
        min_priority=priority_min,
        is_active=is_active,
        start_date=date_from,
        end_date=date_to,
        limit=limit,
        offset=offset,
    )

    total = await ml_repo.count_recommendations(
        config_id=config_id,
        entity_type=entity_type,
        entity_id=entity_id,
        recommendation_type=rec_type,
        status=status_filter,
        min_priority=priority_min,
        is_active=is_active,
        start_date=date_from,
        end_date=date_to,
    )

    # Contar por tipo
    by_type = {}
    for r in recommendations:
        type_val = r.recommendation_type.value
        by_type[type_val] = by_type.get(type_val, 0) + 1

    def _priority_level(priority: int) -> str:
        if priority >= 7:
            return "HIGH"
        if priority >= 4:
            return "MED"
        return "LOW"

    def _status(rec: MLRecommendation) -> str:
        now = datetime.utcnow()
        if rec.dismissed:
            return "dismissed"
        if rec.was_applied:
            return "applied"
        if rec.expires_at and rec.expires_at <= now:
            return "expired"
        return "active"

    data = [
        RecommendationResponse(
            id=r.id,
            config_id=r.config_id,
            entity_type=r.entity_type,
            entity_id=r.entity_id,
            recommendation_type=r.recommendation_type.value,
            action_type=r.recommendation_type.value,
            priority=r.priority,
            priority_level=_priority_level(r.priority),
            title=r.title,
            description=r.description,
            suggested_action=r.suggested_action,
            confidence_score=r.confidence_score,
            reasoning=r.reasoning,
            reason_code=(r.reasoning or {}).get("rule"),
            explain=r.reasoning,
            is_active=r.is_active,
            was_applied=r.was_applied,
            applied_at=r.applied_at,
            dismissed=r.dismissed,
            dismissed_reason=r.dismissed_reason,
            status=_status(r),
            created_at=r.created_at,
            expires_at=r.expires_at,
        )
        for r in recommendations
    ]

    page_value = page or 1
    page_size_value = page_size or limit

    return RecommendationListResponse(
        config_id=config_id,
        total=total,
        page=page_value,
        page_size=page_size_value,
        by_type=by_type,
        data=data,
        recommendations=data,
    )


@router.get("/campaign/{campaign_id}", response_model=RecommendationListResponse)
async def get_campaign_recommendations(
    campaign_id: str,
    config_id: int = Query(..., description="ID da configuração"),
    db: AsyncSession = Depends(get_db),
):
    """
    Obtém recomendações de uma campanha específica.
    """
    ml_repo = MLRepository(db)

    recommendations = await ml_repo.get_active_recommendations(
        config_id=config_id,
        entity_type="campaign",
        entity_id=campaign_id
    )

    by_type = {}
    for r in recommendations:
        type_val = r.recommendation_type.value
        by_type[type_val] = by_type.get(type_val, 0) + 1

    return RecommendationListResponse(
        config_id=config_id,
        total=len(recommendations),
        page=1,
        page_size=len(recommendations),
        by_type=by_type,
        data=[
            RecommendationResponse(
                id=r.id,
                config_id=r.config_id,
                entity_type=r.entity_type,
                entity_id=r.entity_id,
                recommendation_type=r.recommendation_type.value,
                action_type=r.recommendation_type.value,
                priority=r.priority,
                priority_level="HIGH" if r.priority >= 7 else "MED" if r.priority >= 4 else "LOW",
                title=r.title,
                description=r.description,
                suggested_action=r.suggested_action,
                confidence_score=r.confidence_score,
                reasoning=r.reasoning,
                reason_code=(r.reasoning or {}).get("rule"),
                explain=r.reasoning,
                is_active=r.is_active,
                was_applied=r.was_applied,
                applied_at=r.applied_at,
                dismissed=r.dismissed,
                dismissed_reason=r.dismissed_reason,
                status="dismissed" if r.dismissed else "applied" if r.was_applied else "expired" if r.expires_at and r.expires_at <= datetime.utcnow() else "active",
                created_at=r.created_at,
                expires_at=r.expires_at
            )
            for r in recommendations
        ],
        recommendations=[
            RecommendationResponse(
                id=r.id,
                config_id=r.config_id,
                entity_type=r.entity_type,
                entity_id=r.entity_id,
                recommendation_type=r.recommendation_type.value,
                action_type=r.recommendation_type.value,
                priority=r.priority,
                priority_level="HIGH" if r.priority >= 7 else "MED" if r.priority >= 4 else "LOW",
                title=r.title,
                description=r.description,
                suggested_action=r.suggested_action,
                confidence_score=r.confidence_score,
                reasoning=r.reasoning,
                reason_code=(r.reasoning or {}).get("rule"),
                explain=r.reasoning,
                is_active=r.is_active,
                was_applied=r.was_applied,
                applied_at=r.applied_at,
                dismissed=r.dismissed,
                dismissed_reason=r.dismissed_reason,
                status="dismissed" if r.dismissed else "applied" if r.was_applied else "expired" if r.expires_at and r.expires_at <= datetime.utcnow() else "active",
                created_at=r.created_at,
                expires_at=r.expires_at
            )
            for r in recommendations
        ],
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate_recommendations(
    request: GenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Gera novas recomendações para entidades.

    Analisa métricas atuais e gera sugestões de otimização
    baseadas em regras de negócio e modelos ML.
    Suporta múltiplos níveis: campaign, adset, ad.
    """
    entity_type = request.entity_type
    if entity_type not in VALID_ENTITY_TYPES:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"entity_type inválido: {entity_type}. Opções: {list(VALID_ENTITY_TYPES)}"
        )

    logger.info(
        "Solicitação de geração de recomendações",
        config_id=request.config_id,
        entity_type=entity_type,
        entity_ids=request.entity_ids,
        force_refresh=request.force_refresh
    )

    service = RecommendationService(db)
    recommendations = await service.generate_entity_recommendations(
        config_id=request.config_id,
        entity_type=entity_type,
        entity_ids=request.entity_ids,
        force_refresh=request.force_refresh,
    )

    return GenerateResponse(
        config_id=request.config_id,
        generated_count=len(recommendations),
        recommendations=[
            RecommendationResponse(
                id=r["id"],
                config_id=r["config_id"],
                entity_type=r["entity_type"],
                entity_id=r["entity_id"],
                recommendation_type=r["recommendation_type"],
                priority=r["priority"],
                title=r["title"],
                description=r["description"],
                suggested_action=r.get("suggested_action"),
                confidence_score=r["confidence_score"],
                reasoning=r.get("reasoning"),
                is_active=r.get("is_active", True),
                was_applied=r.get("was_applied", False),
                applied_at=datetime.fromisoformat(r["applied_at"])
                if r.get("applied_at")
                else None,
                dismissed=r.get("dismissed", False),
                dismissed_reason=r.get("dismissed_reason"),
                created_at=datetime.fromisoformat(r["created_at"])
                if r.get("created_at")
                else datetime.utcnow(),
                expires_at=datetime.fromisoformat(r["expires_at"])
                if r.get("expires_at")
                else None,
            )
            for r in recommendations
        ],
    )


@router.post("/{recommendation_id}/dismiss")
async def dismiss_recommendation(
    recommendation_id: int,
    request: DismissRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Descarta uma recomendação.

    Marca a recomendação como descartada e registra o motivo.
    """
    ml_repo = MLRepository(db)

    await ml_repo.dismiss_recommendation(
        recommendation_id=recommendation_id,
        user_id=request.user_id,
        reason=request.reason
    )

    logger.info(
        "Recomendação descartada",
        recommendation_id=recommendation_id,
        user_id=request.user_id,
        reason=request.reason
    )

    return {"status": "dismissed", "recommendation_id": recommendation_id}


@router.post("/{recommendation_id}/apply")
async def apply_recommendation(
    recommendation_id: int,
    request: ApplyRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Marca recomendação como aplicada.

    Registra que o usuário aplicou a sugestão manualmente.
    """
    ml_repo = MLRepository(db)

    await ml_repo.apply_recommendation(
        recommendation_id=recommendation_id,
        user_id=request.user_id
    )

    logger.info(
        "Recomendação aplicada",
        recommendation_id=recommendation_id,
        user_id=request.user_id
    )

    return {"status": "applied", "recommendation_id": recommendation_id}


@router.get("/summary", response_model=RecommendationSummaryResponse)
async def get_recommendation_summary(
    config_id: Optional[int] = Query(None, description="ID da configuração"),
    ad_account_id: Optional[str] = Query(None, description="Ad account ID"),
    window_days: int = Query(7, ge=1, le=30),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Resumo de recomendacoes para o dashboard."""
    if config_id is None and ad_account_id:
        from projects.ml.db.repositories.insights_repo import InsightsRepository
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
        window_start = date_from or (now - timedelta(days=window_days))
        window_end = date_to or now
        window_days = (window_end - window_start).days or window_days
    else:
        window_start = now - timedelta(days=window_days)
        window_end = now

    active_query = select(MLRecommendation).where(
        and_(
            MLRecommendation.config_id == config_id,
            MLRecommendation.dismissed == False,
            MLRecommendation.is_active == True,
            or_(
                MLRecommendation.expires_at.is_(None),
                MLRecommendation.expires_at > now,
            ),
        )
    )
    active_result = await db.execute(active_query)
    active = list(active_result.scalars().all())

    by_type = {}
    by_priority = {"high": 0, "medium": 0, "low": 0}
    by_status = {"active": 0, "expired": 0, "dismissed": 0, "applied": 0}
    for rec in active:
        type_val = rec.recommendation_type.value
        by_type[type_val] = by_type.get(type_val, 0) + 1
        if rec.priority >= 7:
            by_priority["high"] += 1
        elif rec.priority >= 4:
            by_priority["medium"] += 1
        else:
            by_priority["low"] += 1
        by_status["active"] += 1

    applied_query = select(func.count(MLRecommendation.id)).where(
        and_(
            MLRecommendation.config_id == config_id,
            MLRecommendation.was_applied == True,
            MLRecommendation.applied_at >= window_start,
            MLRecommendation.applied_at <= window_end,
        )
    )
    dismissed_query = select(func.count(MLRecommendation.id)).where(
        and_(
            MLRecommendation.config_id == config_id,
            MLRecommendation.dismissed == True,
            MLRecommendation.dismissed_at >= window_start,
            MLRecommendation.dismissed_at <= window_end,
        )
    )

    expired_query = select(func.count(MLRecommendation.id)).where(
        and_(
            MLRecommendation.config_id == config_id,
            MLRecommendation.dismissed == False,
            MLRecommendation.was_applied == False,
            MLRecommendation.expires_at.is_not(None),
            MLRecommendation.expires_at <= now,
            MLRecommendation.created_at >= window_start,
            MLRecommendation.created_at <= window_end,
        )
    )
    applied_count = int((await db.execute(applied_query)).scalar() or 0)
    dismissed_count = int((await db.execute(dismissed_query)).scalar() or 0)
    expired_count = int((await db.execute(expired_query)).scalar() or 0)
    by_status["applied"] = applied_count
    by_status["dismissed"] = dismissed_count
    by_status["expired"] = expired_count

    last_query = select(func.max(MLRecommendation.created_at)).where(
        MLRecommendation.config_id == config_id
    )
    last_result = await db.execute(last_query)
    last_generated_at = last_result.scalar()

    top_items = sorted(active, key=lambda x: x.priority, reverse=True)[:5]

    return RecommendationSummaryResponse(
        config_id=config_id,
        window_days=window_days,
        date_from=window_start,
        date_to=window_end,
        total=len(active),
        total_active=len(active),
        by_type=by_type,
        by_action=by_type,
        by_priority=by_priority,
        high_priority_count=by_priority["high"],
        by_status=by_status,
        applied_last_7_days=applied_count,
        dismissed_last_7_days=dismissed_count,
        last_generated_at=last_generated_at,
        top_items=[
            RecommendationResponse(
                id=r.id,
                config_id=r.config_id,
                entity_type=r.entity_type,
                entity_id=r.entity_id,
                recommendation_type=r.recommendation_type.value,
                action_type=r.recommendation_type.value,
                priority=r.priority,
                priority_level="HIGH" if r.priority >= 7 else "MED" if r.priority >= 4 else "LOW",
                title=r.title,
                description=r.description,
                suggested_action=r.suggested_action,
                confidence_score=r.confidence_score,
                reasoning=r.reasoning,
                reason_code=(r.reasoning or {}).get("rule"),
                explain=r.reasoning,
                is_active=r.is_active,
                was_applied=r.was_applied,
                applied_at=r.applied_at,
                dismissed=r.dismissed,
                dismissed_reason=r.dismissed_reason,
                status="dismissed" if r.dismissed else "applied" if r.was_applied else "expired" if r.expires_at and r.expires_at <= now else "active",
                created_at=r.created_at,
                expires_at=r.expires_at,
            )
            for r in top_items
        ],
    )
