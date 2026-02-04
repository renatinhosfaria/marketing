"""
Endpoints de classificação de entidades (campaigns, adsets, ads).
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from shared.db.session import get_db
from projects.ml.db.repositories.ml_repo import MLRepository
from projects.ml.db.models import (
    CampaignTier,
    MLClassification,
    MLCampaignClassification,  # Alias for backward compatibility
    ModelType,
    JobStatus,
)
from shared.core.logging import get_logger
from projects.ml.services.classification_service import ClassificationService

logger = get_logger(__name__)
router = APIRouter()

# Valid entity types
VALID_ENTITY_TYPES = {"campaign", "adset", "ad"}


def _label_from_tier(tier: str) -> str:
    if tier == "HIGH_PERFORMER":
        return "GOOD"
    if tier == "MODERATE":
        return "OK"
    return "BAD"


# ==================== SCHEMAS ====================

class ClassificationResponse(BaseModel):
    """Resposta de classificação de entidade."""
    id: int
    config_id: int
    entity_type: str = "campaign"
    entity_id: str
    parent_id: Optional[str] = None
    # Backward compatibility
    campaign_id: Optional[str] = None
    campaign_name: Optional[str] = None
    tier: str
    classification_label: Optional[str] = None
    confidence_score: float
    metrics_snapshot: Optional[dict] = None
    feature_importances: Optional[dict] = None
    model_version: Optional[str] = None
    previous_tier: Optional[str] = None
    tier_change_direction: Optional[str] = None
    classified_at: datetime
    valid_until: Optional[datetime] = None
    generated_at: Optional[datetime] = None
    window_days: Optional[int] = None


class ClassificationListResponse(BaseModel):
    """Lista de classificações."""
    config_id: int
    total: int
    page: int
    page_size: int
    by_tier: dict[str, int]
    data: list[ClassificationResponse]
    classifications: list[ClassificationResponse]


class ClassificationSummaryResponse(BaseModel):
    """Resumo de classificacoes."""
    config_id: int
    window_days: int
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    total_campaigns: int
    total: int
    by_tier: dict[str, int]
    by_label: dict[str, int]
    average_confidence: float
    recent_changes: int
    last_classification_at: Optional[datetime] = None
    last_classification: Optional[datetime] = None
    top_items: list[ClassificationResponse] = Field(default_factory=list)


class ClassifyRequest(BaseModel):
    """Request para classificar entidades."""
    config_id: int = Field(..., description="ID da configuração FB Ads")
    entity_type: str = Field("campaign", description="Tipo de entidade (campaign, adset, ad)")
    entity_ids: Optional[list[str]] = Field(
        None, description="IDs específicos ou None para todas"
    )
    # Backward compatibility
    campaign_ids: Optional[list[str]] = Field(
        None, description="Alias para entity_ids (deprecated)"
    )
    force_reclassify: bool = Field(
        False, description="Forcar reclassificacao mesmo com validade ativa"
    )


class ClassifyResponse(BaseModel):
    """Resposta de classificação em batch."""
    config_id: int
    classified_count: int
    classifications: list[ClassificationResponse]


class TrainRequest(BaseModel):
    """Request para treinar o classificador."""
    config_id: int = Field(..., description="ID da configuração FB Ads")
    min_samples: int = Field(30, ge=1, description="Mínimo de amostras")


class TrainResponse(BaseModel):
    """Resposta do treinamento do classificador."""
    success: bool
    model_id: Optional[int] = None
    metrics: Optional[dict] = None
    job_id: Optional[int] = None
    message: Optional[str] = None


# ==================== ENDPOINTS ====================

@router.get("/campaigns", response_model=ClassificationListResponse)
async def list_campaign_classifications(
    config_id: Optional[int] = Query(None, description="ID da configuração"),
    ad_account_id: Optional[str] = Query(None, description="Ad account ID"),
    tier: Optional[str] = Query(None, description="Filtrar por tier"),
    entity_type: str = Query("campaign", description="Tipo de entidade (campaign, adset, ad)"),
    entity_id: Optional[str] = Query(None, description="Filtrar por entidade específica"),
    campaign_id: Optional[str] = Query(None, description="Filtrar por campanha (alias para entity_id)"),
    min_confidence: Optional[float] = Query(None, ge=0, le=1),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    page: Optional[int] = Query(None, ge=1),
    page_size: Optional[int] = Query(None, ge=1, le=200),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    Lista classificações de entidades.

    - **config_id**: ID da configuração do Facebook Ads
    - **entity_type**: Tipo de entidade (campaign, adset, ad)
    - **entity_id**: ID da entidade específica (opcional)
    - **campaign_id**: Alias para entity_id (retrocompatibilidade)
    - **tier**: Filtrar por tier específico (HIGH_PERFORMER, MODERATE, LOW, UNDERPERFORMER)
    """
    ml_repo = MLRepository(db)

    # Validate entity_type
    if entity_type not in VALID_ENTITY_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"entity_type inválido: {entity_type}. Opções: {list(VALID_ENTITY_TYPES)}"
        )

    # Use campaign_id as alias for entity_id (backward compatibility)
    resolved_entity_id = entity_id or campaign_id

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

    tier_filter = None
    if tier:
        try:
            tier_filter = CampaignTier(tier)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tier inválido: {tier}. Opções: {[t.value for t in CampaignTier]}"
            )

    start_date = date_from
    end_date = date_to

    if page and page_size:
        offset = (page - 1) * page_size
        limit = page_size

    classifications = await ml_repo.get_classifications(
        config_id=config_id,
        entity_type=entity_type,
        entity_id=resolved_entity_id,
        tier=tier_filter,
        min_confidence=min_confidence,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )

    total = await ml_repo.count_classifications(
        config_id=config_id,
        entity_type=entity_type,
        entity_id=resolved_entity_id,
        tier=tier_filter,
        min_confidence=min_confidence,
        start_date=start_date,
        end_date=end_date,
    )

    by_tier = await ml_repo.get_classification_counts_by_tier(
        config_id=config_id,
        entity_type=entity_type,
        entity_id=resolved_entity_id,
        min_confidence=min_confidence,
        start_date=start_date,
        end_date=end_date,
    )
    for tier_key in [
        "HIGH_PERFORMER",
        "MODERATE",
        "LOW",
        "UNDERPERFORMER",
    ]:
        by_tier.setdefault(tier_key, 0)

    data = [
        ClassificationResponse(
            id=c.id,
            config_id=c.config_id,
            entity_type=c.entity_type,
            entity_id=c.entity_id,
            parent_id=c.parent_id,
            campaign_id=c.entity_id if c.entity_type == "campaign" else None,
            campaign_name=None,
            tier=c.tier.value,
            classification_label=_label_from_tier(c.tier.value),
            confidence_score=c.confidence_score,
            metrics_snapshot=c.metrics_snapshot,
            feature_importances=c.feature_importances,
            model_version=c.model_version,
            previous_tier=c.previous_tier.value if c.previous_tier else None,
            tier_change_direction=c.tier_change_direction,
            classified_at=c.classified_at,
            valid_until=c.valid_until,
            generated_at=c.classified_at,
            window_days=7,
        )
        for c in classifications
    ]

    page_value = page or 1
    page_size_value = page_size or limit

    return ClassificationListResponse(
        config_id=config_id,
        total=total,
        page=page_value,
        page_size=page_size_value,
        by_tier=by_tier,
        data=data,
        classifications=data,
    )


@router.get("", response_model=ClassificationListResponse)
async def list_classifications(
    config_id: Optional[int] = Query(None, description="ID da configuração"),
    ad_account_id: Optional[str] = Query(None, description="Ad account ID"),
    tier: Optional[str] = Query(None, description="Filtrar por tier"),
    entity_type: str = Query("campaign", description="Tipo de entidade (campaign, adset, ad)"),
    entity_id: Optional[str] = Query(None, description="Filtrar por entidade específica"),
    campaign_id: Optional[str] = Query(None, description="Filtrar por campanha (alias para entity_id)"),
    min_confidence: Optional[float] = Query(None, ge=0, le=1),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    page: Optional[int] = Query(None, ge=1),
    page_size: Optional[int] = Query(None, ge=1, le=200),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    Lista classificações de entidades (campaign, adset, ad).
    """
    return await list_campaign_classifications(
        config_id=config_id,
        ad_account_id=ad_account_id,
        tier=tier,
        entity_type=entity_type,
        entity_id=entity_id,
        campaign_id=campaign_id,
        min_confidence=min_confidence,
        date_from=date_from,
        date_to=date_to,
        page=page,
        page_size=page_size,
        limit=limit,
        offset=offset,
        db=db,
    )


@router.get("/campaigns/{campaign_id}", response_model=ClassificationResponse)
async def get_campaign_classification(
    campaign_id: str,
    config_id: int = Query(..., description="ID da configuração"),
    db: AsyncSession = Depends(get_db),
):
    """
    Obtém classificação de uma campanha específica.
    """
    ml_repo = MLRepository(db)

    classification = await ml_repo.get_latest_classification(
        config_id=config_id,
        campaign_id=campaign_id
    )

    if not classification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Classificação não encontrada para campanha {campaign_id}"
        )

    return ClassificationResponse(
        id=classification.id,
        campaign_id=classification.campaign_id,
        tier=classification.tier.value,
        confidence_score=classification.confidence_score,
        metrics_snapshot=classification.metrics_snapshot,
        feature_importances=classification.feature_importances,
        previous_tier=classification.previous_tier.value if classification.previous_tier else None,
        tier_change_direction=classification.tier_change_direction,
        classified_at=classification.classified_at,
        valid_until=classification.valid_until
    )


@router.post("/classify", response_model=ClassifyResponse)
async def classify_entities(
    request: ClassifyRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Classifica entidades por performance.

    Executa o classificador ML nas entidades especificadas
    ou em todas as entidades ativas da configuração.
    Suporta múltiplos níveis: campaign, adset, ad.
    """
    entity_type = request.entity_type
    if entity_type not in VALID_ENTITY_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"entity_type inválido: {entity_type}. Opções: {list(VALID_ENTITY_TYPES)}"
        )

    # Backward compatibility: use campaign_ids if entity_ids not provided
    entity_ids = request.entity_ids or request.campaign_ids

    logger.info(
        "Solicitação de classificação de entidades",
        config_id=request.config_id,
        entity_type=entity_type,
        entity_ids=entity_ids
    )

    service = ClassificationService(db)
    classifications = await service.classify_entities(
        config_id=request.config_id,
        entity_type=entity_type,
        entity_ids=entity_ids,
        force_reclassify=request.force_reclassify,
    )

    return ClassifyResponse(
        config_id=request.config_id,
        classified_count=len(classifications),
        classifications=[
            ClassificationResponse(
                id=c["id"],
                config_id=c["config_id"],
                entity_type=c.get("entity_type", "campaign"),
                entity_id=c.get("entity_id", c.get("campaign_id", "")),
                parent_id=c.get("parent_id"),
                campaign_id=c.get("campaign_id"),
                campaign_name=None,
                tier=c["tier"],
                classification_label=_label_from_tier(c["tier"]),
                confidence_score=c["confidence_score"],
                metrics_snapshot=c.get("metrics_snapshot"),
                feature_importances=c.get("feature_importances"),
                model_version=c.get("model_version"),
                previous_tier=c.get("previous_tier"),
                tier_change_direction=c.get("tier_change_direction"),
                classified_at=datetime.fromisoformat(c["classified_at"])
                if c.get("classified_at")
                else datetime.utcnow(),
                valid_until=datetime.fromisoformat(c["valid_until"])
                if c.get("valid_until")
                else None,
                generated_at=datetime.fromisoformat(c["classified_at"])
                if c.get("classified_at")
                else None,
                window_days=7,
            )
            for c in classifications
        ],
    )


@router.post("/campaigns/classify", response_model=ClassifyResponse)
async def classify_campaigns(
    request: ClassifyRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Classifica campanhas por performance.
    Endpoint de compatibilidade - delega para /classify.
    """
    # Force entity_type to campaign for this endpoint
    request.entity_type = "campaign"
    return await classify_entities(request, db)


@router.post("/train", response_model=TrainResponse)
async def train_classifier(
    request: TrainRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Treina o classificador de campanhas.
    """
    ml_repo = MLRepository(db)
    job = await ml_repo.create_training_job(
        model_type=ModelType.CAMPAIGN_CLASSIFIER,
        config_id=request.config_id,
    )
    await ml_repo.update_job_status(
        job.id,
        JobStatus.RUNNING,
        progress=0.0,
    )
    await db.commit()

    try:
        service = ClassificationService(db)
        result = await service.train_classifier(
            config_id=request.config_id,
            min_samples=request.min_samples,
        )
        if not result:
            await ml_repo.update_job_status(
                job.id,
                JobStatus.FAILED,
                error_message="Dados insuficientes para treinamento",
            )
            await db.commit()
            return TrainResponse(
                success=False,
                job_id=job.id,
                message="Dados insuficientes para treinamento.",
            )

        await ml_repo.update_job_status(
            job.id,
            JobStatus.COMPLETED,
            progress=1.0,
            model_id=result.get("model_id"),
        )
        await db.commit()

        return TrainResponse(
            success=True,
            model_id=result.get("model_id"),
            metrics=result.get("metrics"),
            job_id=job.id,
            message="Treinamento concluído.",
        )
    except Exception as exc:
        await ml_repo.update_job_status(
            job.id,
            JobStatus.FAILED,
            error_message=str(exc),
        )
        await db.commit()
        raise


@router.get("/summary", response_model=ClassificationSummaryResponse)
async def get_classification_summary(
    config_id: Optional[int] = Query(None, description="ID da configuração"),
    ad_account_id: Optional[str] = Query(None, description="Ad account ID"),
    entity_type: str = Query("campaign", description="Tipo de entidade (campaign, adset, ad)"),
    window_days: int = Query(7, ge=1, le=30),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Resumo das classificacoes para o dashboard.

    - **entity_type**: Tipo de entidade (campaign, adset, ad)
    """
    # Validate entity_type
    if entity_type not in VALID_ENTITY_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"entity_type inválido: {entity_type}. Opções: {list(VALID_ENTITY_TYPES)}"
        )

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

    # Use MLClassification with entity_type filter
    subquery = (
        select(
            MLClassification.entity_id,
            func.max(MLClassification.classified_at).label("max_classified_at"),
        )
        .where(
            and_(
                MLClassification.config_id == config_id,
                MLClassification.entity_type == entity_type,
                MLClassification.classified_at >= window_start,
                MLClassification.classified_at <= window_end,
            )
        )
        .group_by(MLClassification.entity_id)
        .subquery()
    )

    latest_query = (
        select(MLClassification)
        .join(
            subquery,
            and_(
                MLClassification.entity_id == subquery.c.entity_id,
                MLClassification.classified_at == subquery.c.max_classified_at,
            ),
        )
        .where(
            and_(
                MLClassification.config_id == config_id,
                MLClassification.entity_type == entity_type,
            )
        )
    )

    result = await db.execute(latest_query)
    latest = list(result.scalars().all())

    by_tier = {
        "HIGH_PERFORMER": 0,
        "MODERATE": 0,
        "LOW": 0,
        "UNDERPERFORMER": 0,
    }
    by_label = {"GOOD": 0, "OK": 0, "BAD": 0}
    for item in latest:
        by_tier[item.tier.value] = by_tier.get(item.tier.value, 0) + 1
        label = _label_from_tier(item.tier.value)
        by_label[label] = by_label.get(label, 0) + 1

    total_entities = len(latest)
    avg_confidence = (
        sum(item.confidence_score for item in latest) / total_entities
        if total_entities
        else 0.0
    )

    changes_query = select(func.count(MLClassification.id)).where(
        and_(
            MLClassification.config_id == config_id,
            MLClassification.entity_type == entity_type,
            MLClassification.classified_at >= window_start,
            MLClassification.classified_at <= window_end,
            MLClassification.tier_change_direction != "stable",
        )
    )
    changes_result = await db.execute(changes_query)
    recent_changes = int(changes_result.scalar() or 0)

    last_query = select(func.max(MLClassification.classified_at)).where(
        and_(
            MLClassification.config_id == config_id,
            MLClassification.entity_type == entity_type,
        )
    )
    last_result = await db.execute(last_query)
    last_classification_at = last_result.scalar()

    top_items = sorted(
        latest, key=lambda x: x.confidence_score, reverse=True
    )[:5]

    return ClassificationSummaryResponse(
        config_id=config_id,
        window_days=window_days,
        date_from=window_start,
        date_to=window_end,
        total_campaigns=total_entities,  # Kept for backward compatibility
        total=total_entities,
        by_tier=by_tier,
        by_label=by_label,
        average_confidence=avg_confidence,
        recent_changes=recent_changes,
        last_classification_at=last_classification_at,
        last_classification=last_classification_at,
        top_items=[
            ClassificationResponse(
                id=c.id,
                config_id=c.config_id,
                entity_type=c.entity_type,
                entity_id=c.entity_id,
                parent_id=c.parent_id,
                campaign_id=c.entity_id if c.entity_type == "campaign" else None,
                campaign_name=None,
                tier=c.tier.value,
                classification_label=_label_from_tier(c.tier.value),
                confidence_score=c.confidence_score,
                metrics_snapshot=c.metrics_snapshot,
                feature_importances=c.feature_importances,
                model_version=getattr(c, 'model_version', 'v1.0'),
                previous_tier=c.previous_tier.value if c.previous_tier else None,
                tier_change_direction=c.tier_change_direction,
                classified_at=c.classified_at,
                valid_until=c.valid_until,
                generated_at=c.classified_at,
                window_days=7,
            )
            for c in top_items
        ],
    )
