"""Endpoints de campanhas, ad sets e ads do Facebook Ads com métricas agregadas."""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.session import get_db
from shared.core.logging import get_logger
from shared.db.models.famachat_readonly import (
    SistemaFacebookAdsCampaigns,
    SistemaFacebookAdsAdsets,
    SistemaFacebookAdsAds,
    SistemaFacebookAdsInsightsHistory,
    SistemaFacebookAdsInsightsToday,
)
from projects.facebook_ads.schemas.campaigns import (
    CampaignResponse,
    AdSetResponse,
    AdResponse,
)
from projects.facebook_ads.utils.metrics_calculator import (
    calculate_ctr, calculate_cpc, calculate_cpm, calculate_cpl,
)
from projects.facebook_ads.utils.date_helpers import get_today_sao_paulo
from projects.facebook_ads.schemas.base import camel_keys

logger = get_logger(__name__)
router = APIRouter()


# Presets que devem incluir dados de hoje
PRESETS_WITH_TODAY = {"today", "this_month", "this_year", "last_7d", "last_14d", "last_30d", "last_90d"}
PRESETS_ONLY_TODAY = {"today"}


def _parse_date_params(date_preset: Optional[str]) -> tuple[datetime, datetime]:
    """Parse parâmetros de data, retorna (since, until) como datetime."""
    today = get_today_sao_paulo()

    presets = {
        "today": (today, today),
        "yesterday": (today - timedelta(days=1), today - timedelta(days=1)),
        "last_7d": (today - timedelta(days=7), today),
        "last_14d": (today - timedelta(days=14), today),
        "last_30d": (today - timedelta(days=30), today),
        "last_90d": (today - timedelta(days=90), today),
        "this_month": (today.replace(day=1), today),
        "last_month": ((today.replace(day=1) - timedelta(days=1)).replace(day=1), today.replace(day=1) - timedelta(days=1)),
        "this_year": (today.replace(month=1, day=1), today),
        "last_year": (today.replace(year=today.year - 1, month=1, day=1), today.replace(year=today.year - 1, month=12, day=31)),
    }

    since, until = presets.get(date_preset or "last_30d", presets["last_30d"])
    return datetime.combine(since, datetime.min.time()), datetime.combine(until, datetime.max.time())


def _should_use_today_table(date_preset: Optional[str]) -> bool:
    """Determina se deve incluir dados da tabela today."""
    return date_preset in PRESETS_WITH_TODAY


def _should_use_only_today_table(date_preset: Optional[str]) -> bool:
    """Determina se deve usar APENAS a tabela today."""
    return date_preset in PRESETS_ONLY_TODAY


@router.get("/campaigns")
async def list_campaigns(
    config_id: int = Query(..., alias="configId"),
    status_filter: Optional[str] = Query(None, alias="status"),
    date_preset: Optional[str] = Query("last_30d", alias="datePreset"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Lista campanhas com métricas agregadas do período."""
    since, until = _parse_date_params(date_preset)
    use_only_today = _should_use_only_today_table(date_preset)
    use_today = _should_use_today_table(date_preset)

    # Buscar campanhas
    query = select(SistemaFacebookAdsCampaigns).where(
        SistemaFacebookAdsCampaigns.config_id == config_id
    )

    if status_filter:
        query = query.where(SistemaFacebookAdsCampaigns.status == status_filter)

    # Total
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar_one()

    # Data
    result = await db.execute(
        query.order_by(desc(SistemaFacebookAdsCampaigns.synced_at))
        .limit(limit)
        .offset(offset)
    )
    campaigns = result.scalars().all()
    campaign_ids = [c.campaign_id for c in campaigns]

    # Agregar métricas por campanha
    metrics_data = {}

    # Buscar dados da tabela history (exceto se for only_today)
    if not use_only_today and campaign_ids:
        history_result = await db.execute(
            select(
                SistemaFacebookAdsInsightsHistory.campaign_id,
                func.sum(SistemaFacebookAdsInsightsHistory.spend).label("spend"),
                func.sum(SistemaFacebookAdsInsightsHistory.impressions).label("impressions"),
                func.sum(SistemaFacebookAdsInsightsHistory.clicks).label("clicks"),
                func.sum(SistemaFacebookAdsInsightsHistory.leads).label("leads"),
            ).where(
                and_(
                    SistemaFacebookAdsInsightsHistory.config_id == config_id,
                    SistemaFacebookAdsInsightsHistory.campaign_id.in_(campaign_ids),
                    SistemaFacebookAdsInsightsHistory.date >= since,
                    SistemaFacebookAdsInsightsHistory.date <= until,
                )
            ).group_by(SistemaFacebookAdsInsightsHistory.campaign_id)
        )
        for row in history_result.all():
            metrics_data[row.campaign_id] = {
                "spend": Decimal(str(row.spend or 0)),
                "impressions": int(row.impressions or 0),
                "clicks": int(row.clicks or 0),
                "leads": int(row.leads or 0),
            }

    # Adicionar dados da tabela today se necessário
    if use_today and campaign_ids:
        today_result = await db.execute(
            select(
                SistemaFacebookAdsInsightsToday.campaign_id,
                func.sum(SistemaFacebookAdsInsightsToday.spend).label("spend"),
                func.sum(SistemaFacebookAdsInsightsToday.impressions).label("impressions"),
                func.sum(SistemaFacebookAdsInsightsToday.clicks).label("clicks"),
                func.sum(SistemaFacebookAdsInsightsToday.leads).label("leads"),
            ).where(
                and_(
                    SistemaFacebookAdsInsightsToday.config_id == config_id,
                    SistemaFacebookAdsInsightsToday.campaign_id.in_(campaign_ids),
                )
            ).group_by(SistemaFacebookAdsInsightsToday.campaign_id)
        )
        for row in today_result.all():
            if row.campaign_id in metrics_data:
                metrics_data[row.campaign_id]["spend"] += Decimal(str(row.spend or 0))
                metrics_data[row.campaign_id]["impressions"] += int(row.impressions or 0)
                metrics_data[row.campaign_id]["clicks"] += int(row.clicks or 0)
                metrics_data[row.campaign_id]["leads"] += int(row.leads or 0)
            else:
                metrics_data[row.campaign_id] = {
                    "spend": Decimal(str(row.spend or 0)),
                    "impressions": int(row.impressions or 0),
                    "clicks": int(row.clicks or 0),
                    "leads": int(row.leads or 0),
                }

    # Construir resposta com métricas
    data = []
    for c in campaigns:
        metrics = metrics_data.get(c.campaign_id, {"spend": Decimal("0"), "impressions": 0, "clicks": 0, "leads": 0})
        spend = metrics["spend"]
        impressions = metrics["impressions"]
        clicks = metrics["clicks"]
        leads = metrics["leads"]

        campaign_data = {
            "id": c.id,
            "configId": c.config_id,
            "campaignId": c.campaign_id,
            "name": c.name,
            "objective": c.objective,
            "status": c.status,
            "effectiveStatus": c.effective_status,
            "dailyBudget": float(c.daily_budget) if c.daily_budget else None,
            "lifetimeBudget": float(c.lifetime_budget) if c.lifetime_budget else None,
            "budgetRemaining": float(c.budget_remaining) if c.budget_remaining else None,
            "startTime": c.start_time.isoformat() if c.start_time else None,
            "stopTime": c.stop_time.isoformat() if c.stop_time else None,
            "syncedAt": c.synced_at.isoformat() if c.synced_at else None,
            # Métricas
            "spend": float(spend),
            "impressions": impressions,
            "clicks": clicks,
            "leads": leads,
            "ctr": float(calculate_ctr(clicks, impressions) or 0),
            "cpc": float(calculate_cpc(spend, clicks) or 0),
            "cpm": float(calculate_cpm(spend, impressions) or 0),
            "cpl": float(calculate_cpl(spend, leads)) if leads > 0 else None,
        }
        data.append(campaign_data)

    return {
        "success": True,
        "data": data,
        "total": total,
        "pagination": {"limit": limit, "offset": offset, "hasMore": offset + limit < total},
    }


@router.get("/campaigns/{campaign_id}")
async def get_campaign(
    campaign_id: str,
    config_id: int = Query(..., alias="configId"),
    db: AsyncSession = Depends(get_db),
):
    """Busca campanha por ID."""
    result = await db.execute(
        select(SistemaFacebookAdsCampaigns).where(
            SistemaFacebookAdsCampaigns.config_id == config_id,
            SistemaFacebookAdsCampaigns.campaign_id == campaign_id,
        )
    )
    campaign = result.scalar_one_or_none()

    if not campaign:
        raise HTTPException(status_code=404, detail="Campanha não encontrada")

    return {"success": True, "data": CampaignResponse.model_validate(campaign).model_dump(by_alias=True)}


@router.get("/campaigns/{campaign_id}/adsets")
async def list_campaign_adsets(
    campaign_id: str,
    config_id: int = Query(..., alias="configId"),
    db: AsyncSession = Depends(get_db),
):
    """Lista ad sets de uma campanha."""
    result = await db.execute(
        select(SistemaFacebookAdsAdsets).where(
            SistemaFacebookAdsAdsets.config_id == config_id,
            SistemaFacebookAdsAdsets.campaign_id == campaign_id,
        )
    )
    adsets = result.scalars().all()

    return {
        "success": True,
        "data": [AdSetResponse.model_validate(a).model_dump(by_alias=True) for a in adsets],
        "total": len(adsets),
    }


@router.get("/campaigns/{campaign_id}/ads")
async def list_campaign_ads(
    campaign_id: str,
    config_id: int = Query(..., alias="configId"),
    db: AsyncSession = Depends(get_db),
):
    """Lista anúncios de uma campanha."""
    result = await db.execute(
        select(SistemaFacebookAdsAds).where(
            SistemaFacebookAdsAds.config_id == config_id,
            SistemaFacebookAdsAds.campaign_id == campaign_id,
        )
    )
    ads = result.scalars().all()

    return {
        "success": True,
        "data": [AdResponse.model_validate(a).model_dump(by_alias=True) for a in ads],
        "total": len(ads),
    }


@router.get("/adsets")
async def list_adsets(
    config_id: int = Query(..., alias="configId"),
    campaign_id: Optional[str] = Query(None, alias="campaignId"),
    status_filter: Optional[str] = Query(None, alias="status"),
    date_preset: Optional[str] = Query("last_30d", alias="datePreset"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Lista todos os ad sets com métricas agregadas."""
    since, until = _parse_date_params(date_preset)
    use_only_today = _should_use_only_today_table(date_preset)
    use_today = _should_use_today_table(date_preset)

    query = select(SistemaFacebookAdsAdsets).where(
        SistemaFacebookAdsAdsets.config_id == config_id
    )
    if campaign_id:
        query = query.where(SistemaFacebookAdsAdsets.campaign_id == campaign_id)
    if status_filter:
        query = query.where(SistemaFacebookAdsAdsets.status == status_filter)

    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar_one()

    result = await db.execute(query.limit(limit).offset(offset))
    adsets = result.scalars().all()
    adset_ids = [a.adset_id for a in adsets]

    # Agregar métricas por adset
    metrics_data = {}

    if not use_only_today and adset_ids:
        history_result = await db.execute(
            select(
                SistemaFacebookAdsInsightsHistory.adset_id,
                func.sum(SistemaFacebookAdsInsightsHistory.spend).label("spend"),
                func.sum(SistemaFacebookAdsInsightsHistory.impressions).label("impressions"),
                func.sum(SistemaFacebookAdsInsightsHistory.clicks).label("clicks"),
                func.sum(SistemaFacebookAdsInsightsHistory.leads).label("leads"),
            ).where(
                and_(
                    SistemaFacebookAdsInsightsHistory.config_id == config_id,
                    SistemaFacebookAdsInsightsHistory.adset_id.in_(adset_ids),
                    SistemaFacebookAdsInsightsHistory.date >= since,
                    SistemaFacebookAdsInsightsHistory.date <= until,
                )
            ).group_by(SistemaFacebookAdsInsightsHistory.adset_id)
        )
        for row in history_result.all():
            metrics_data[row.adset_id] = {
                "spend": Decimal(str(row.spend or 0)),
                "impressions": int(row.impressions or 0),
                "clicks": int(row.clicks or 0),
                "leads": int(row.leads or 0),
            }

    if use_today and adset_ids:
        today_result = await db.execute(
            select(
                SistemaFacebookAdsInsightsToday.adset_id,
                func.sum(SistemaFacebookAdsInsightsToday.spend).label("spend"),
                func.sum(SistemaFacebookAdsInsightsToday.impressions).label("impressions"),
                func.sum(SistemaFacebookAdsInsightsToday.clicks).label("clicks"),
                func.sum(SistemaFacebookAdsInsightsToday.leads).label("leads"),
            ).where(
                and_(
                    SistemaFacebookAdsInsightsToday.config_id == config_id,
                    SistemaFacebookAdsInsightsToday.adset_id.in_(adset_ids),
                )
            ).group_by(SistemaFacebookAdsInsightsToday.adset_id)
        )
        for row in today_result.all():
            if row.adset_id in metrics_data:
                metrics_data[row.adset_id]["spend"] += Decimal(str(row.spend or 0))
                metrics_data[row.adset_id]["impressions"] += int(row.impressions or 0)
                metrics_data[row.adset_id]["clicks"] += int(row.clicks or 0)
                metrics_data[row.adset_id]["leads"] += int(row.leads or 0)
            else:
                metrics_data[row.adset_id] = {
                    "spend": Decimal(str(row.spend or 0)),
                    "impressions": int(row.impressions or 0),
                    "clicks": int(row.clicks or 0),
                    "leads": int(row.leads or 0),
                }

    # Construir resposta com métricas
    data = []
    for a in adsets:
        metrics = metrics_data.get(a.adset_id, {"spend": Decimal("0"), "impressions": 0, "clicks": 0, "leads": 0})
        spend = metrics["spend"]
        impressions = metrics["impressions"]
        clicks = metrics["clicks"]
        leads = metrics["leads"]

        adset_data = {
            "id": a.id,
            "configId": a.config_id,
            "campaignId": a.campaign_id,
            "adsetId": a.adset_id,
            "name": a.name,
            "status": a.status,
            "effectiveStatus": a.effective_status,
            "dailyBudget": float(a.daily_budget) if a.daily_budget else None,
            "lifetimeBudget": float(a.lifetime_budget) if a.lifetime_budget else None,
            "optimizationGoal": a.optimization_goal,
            "syncedAt": a.synced_at.isoformat() if a.synced_at else None,
            # Métricas
            "spend": float(spend),
            "impressions": impressions,
            "clicks": clicks,
            "leads": leads,
            "ctr": float(calculate_ctr(clicks, impressions) or 0),
            "cpc": float(calculate_cpc(spend, clicks) or 0),
            "cpm": float(calculate_cpm(spend, impressions) or 0),
            "cpl": float(calculate_cpl(spend, leads)) if leads > 0 else None,
        }
        data.append(adset_data)

    return {
        "success": True,
        "data": data,
        "total": total,
        "pagination": {"limit": limit, "offset": offset, "hasMore": offset + limit < total},
    }


@router.get("/ads")
async def list_ads(
    config_id: int = Query(..., alias="configId"),
    campaign_id: Optional[str] = Query(None, alias="campaignId"),
    adset_id: Optional[str] = Query(None, alias="adsetId"),
    status_filter: Optional[str] = Query(None, alias="status"),
    date_preset: Optional[str] = Query("last_30d", alias="datePreset"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Lista todos os anúncios com métricas agregadas."""
    since, until = _parse_date_params(date_preset)
    use_only_today = _should_use_only_today_table(date_preset)
    use_today = _should_use_today_table(date_preset)

    query = select(SistemaFacebookAdsAds).where(
        SistemaFacebookAdsAds.config_id == config_id
    )
    if campaign_id:
        query = query.where(SistemaFacebookAdsAds.campaign_id == campaign_id)
    if adset_id:
        query = query.where(SistemaFacebookAdsAds.adset_id == adset_id)
    if status_filter:
        query = query.where(SistemaFacebookAdsAds.status == status_filter)

    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar_one()

    result = await db.execute(query.limit(limit).offset(offset))
    ads = result.scalars().all()
    ad_ids = [ad.ad_id for ad in ads]

    # Agregar métricas por ad
    metrics_data = {}

    if not use_only_today and ad_ids:
        history_result = await db.execute(
            select(
                SistemaFacebookAdsInsightsHistory.ad_id,
                func.sum(SistemaFacebookAdsInsightsHistory.spend).label("spend"),
                func.sum(SistemaFacebookAdsInsightsHistory.impressions).label("impressions"),
                func.sum(SistemaFacebookAdsInsightsHistory.clicks).label("clicks"),
                func.sum(SistemaFacebookAdsInsightsHistory.leads).label("leads"),
            ).where(
                and_(
                    SistemaFacebookAdsInsightsHistory.config_id == config_id,
                    SistemaFacebookAdsInsightsHistory.ad_id.in_(ad_ids),
                    SistemaFacebookAdsInsightsHistory.date >= since,
                    SistemaFacebookAdsInsightsHistory.date <= until,
                )
            ).group_by(SistemaFacebookAdsInsightsHistory.ad_id)
        )
        for row in history_result.all():
            metrics_data[row.ad_id] = {
                "spend": Decimal(str(row.spend or 0)),
                "impressions": int(row.impressions or 0),
                "clicks": int(row.clicks or 0),
                "leads": int(row.leads or 0),
            }

    if use_today and ad_ids:
        today_result = await db.execute(
            select(
                SistemaFacebookAdsInsightsToday.ad_id,
                func.sum(SistemaFacebookAdsInsightsToday.spend).label("spend"),
                func.sum(SistemaFacebookAdsInsightsToday.impressions).label("impressions"),
                func.sum(SistemaFacebookAdsInsightsToday.clicks).label("clicks"),
                func.sum(SistemaFacebookAdsInsightsToday.leads).label("leads"),
            ).where(
                and_(
                    SistemaFacebookAdsInsightsToday.config_id == config_id,
                    SistemaFacebookAdsInsightsToday.ad_id.in_(ad_ids),
                )
            ).group_by(SistemaFacebookAdsInsightsToday.ad_id)
        )
        for row in today_result.all():
            if row.ad_id in metrics_data:
                metrics_data[row.ad_id]["spend"] += Decimal(str(row.spend or 0))
                metrics_data[row.ad_id]["impressions"] += int(row.impressions or 0)
                metrics_data[row.ad_id]["clicks"] += int(row.clicks or 0)
                metrics_data[row.ad_id]["leads"] += int(row.leads or 0)
            else:
                metrics_data[row.ad_id] = {
                    "spend": Decimal(str(row.spend or 0)),
                    "impressions": int(row.impressions or 0),
                    "clicks": int(row.clicks or 0),
                    "leads": int(row.leads or 0),
                }

    # Construir resposta com métricas
    data = []
    for ad in ads:
        metrics = metrics_data.get(ad.ad_id, {"spend": Decimal("0"), "impressions": 0, "clicks": 0, "leads": 0})
        spend = metrics["spend"]
        impressions = metrics["impressions"]
        clicks = metrics["clicks"]
        leads = metrics["leads"]

        ad_data = {
            "id": ad.id,
            "configId": ad.config_id,
            "campaignId": ad.campaign_id,
            "adsetId": ad.adset_id,
            "adId": ad.ad_id,
            "name": ad.name,
            "status": ad.status,
            "effectiveStatus": ad.effective_status,
            "previewShareableLink": ad.preview_shareable_link,
            "syncedAt": ad.synced_at.isoformat() if ad.synced_at else None,
            # Métricas
            "spend": float(spend),
            "impressions": impressions,
            "clicks": clicks,
            "leads": leads,
            "ctr": float(calculate_ctr(clicks, impressions) or 0),
            "cpc": float(calculate_cpc(spend, clicks) or 0),
            "cpm": float(calculate_cpm(spend, impressions) or 0),
            "cpl": float(calculate_cpl(spend, leads)) if leads > 0 else None,
        }
        data.append(ad_data)

    return {
        "success": True,
        "data": data,
        "total": total,
        "pagination": {"limit": limit, "offset": offset, "hasMore": offset + limit < total},
    }


@router.get("/adsets/{adset_id}")
async def get_adset(
    adset_id: str,
    config_id: int = Query(..., alias="configId"),
    db: AsyncSession = Depends(get_db),
):
    """Busca ad set por ID."""
    result = await db.execute(
        select(SistemaFacebookAdsAdsets).where(
            SistemaFacebookAdsAdsets.config_id == config_id,
            SistemaFacebookAdsAdsets.adset_id == adset_id,
        )
    )
    adset = result.scalar_one_or_none()

    if not adset:
        raise HTTPException(status_code=404, detail="Ad Set não encontrado")

    return {"success": True, "data": AdSetResponse.model_validate(adset).model_dump(by_alias=True)}


@router.get("/ads/{ad_id}")
async def get_ad(
    ad_id: str,
    config_id: int = Query(..., alias="configId"),
    db: AsyncSession = Depends(get_db),
):
    """Busca anúncio por ID."""
    result = await db.execute(
        select(SistemaFacebookAdsAds).where(
            SistemaFacebookAdsAds.config_id == config_id,
            SistemaFacebookAdsAds.ad_id == ad_id,
        )
    )
    ad = result.scalar_one_or_none()

    if not ad:
        raise HTTPException(status_code=404, detail="Anúncio não encontrado")

    return {"success": True, "data": AdResponse.model_validate(ad).model_dump(by_alias=True)}
