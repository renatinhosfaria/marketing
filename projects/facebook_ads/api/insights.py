"""Endpoints de insights e KPIs do Facebook Ads."""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, and_, desc, case
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.session import get_db
from shared.core.logging import get_logger
from shared.db.models.famachat_readonly import (
    SistemaFacebookAdsInsightsHistory,
    SistemaFacebookAdsInsightsToday,
    SistemaFacebookAdsCampaigns,
    SistemaFacebookAdsInsightsBreakdowns,
)
from projects.facebook_ads.utils.metrics_calculator import (
    calculate_ctr, calculate_cpc, calculate_cpm, calculate_cpl, calculate_cpp, safe_divide,
)
from projects.facebook_ads.utils.date_helpers import get_today_sao_paulo
from projects.facebook_ads.schemas.base import camel_keys

logger = get_logger(__name__)
router = APIRouter()


# Presets que devem incluir dados de hoje (tabela today)
PRESETS_WITH_TODAY = {"today", "this_month", "this_year"}
# Presets que usam apenas dados de hoje
PRESETS_ONLY_TODAY = {"today"}


def _should_use_today_table(date_preset: Optional[str], date_from: Optional[str], date_to: Optional[str]) -> bool:
    """Determina se deve incluir dados da tabela today."""
    if date_preset in PRESETS_WITH_TODAY:
        return True
    # Se for range customizado que inclui hoje
    if date_from and date_to:
        today = get_today_sao_paulo()
        until = datetime.strptime(date_to, "%Y-%m-%d").date()
        return until >= today
    return False


def _should_use_only_today_table(date_preset: Optional[str]) -> bool:
    """Determina se deve usar APENAS a tabela today."""
    return date_preset in PRESETS_ONLY_TODAY


def _parse_date_params(
    date_from: Optional[str],
    date_to: Optional[str],
    date_preset: Optional[str],
) -> tuple[datetime, datetime]:
    """Parse parâmetros de data, retorna (since, until) como datetime."""
    today = get_today_sao_paulo()

    if date_from and date_to:
        return (
            datetime.strptime(date_from, "%Y-%m-%d"),
            datetime.strptime(date_to, "%Y-%m-%d"),
        )

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


@router.get("/kpis")
async def get_kpis(
    config_id: int = Query(..., alias="configId"),
    date_from: Optional[str] = Query(None, alias="dateFrom"),
    date_to: Optional[str] = Query(None, alias="dateTo"),
    date_preset: Optional[str] = Query("last_30d", alias="datePreset"),
    db: AsyncSession = Depends(get_db),
):
    """KPIs agregados com comparação ao período anterior."""
    since, until = _parse_date_params(date_from, date_to, date_preset)
    
    use_only_today = _should_use_only_today_table(date_preset)
    use_today = _should_use_today_table(date_preset, date_from, date_to)

    # Inicializar métricas
    spend = Decimal("0")
    impressions = 0
    reach = 0
    clicks = 0
    leads = 0
    conversions = 0

    # Buscar dados da tabela history (exceto se for only_today)
    if not use_only_today:
        result = await db.execute(
            select(
                func.coalesce(func.sum(SistemaFacebookAdsInsightsHistory.spend), 0).label("spend"),
                func.coalesce(func.sum(SistemaFacebookAdsInsightsHistory.impressions), 0).label("impressions"),
                func.coalesce(func.sum(SistemaFacebookAdsInsightsHistory.reach), 0).label("reach"),
                func.coalesce(func.sum(SistemaFacebookAdsInsightsHistory.clicks), 0).label("clicks"),
                func.coalesce(func.sum(SistemaFacebookAdsInsightsHistory.leads), 0).label("leads"),
                func.coalesce(func.sum(SistemaFacebookAdsInsightsHistory.conversions), 0).label("conversions"),
            ).where(
                and_(
                    SistemaFacebookAdsInsightsHistory.config_id == config_id,
                    SistemaFacebookAdsInsightsHistory.date >= since,
                    SistemaFacebookAdsInsightsHistory.date <= until,
                )
            )
        )
        row = result.one()
        spend = Decimal(str(row.spend))
        impressions = int(row.impressions)
        clicks = int(row.clicks)
        leads = int(row.leads)
        reach = int(row.reach)
        conversions = int(row.conversions)

    # Adicionar dados da tabela today se necessário
    if use_today:
        today_result = await db.execute(
            select(
                func.coalesce(func.sum(SistemaFacebookAdsInsightsToday.spend), 0).label("spend"),
                func.coalesce(func.sum(SistemaFacebookAdsInsightsToday.impressions), 0).label("impressions"),
                func.coalesce(func.sum(SistemaFacebookAdsInsightsToday.reach), 0).label("reach"),
                func.coalesce(func.sum(SistemaFacebookAdsInsightsToday.clicks), 0).label("clicks"),
                func.coalesce(func.sum(SistemaFacebookAdsInsightsToday.leads), 0).label("leads"),
                func.coalesce(func.sum(SistemaFacebookAdsInsightsToday.conversions), 0).label("conversions"),
            ).where(
                SistemaFacebookAdsInsightsToday.config_id == config_id,
            )
        )
        today_row = today_result.one()
        spend += Decimal(str(today_row.spend))
        impressions += int(today_row.impressions)
        clicks += int(today_row.clicks)
        leads += int(today_row.leads)
        reach += int(today_row.reach)
        conversions += int(today_row.conversions)

    current_metrics = {
        "spend": float(spend),
        "impressions": impressions,
        "reach": reach,
        "clicks": clicks,
        "leads": leads,
        "conversions": conversions,
        "ctr": float(calculate_ctr(clicks, impressions) or 0),
        "cpc": float(calculate_cpc(spend, clicks) or 0),
        "cpm": float(calculate_cpm(spend, impressions) or 0),
        "cpl": float(calculate_cpl(spend, leads) or 0),
        "frequency": float(safe_divide(impressions, reach) or 0),
        "period": date_preset or f"{date_from}_{date_to}",
    }

    # Período anterior (mesma duração, deslocado para trás)
    period_duration = until - since
    prev_until = since - timedelta(days=1)
    prev_since = prev_until - period_duration

    prev_result = await db.execute(
        select(
            func.coalesce(func.sum(SistemaFacebookAdsInsightsHistory.spend), 0).label("spend"),
            func.coalesce(func.sum(SistemaFacebookAdsInsightsHistory.impressions), 0).label("impressions"),
            func.coalesce(func.sum(SistemaFacebookAdsInsightsHistory.reach), 0).label("reach"),
            func.coalesce(func.sum(SistemaFacebookAdsInsightsHistory.clicks), 0).label("clicks"),
            func.coalesce(func.sum(SistemaFacebookAdsInsightsHistory.leads), 0).label("leads"),
            func.coalesce(func.sum(SistemaFacebookAdsInsightsHistory.conversions), 0).label("conversions"),
        ).where(
            and_(
                SistemaFacebookAdsInsightsHistory.config_id == config_id,
                SistemaFacebookAdsInsightsHistory.date >= prev_since,
                SistemaFacebookAdsInsightsHistory.date <= prev_until,
            )
        )
    )
    prev_row = prev_result.one()

    prev_spend = Decimal(str(prev_row.spend))
    prev_impressions = int(prev_row.impressions)
    prev_clicks = int(prev_row.clicks)
    prev_leads = int(prev_row.leads)
    prev_reach = int(prev_row.reach)
    prev_conversions = int(prev_row.conversions)

    prev_metrics = {
        "spend": float(prev_spend),
        "impressions": prev_impressions,
        "reach": prev_reach,
        "clicks": prev_clicks,
        "leads": prev_leads,
        "conversions": prev_conversions,
        "ctr": float(calculate_ctr(prev_clicks, prev_impressions) or 0),
        "cpc": float(calculate_cpc(prev_spend, prev_clicks) or 0),
        "cpm": float(calculate_cpm(prev_spend, prev_impressions) or 0),
        "cpl": float(calculate_cpl(prev_spend, prev_leads) or 0),
    }

    # Calcular variação percentual
    comparison = {}
    compare_keys = ["spend", "impressions", "reach", "clicks", "leads", "conversions", "ctr", "cpc", "cpm", "cpl"]
    for key in compare_keys:
        prev_val = prev_metrics.get(key, 0)
        curr_val = current_metrics.get(key, 0)
        if prev_val and prev_val != 0:
            comparison[key] = round((curr_val - prev_val) / prev_val * 100, 2)
        else:
            comparison[key] = None

    return {
        "success": True,
        "data": {
            "metrics": current_metrics,
            "comparison": comparison,
        },
    }


@router.get("/daily")
async def get_daily_insights(
    config_id: int = Query(..., alias="configId"),
    date_from: Optional[str] = Query(None, alias="dateFrom"),
    date_to: Optional[str] = Query(None, alias="dateTo"),
    date_preset: Optional[str] = Query("last_30d", alias="datePreset"),
    db: AsyncSession = Depends(get_db),
):
    """Série temporal diária de insights."""
    since, until = _parse_date_params(date_from, date_to, date_preset)
    
    use_only_today = _should_use_only_today_table(date_preset)
    use_today = _should_use_today_table(date_preset, date_from, date_to)

    data = []
    
    # Buscar dados da tabela history (exceto se for only_today)
    if not use_only_today:
        result = await db.execute(
            select(
                func.date(SistemaFacebookAdsInsightsHistory.date).label("day"),
                func.sum(SistemaFacebookAdsInsightsHistory.spend).label("spend"),
                func.sum(SistemaFacebookAdsInsightsHistory.impressions).label("impressions"),
                func.sum(SistemaFacebookAdsInsightsHistory.reach).label("reach"),
                func.sum(SistemaFacebookAdsInsightsHistory.clicks).label("clicks"),
                func.sum(SistemaFacebookAdsInsightsHistory.leads).label("leads"),
                func.sum(SistemaFacebookAdsInsightsHistory.conversions).label("conversions"),
            ).where(
                and_(
                    SistemaFacebookAdsInsightsHistory.config_id == config_id,
                    SistemaFacebookAdsInsightsHistory.date >= since,
                    SistemaFacebookAdsInsightsHistory.date <= until,
                )
            ).group_by(
                func.date(SistemaFacebookAdsInsightsHistory.date)
            ).order_by(
                func.date(SistemaFacebookAdsInsightsHistory.date)
            )
        )
        rows = result.all()

        for row in rows:
            spend = Decimal(str(row.spend or 0))
            clicks = int(row.clicks or 0)
            impressions = int(row.impressions or 0)
            leads = int(row.leads or 0)

            data.append({
                "date": str(row.day),
                "spend": float(spend),
                "impressions": impressions,
                "reach": int(row.reach or 0),
                "clicks": clicks,
                "leads": leads,
                "conversions": int(row.conversions or 0),
                "ctr": float(calculate_ctr(clicks, impressions) or 0),
                "cpc": float(calculate_cpc(spend, clicks) or 0),
                "cpl": float(calculate_cpl(spend, leads) or 0),
            })

    # Adicionar dados de hoje se necessário
    if use_today:
        today_result = await db.execute(
            select(
                func.date(SistemaFacebookAdsInsightsToday.date).label("day"),
                func.sum(SistemaFacebookAdsInsightsToday.spend).label("spend"),
                func.sum(SistemaFacebookAdsInsightsToday.impressions).label("impressions"),
                func.sum(SistemaFacebookAdsInsightsToday.reach).label("reach"),
                func.sum(SistemaFacebookAdsInsightsToday.clicks).label("clicks"),
                func.sum(SistemaFacebookAdsInsightsToday.leads).label("leads"),
                func.sum(SistemaFacebookAdsInsightsToday.conversions).label("conversions"),
            ).where(
                SistemaFacebookAdsInsightsToday.config_id == config_id,
            ).group_by(
                func.date(SistemaFacebookAdsInsightsToday.date)
            )
        )
        today_rows = today_result.all()

        for row in today_rows:
            spend = Decimal(str(row.spend or 0))
            clicks = int(row.clicks or 0)
            impressions = int(row.impressions or 0)
            leads = int(row.leads or 0)

            data.append({
                "date": str(row.day),
                "spend": float(spend),
                "impressions": impressions,
                "reach": int(row.reach or 0),
                "clicks": clicks,
                "leads": leads,
                "conversions": int(row.conversions or 0),
                "ctr": float(calculate_ctr(clicks, impressions) or 0),
                "cpc": float(calculate_cpc(spend, clicks) or 0),
                "cpl": float(calculate_cpl(spend, leads) or 0),
            })

    # Ordenar por data
    data.sort(key=lambda x: x["date"])

    return {
        "success": True,
        "data": data,
        "meta": camel_keys({"period": date_preset, "total_days": len(data)}),
    }


@router.get("/campaigns")
async def get_campaign_insights(
    config_id: int = Query(..., alias="configId"),
    date_from: Optional[str] = Query(None, alias="dateFrom"),
    date_to: Optional[str] = Query(None, alias="dateTo"),
    date_preset: Optional[str] = Query("last_30d", alias="datePreset"),
    sort_by: str = Query("spend", alias="sortBy"),
    sort_order: str = Query("desc", alias="sortOrder"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Insights agregados por campanha."""
    since, until = _parse_date_params(date_from, date_to, date_preset)
    
    use_only_today = _should_use_only_today_table(date_preset)
    use_today = _should_use_today_table(date_preset, date_from, date_to)
    
    # Dicionário para agregar dados por campanha
    campaign_data = {}

    # Buscar dados da tabela history (exceto se for only_today)
    if not use_only_today:
        result = await db.execute(
            select(
                SistemaFacebookAdsInsightsHistory.campaign_id,
                func.sum(SistemaFacebookAdsInsightsHistory.spend).label("spend"),
                func.sum(SistemaFacebookAdsInsightsHistory.impressions).label("impressions"),
                func.sum(SistemaFacebookAdsInsightsHistory.reach).label("reach"),
                func.sum(SistemaFacebookAdsInsightsHistory.clicks).label("clicks"),
                func.sum(SistemaFacebookAdsInsightsHistory.leads).label("leads"),
                func.sum(SistemaFacebookAdsInsightsHistory.conversions).label("conversions"),
            ).where(
                and_(
                    SistemaFacebookAdsInsightsHistory.config_id == config_id,
                    SistemaFacebookAdsInsightsHistory.date >= since,
                    SistemaFacebookAdsInsightsHistory.date <= until,
                )
            ).group_by(
                SistemaFacebookAdsInsightsHistory.campaign_id
            )
        )
        for row in result.all():
            campaign_data[row.campaign_id] = {
                "spend": Decimal(str(row.spend or 0)),
                "impressions": int(row.impressions or 0),
                "reach": int(row.reach or 0),
                "clicks": int(row.clicks or 0),
                "leads": int(row.leads or 0),
                "conversions": int(row.conversions or 0),
            }

    # Adicionar dados da tabela today se necessário
    if use_today:
        today_result = await db.execute(
            select(
                SistemaFacebookAdsInsightsToday.campaign_id,
                func.sum(SistemaFacebookAdsInsightsToday.spend).label("spend"),
                func.sum(SistemaFacebookAdsInsightsToday.impressions).label("impressions"),
                func.sum(SistemaFacebookAdsInsightsToday.reach).label("reach"),
                func.sum(SistemaFacebookAdsInsightsToday.clicks).label("clicks"),
                func.sum(SistemaFacebookAdsInsightsToday.leads).label("leads"),
                func.sum(SistemaFacebookAdsInsightsToday.conversions).label("conversions"),
            ).where(
                SistemaFacebookAdsInsightsToday.config_id == config_id,
            ).group_by(
                SistemaFacebookAdsInsightsToday.campaign_id
            )
        )
        for row in today_result.all():
            if row.campaign_id in campaign_data:
                campaign_data[row.campaign_id]["spend"] += Decimal(str(row.spend or 0))
                campaign_data[row.campaign_id]["impressions"] += int(row.impressions or 0)
                campaign_data[row.campaign_id]["reach"] += int(row.reach or 0)
                campaign_data[row.campaign_id]["clicks"] += int(row.clicks or 0)
                campaign_data[row.campaign_id]["leads"] += int(row.leads or 0)
                campaign_data[row.campaign_id]["conversions"] += int(row.conversions or 0)
            else:
                campaign_data[row.campaign_id] = {
                    "spend": Decimal(str(row.spend or 0)),
                    "impressions": int(row.impressions or 0),
                    "reach": int(row.reach or 0),
                    "clicks": int(row.clicks or 0),
                    "leads": int(row.leads or 0),
                    "conversions": int(row.conversions or 0),
                }

    # Buscar nomes das campanhas
    campaign_ids = list(campaign_data.keys())
    campaigns_result = await db.execute(
        select(SistemaFacebookAdsCampaigns).where(
            SistemaFacebookAdsCampaigns.config_id == config_id,
            SistemaFacebookAdsCampaigns.campaign_id.in_(campaign_ids),
        )
    )
    campaign_map = {c.campaign_id: c for c in campaigns_result.scalars().all()}

    data = []
    for campaign_id, metrics in campaign_data.items():
        camp = campaign_map.get(campaign_id)
        spend = metrics["spend"]
        clicks = metrics["clicks"]
        impressions = metrics["impressions"]
        leads = metrics["leads"]

        data.append(camel_keys({
            "campaign_id": campaign_id,
            "campaign_name": camp.name if camp else "Desconhecida",
            "objective": camp.objective if camp else None,
            "status": camp.effective_status or camp.status if camp else None,
            "spend": float(spend),
            "impressions": impressions,
            "reach": metrics["reach"],
            "clicks": clicks,
            "leads": leads,
            "conversions": metrics["conversions"],
            "ctr": float(calculate_ctr(clicks, impressions) or 0),
            "cpc": float(calculate_cpc(spend, clicks) or 0),
            "cpl": float(calculate_cpl(spend, leads) or 0),
        }))

    # Ordenar por spend (desc) e limitar
    data.sort(key=lambda x: x.get("spend", 0), reverse=True)
    data = data[:limit]

    return {"success": True, "data": data, "meta": camel_keys({"period": date_preset, "total": len(data)})}


@router.get("/rankings")
async def get_campaign_rankings(
    config_id: int = Query(..., alias="configId"),
    date_preset: str = Query("last_30d", alias="datePreset"),
    metric: str = Query("cpl"),
    top_n: int = Query(5, ge=1, le=20, alias="topN"),
    db: AsyncSession = Depends(get_db),
):
    """Top/bottom campanhas por métrica."""
    since, until = _parse_date_params(None, None, date_preset)

    result = await db.execute(
        select(
            SistemaFacebookAdsInsightsHistory.campaign_id,
            func.sum(SistemaFacebookAdsInsightsHistory.spend).label("spend"),
            func.sum(SistemaFacebookAdsInsightsHistory.impressions).label("impressions"),
            func.sum(SistemaFacebookAdsInsightsHistory.clicks).label("clicks"),
            func.sum(SistemaFacebookAdsInsightsHistory.leads).label("leads"),
        ).where(
            and_(
                SistemaFacebookAdsInsightsHistory.config_id == config_id,
                SistemaFacebookAdsInsightsHistory.date >= since,
                SistemaFacebookAdsInsightsHistory.date <= until,
            )
        ).group_by(
            SistemaFacebookAdsInsightsHistory.campaign_id
        ).having(
            func.sum(SistemaFacebookAdsInsightsHistory.spend) > 0
        )
    )
    rows = result.all()

    # Calcular métricas
    campaign_ids = [r.campaign_id for r in rows]
    campaigns_result = await db.execute(
        select(SistemaFacebookAdsCampaigns).where(
            SistemaFacebookAdsCampaigns.config_id == config_id,
            SistemaFacebookAdsCampaigns.campaign_id.in_(campaign_ids),
        )
    )
    campaign_map = {c.campaign_id: c for c in campaigns_result.scalars().all()}

    enriched = []
    for row in rows:
        camp = campaign_map.get(row.campaign_id)
        spend = float(row.spend or 0)
        clicks = int(row.clicks or 0)
        leads = int(row.leads or 0)
        impressions = int(row.impressions or 0)

        cpl = spend / leads if leads > 0 else float("inf")
        cpc = spend / clicks if clicks > 0 else float("inf")
        ctr = (clicks / impressions * 100) if impressions > 0 else 0

        enriched.append(camel_keys({
            "campaign_id": row.campaign_id,
            "campaign_name": camp.name if camp else "Desconhecida",
            "spend": spend,
            "leads": leads,
            "clicks": clicks,
            "impressions": impressions,
            "cpl": cpl if cpl != float("inf") else None,
            "cpc": cpc if cpc != float("inf") else None,
            "ctr": ctr,
        }))

    # Ordenar
    metric_key = metric if metric in ("spend", "leads", "clicks", "cpl", "cpc", "ctr") else "cpl"
    ascending = metric_key in ("cpl", "cpc")  # Menor é melhor

    sorted_data = sorted(
        [e for e in enriched if e.get(metric_key) is not None],
        key=lambda x: x.get(metric_key, float("inf")),
        reverse=not ascending,
    )

    return {
        "success": True,
        "data": {
            "top": sorted_data[:top_n],
            "bottom": sorted_data[-top_n:] if len(sorted_data) > top_n else [],
        },
        "meta": camel_keys({"metric": metric_key, "period": date_preset}),
    }


@router.get("/compare")
async def compare_periods(
    config_id: int = Query(..., alias="configId"),
    current_from: str = Query(..., alias="currentFrom"),
    current_to: str = Query(..., alias="currentTo"),
    previous_from: str = Query(..., alias="previousFrom"),
    previous_to: str = Query(..., alias="previousTo"),
    db: AsyncSession = Depends(get_db),
):
    """Compara dois períodos."""
    async def _get_period_kpis(since_str: str, until_str: str) -> dict:
        since = datetime.strptime(since_str, "%Y-%m-%d")
        until = datetime.strptime(until_str, "%Y-%m-%d")

        result = await db.execute(
            select(
                func.coalesce(func.sum(SistemaFacebookAdsInsightsHistory.spend), 0).label("spend"),
                func.coalesce(func.sum(SistemaFacebookAdsInsightsHistory.impressions), 0).label("impressions"),
                func.coalesce(func.sum(SistemaFacebookAdsInsightsHistory.clicks), 0).label("clicks"),
                func.coalesce(func.sum(SistemaFacebookAdsInsightsHistory.leads), 0).label("leads"),
                func.coalesce(func.sum(SistemaFacebookAdsInsightsHistory.reach), 0).label("reach"),
            ).where(
                and_(
                    SistemaFacebookAdsInsightsHistory.config_id == config_id,
                    SistemaFacebookAdsInsightsHistory.date >= since,
                    SistemaFacebookAdsInsightsHistory.date <= until,
                )
            )
        )
        row = result.one()
        spend = float(row.spend or 0)
        impressions = int(row.impressions or 0)
        clicks = int(row.clicks or 0)
        leads = int(row.leads or 0)

        return {
            "spend": spend,
            "impressions": impressions,
            "clicks": clicks,
            "leads": leads,
            "reach": int(row.reach or 0),
            "cpl": spend / leads if leads > 0 else 0,
            "cpc": spend / clicks if clicks > 0 else 0,
            "ctr": clicks / impressions * 100 if impressions > 0 else 0,
        }

    current_kpis = await _get_period_kpis(current_from, current_to)
    previous_kpis = await _get_period_kpis(previous_from, previous_to)

    # Calcular variações percentuais
    changes = {}
    for key in ["spend", "impressions", "clicks", "leads", "cpl", "cpc", "ctr"]:
        prev = previous_kpis.get(key, 0)
        curr = current_kpis.get(key, 0)
        if prev and prev > 0:
            changes[key] = round((curr - prev) / prev * 100, 2)
        else:
            changes[key] = None

    return {
        "success": True,
        "data": {
            "current": current_kpis,
            "previous": previous_kpis,
            "changes": changes,
        },
    }


@router.get("/quality-diagnostics")
async def get_quality_diagnostics(
    config_id: int = Query(..., alias="configId"),
    date_from: Optional[str] = Query(None, alias="dateFrom"),
    date_to: Optional[str] = Query(None, alias="dateTo"),
    date_preset: Optional[str] = Query("last_30d", alias="datePreset"),
    db: AsyncSession = Depends(get_db),
):
    """Diagnósticos de qualidade dos anúncios (rankings)."""
    since, until = _parse_date_params(date_from, date_to, date_preset)
    I = SistemaFacebookAdsInsightsHistory

    result = await db.execute(
        select(
            I.ad_id,
            I.quality_ranking,
            I.engagement_rate_ranking,
            I.conversion_rate_ranking,
            func.sum(I.impressions).label("impressions"),
            func.sum(I.spend).label("spend"),
        ).where(
            and_(
                I.config_id == config_id,
                I.date >= since,
                I.date <= until,
                I.quality_ranking.isnot(None),
            )
        ).group_by(
            I.ad_id, I.quality_ranking, I.engagement_rate_ranking, I.conversion_rate_ranking,
        ).order_by(desc(func.sum(I.spend)))
    )
    rows = result.all()

    data = [
        camel_keys({
            "ad_id": r.ad_id,
            "quality_ranking": r.quality_ranking,
            "engagement_rate_ranking": r.engagement_rate_ranking,
            "conversion_rate_ranking": r.conversion_rate_ranking,
            "impressions": int(r.impressions or 0),
            "spend": float(r.spend or 0),
        })
        for r in rows
    ]
    return {"success": True, "data": data}


@router.get("/video-funnel")
async def get_video_funnel(
    config_id: int = Query(..., alias="configId"),
    date_from: Optional[str] = Query(None, alias="dateFrom"),
    date_to: Optional[str] = Query(None, alias="dateTo"),
    date_preset: Optional[str] = Query("last_30d", alias="datePreset"),
    campaign_id: Optional[str] = Query(None, alias="campaignId"),
    db: AsyncSession = Depends(get_db),
):
    """Funil completo de métricas de vídeo."""
    since, until = _parse_date_params(date_from, date_to, date_preset)
    I = SistemaFacebookAdsInsightsHistory

    filters = [I.config_id == config_id, I.date >= since, I.date <= until]
    if campaign_id:
        filters.append(I.campaign_id == campaign_id)

    result = await db.execute(
        select(
            func.coalesce(func.sum(I.video_plays), 0).label("video_plays"),
            func.coalesce(func.sum(I.video_15s_watched), 0).label("video_15s_watched"),
            func.coalesce(func.sum(I.video_p25_watched), 0).label("video_p25_watched"),
            func.coalesce(func.sum(I.video_p50_watched), 0).label("video_p50_watched"),
            func.coalesce(func.sum(I.video_p75_watched), 0).label("video_p75_watched"),
            func.coalesce(func.sum(I.video_p95_watched), 0).label("video_p95_watched"),
            func.coalesce(func.sum(I.video_views), 0).label("video_30s_watched"),
            func.coalesce(func.sum(I.video_p100_watched), 0).label("video_p100_watched"),
            func.coalesce(func.sum(I.video_thruplay), 0).label("video_thruplay"),
            func.avg(I.video_avg_time).label("video_avg_time"),
        ).where(and_(*filters))
    )
    row = result.one()

    data = camel_keys({
        "video_plays": int(row.video_plays),
        "video_15s_watched": int(row.video_15s_watched),
        "video_p25_watched": int(row.video_p25_watched),
        "video_p50_watched": int(row.video_p50_watched),
        "video_p75_watched": int(row.video_p75_watched),
        "video_p95_watched": int(row.video_p95_watched),
        "video_30s_watched": int(row.video_30s_watched),
        "video_p100_watched": int(row.video_p100_watched),
        "video_thruplay": int(row.video_thruplay),
        "video_avg_time": float(row.video_avg_time or 0),
    })
    return {"success": True, "data": data}


@router.get("/breakdowns")
async def get_breakdown_insights(
    config_id: int = Query(..., alias="configId"),
    breakdown_type: str = Query(..., alias="breakdownType"),
    date_from: Optional[str] = Query(None, alias="dateFrom"),
    date_to: Optional[str] = Query(None, alias="dateTo"),
    date_preset: Optional[str] = Query("last_30d", alias="datePreset"),
    campaign_id: Optional[str] = Query(None, alias="campaignId"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Insights agregados por breakdown avançado."""
    since, until = _parse_date_params(date_from, date_to, date_preset)
    B = SistemaFacebookAdsInsightsBreakdowns

    filters = [
        B.config_id == config_id,
        B.date >= since,
        B.date <= until,
        B.breakdown_type == breakdown_type,
    ]
    if campaign_id:
        filters.append(B.campaign_id == campaign_id)

    result = await db.execute(
        select(
            B.breakdown_value,
            func.sum(B.spend).label("spend"),
            func.sum(B.impressions).label("impressions"),
            func.sum(B.reach).label("reach"),
            func.sum(B.clicks).label("clicks"),
            func.sum(B.leads).label("leads"),
            func.sum(B.conversions).label("conversions"),
        ).where(and_(*filters))
        .group_by(B.breakdown_value)
        .order_by(desc(func.sum(B.spend)))
        .limit(limit)
    )
    rows = result.all()

    data = []
    for row in rows:
        spend = Decimal(str(row.spend or 0))
        clicks = int(row.clicks or 0)
        impressions = int(row.impressions or 0)
        leads = int(row.leads or 0)

        data.append(camel_keys({
            "breakdown_type": breakdown_type,
            "breakdown_value": row.breakdown_value,
            "spend": float(spend),
            "impressions": impressions,
            "reach": int(row.reach or 0),
            "clicks": clicks,
            "leads": leads,
            "conversions": int(row.conversions or 0),
            "ctr": float(calculate_ctr(clicks, impressions) or 0),
            "cpc": float(calculate_cpc(spend, clicks) or 0),
            "cpl": float(calculate_cpl(spend, leads) or 0),
        }))

    return {
        "success": True,
        "data": data,
        "meta": camel_keys({"breakdown_type": breakdown_type, "period": date_preset, "total": len(data)}),
    }


@router.get("/summary")
async def get_insights_summary(
    config_id: int = Query(..., alias="configId"),
    date_from: Optional[str] = Query(None, alias="dateFrom"),
    date_to: Optional[str] = Query(None, alias="dateTo"),
    date_preset: Optional[str] = Query("last_30d", alias="datePreset"),
    db: AsyncSession = Depends(get_db),
):
    """Alias para /kpis — KPIs com comparação."""
    return await get_kpis(
        config_id=config_id,
        date_from=date_from,
        date_to=date_to,
        date_preset=date_preset,
        db=db,
    )


@router.get("/")
async def get_insights_root(
    config_id: int = Query(..., alias="configId"),
    date_from: Optional[str] = Query(None, alias="dateFrom"),
    date_to: Optional[str] = Query(None, alias="dateTo"),
    date_preset: Optional[str] = Query("last_30d", alias="datePreset"),
    db: AsyncSession = Depends(get_db),
):
    """Alias para /daily — série temporal."""
    return await get_daily_insights(
        config_id=config_id,
        date_from=date_from,
        date_to=date_to,
        date_preset=date_preset,
        db=db,
    )
