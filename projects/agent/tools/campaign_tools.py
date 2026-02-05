"""
Tools para acesso a detalhes de campanhas.
"""

from typing import Optional, List
from langchain_core.tools import tool
from sqlalchemy import select, desc, func
from datetime import datetime, timedelta

from shared.db.session import get_db_session
from shared.db.models.famachat_readonly import (
    FacebookAdsCampaign,
    FacebookAdsInsight,
)
from shared.core.logging import get_logger
from projects.agent.tools.base import format_currency, format_percentage, format_number

logger = get_logger(__name__)


@tool
def get_campaign_details(
    config_id: int,
    campaign_id: str
) -> dict:
    """
    Retorna detalhes completos de uma campanha específica.

    Use esta ferramenta para obter todas as informações de uma campanha,
    incluindo configurações, métricas e performance histórica.

    Args:
        config_id: ID da configuração Facebook Ads
        campaign_id: ID da campanha

    Returns:
        Detalhes completos da campanha com métricas
    """
    try:
        with get_db_session() as db:
            # Buscar campanha
            campaign = db.execute(
                select(FacebookAdsCampaign).where(
                    FacebookAdsCampaign.config_id == config_id,
                    FacebookAdsCampaign.campaign_id == campaign_id
                )
            ).scalar_one_or_none()

            if not campaign:
                return {
                    "found": False,
                    "message": f"Campanha {campaign_id} não encontrada."
                }

            # Buscar insights dos últimos 7 dias
            since_7d = datetime.utcnow() - timedelta(days=7)
            insights_7d = db.execute(
                select(FacebookAdsInsight).where(
                    FacebookAdsInsight.campaign_id == campaign_id,
                    FacebookAdsInsight.date >= since_7d
                ).order_by(desc(FacebookAdsInsight.date))
            ).scalars().all()

            # Buscar insights dos últimos 30 dias
            since_30d = datetime.utcnow() - timedelta(days=30)
            insights_30d = db.execute(
                select(FacebookAdsInsight).where(
                    FacebookAdsInsight.campaign_id == campaign_id,
                    FacebookAdsInsight.date >= since_30d
                )
            ).scalars().all()

            # Calcular métricas de 7 dias
            metrics_7d = _calculate_metrics(insights_7d)
            # Calcular métricas de 30 dias
            metrics_30d = _calculate_metrics(insights_30d)

            return {
                "found": True,
                "campaign": {
                    "id": campaign.campaign_id,
                    "name": campaign.name,
                    "status": campaign.status,
                    "objective": campaign.objective,
                    "daily_budget": format_currency(float(campaign.daily_budget)) if campaign.daily_budget else "N/A",
                    "lifetime_budget": format_currency(float(campaign.lifetime_budget)) if campaign.lifetime_budget else "N/A",
                    "created_at": campaign.created_time.isoformat() if campaign.created_time else None,
                },
                "metrics_7d": metrics_7d,
                "metrics_30d": metrics_30d,
                "trend": _calculate_trend(metrics_7d, metrics_30d),
                "summary": _build_campaign_summary(campaign.name, metrics_7d)
            }
    except Exception as e:
        logger.error("Erro ao buscar detalhes da campanha", error=str(e), campaign_id=campaign_id)
        return {"error": str(e), "found": False}


@tool
def list_campaigns(
    config_id: int,
    status: Optional[str] = None,
    limit: int = 50
) -> dict:
    """
    Lista todas as campanhas da conta.

    Use esta ferramenta para obter uma visão geral de todas as campanhas,
    filtrando opcionalmente por status.

    Args:
        config_id: ID da configuração Facebook Ads
        status: Filtrar por status (ACTIVE, PAUSED, etc.) ou None para todos
        limit: Número máximo de campanhas (padrão: 50)

    Returns:
        Lista de campanhas com métricas básicas
    """
    try:
        with get_db_session() as db:
            query = select(FacebookAdsCampaign).where(
                FacebookAdsCampaign.config_id == config_id
            )

            if status:
                query = query.where(FacebookAdsCampaign.status == status.upper())

            query = query.order_by(desc(FacebookAdsCampaign.synced_at)).limit(limit)

            campaigns = db.execute(query).scalars().all()

            # Contar por status
            status_counts = {}
            campaign_list = []

            for c in campaigns:
                camp_status = c.status or "UNKNOWN"
                status_counts[camp_status] = status_counts.get(camp_status, 0) + 1

                campaign_list.append({
                    "id": c.campaign_id,
                    "name": c.name,
                    "status": camp_status,
                    "objective": c.objective,
                    "daily_budget": format_currency(float(c.daily_budget)) if c.daily_budget else "N/A",
                })

            active_count = status_counts.get("ACTIVE", 0)
            paused_count = status_counts.get("PAUSED", 0)

            return {
                "total": len(campaign_list),
                "by_status": status_counts,
                "campaigns": campaign_list,
                "summary": f"Total de {len(campaign_list)} campanhas: {active_count} ativas, {paused_count} pausadas."
            }
    except Exception as e:
        logger.error("Erro ao listar campanhas", error=str(e))
        return {"error": str(e), "campaigns": []}


def _calculate_metrics(insights: list) -> dict:
    """Calcula métricas agregadas dos insights."""
    if not insights:
        return {
            "spend": "R$ 0,00",
            "leads": 0,
            "cpl": "N/A",
            "impressions": 0,
            "clicks": 0,
            "ctr": "N/A",
            "cpc": "N/A",
            "days_with_data": 0,
        }

    total_spend = sum(float(i.spend or 0) for i in insights)
    total_leads = sum(int(i.leads or 0) for i in insights)
    total_impressions = sum(int(i.impressions or 0) for i in insights)
    total_clicks = sum(int(i.clicks or 0) for i in insights)

    cpl = total_spend / total_leads if total_leads > 0 else None
    ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else None
    cpc = total_spend / total_clicks if total_clicks > 0 else None

    return {
        "spend": format_currency(total_spend),
        "spend_raw": total_spend,
        "leads": total_leads,
        "cpl": format_currency(cpl) if cpl else "N/A",
        "cpl_raw": round(cpl, 2) if cpl else None,
        "impressions": format_number(total_impressions),
        "clicks": format_number(total_clicks),
        "ctr": format_percentage(ctr) if ctr else "N/A",
        "cpc": format_currency(cpc) if cpc else "N/A",
        "days_with_data": len(insights),
    }


def _calculate_trend(metrics_7d: dict, metrics_30d: dict) -> dict:
    """Calcula tendência comparando 7 dias com 30 dias."""
    trend = {}

    # CPL trend
    cpl_7d = metrics_7d.get("cpl_raw")
    cpl_30d = metrics_30d.get("cpl_raw")
    if cpl_7d and cpl_30d:
        cpl_change = ((cpl_7d - cpl_30d) / cpl_30d) * 100
        if cpl_change > 10:
            trend["cpl"] = f"📈 Subindo {format_percentage(abs(cpl_change))}"
        elif cpl_change < -10:
            trend["cpl"] = f"📉 Caindo {format_percentage(abs(cpl_change))}"
        else:
            trend["cpl"] = "➡️ Estável"
    else:
        trend["cpl"] = "N/A"

    # Leads trend
    leads_7d = metrics_7d.get("leads", 0)
    days_7d = metrics_7d.get("days_with_data", 7)
    leads_30d = metrics_30d.get("leads", 0)
    days_30d = metrics_30d.get("days_with_data", 30)

    if days_7d > 0 and days_30d > 0:
        daily_avg_7d = leads_7d / days_7d
        daily_avg_30d = leads_30d / days_30d
        if daily_avg_30d > 0:
            leads_change = ((daily_avg_7d - daily_avg_30d) / daily_avg_30d) * 100
            if leads_change > 10:
                trend["leads"] = f"📈 Subindo {format_percentage(abs(leads_change))}"
            elif leads_change < -10:
                trend["leads"] = f"📉 Caindo {format_percentage(abs(leads_change))}"
            else:
                trend["leads"] = "➡️ Estável"
        else:
            trend["leads"] = "N/A"
    else:
        trend["leads"] = "N/A"

    return trend


def _build_campaign_summary(name: str, metrics: dict) -> str:
    """Constrói resumo da campanha."""
    parts = [f"Campanha '{name}' nos últimos 7 dias:"]

    if metrics.get("leads", 0) > 0:
        parts.append(f"{metrics['leads']} leads")
    else:
        parts.append("0 leads")

    if metrics.get("cpl") != "N/A":
        parts.append(f"CPL {metrics['cpl']}")

    if metrics.get("spend") != "R$ 0,00":
        parts.append(f"Gasto {metrics['spend']}")

    return " | ".join(parts)
