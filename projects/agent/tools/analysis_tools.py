"""
Tools para análise avançada de campanhas.
"""

from typing import Optional, List
from langchain_core.tools import tool
from sqlalchemy import select, desc, func, or_
from datetime import datetime, timedelta

from shared.db.session import get_db_session
from shared.db.models.famachat_readonly import (
    FacebookAdsCampaign,
    FacebookAdsInsight,
)
from shared.db.models.ml_readonly import MLCampaignClassification, CampaignTier
from shared.core.logging import get_logger
from projects.agent.tools.base import format_currency, format_percentage, format_number

logger = get_logger(__name__)


@tool
def compare_campaigns(
    config_id: int,
    campaign_ids: List[str],
    days: int = 7
) -> dict:
    """
    Compara métricas entre múltiplas campanhas.

    Use esta ferramenta para comparar performance de 2 ou mais campanhas
    lado a lado, identificando qual está melhor.

    Args:
        config_id: ID da configuração Facebook Ads
        campaign_ids: Lista de IDs das campanhas a comparar (2-5 campanhas)
        days: Período de comparação em dias (padrão: 7)

    Returns:
        Tabela comparativa com métricas e análise
    """
    try:
        if len(campaign_ids) < 2:
            return {"error": "Forneça pelo menos 2 campanhas para comparar."}
        if len(campaign_ids) > 5:
            return {"error": "Máximo de 5 campanhas por comparação."}

        since = datetime.utcnow() - timedelta(days=days)

        with get_db_session() as db:
            comparisons = []
            best_cpl = None
            best_cpl_campaign = None
            best_leads = 0
            best_leads_campaign = None

            for campaign_id in campaign_ids:
                # Buscar campanha
                campaign = db.execute(
                    select(FacebookAdsCampaign).where(
                        FacebookAdsCampaign.config_id == config_id,
                        FacebookAdsCampaign.campaign_id == campaign_id
                    )
                ).scalar_one_or_none()

                if not campaign:
                    comparisons.append({
                        "campaign_id": campaign_id,
                        "name": "Não encontrada",
                        "error": True
                    })
                    continue

                # Buscar insights
                insights = db.execute(
                    select(FacebookAdsInsight).where(
                        FacebookAdsInsight.campaign_id == campaign_id,
                        FacebookAdsInsight.date >= since
                    )
                ).scalars().all()

                # Calcular métricas
                total_spend = sum(float(i.spend or 0) for i in insights)
                total_leads = sum(int(i.leads or 0) for i in insights)
                total_impressions = sum(int(i.impressions or 0) for i in insights)
                total_clicks = sum(int(i.clicks or 0) for i in insights)

                cpl = total_spend / total_leads if total_leads > 0 else None
                ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else None

                # Atualizar melhores
                if cpl is not None and (best_cpl is None or cpl < best_cpl):
                    best_cpl = cpl
                    best_cpl_campaign = campaign.name
                if total_leads > best_leads:
                    best_leads = total_leads
                    best_leads_campaign = campaign.name

                comparisons.append({
                    "campaign_id": campaign_id,
                    "name": campaign.name,
                    "status": campaign.status,
                    "spend": format_currency(total_spend),
                    "spend_raw": total_spend,
                    "leads": total_leads,
                    "cpl": format_currency(cpl) if cpl else "N/A",
                    "cpl_raw": cpl,
                    "ctr": format_percentage(ctr) if ctr else "N/A",
                    "impressions": format_number(total_impressions),
                })

            # Marcar melhores
            for c in comparisons:
                if not c.get("error"):
                    c["is_best_cpl"] = c.get("cpl_raw") == best_cpl if best_cpl else False
                    c["is_best_leads"] = c.get("leads") == best_leads if best_leads > 0 else False

            # Construir análise
            analysis = []
            if best_cpl_campaign:
                analysis.append(f"✅ Melhor CPL: {best_cpl_campaign} ({format_currency(best_cpl)})")
            if best_leads_campaign:
                analysis.append(f"✅ Mais leads: {best_leads_campaign} ({best_leads} leads)")

            return {
                "period_days": days,
                "campaigns_compared": len(campaign_ids),
                "comparison": comparisons,
                "analysis": analysis,
                "summary": f"Comparação de {len(comparisons)} campanhas nos últimos {days} dias."
            }
    except Exception as e:
        logger.error("Erro ao comparar campanhas", error=str(e))
        return {"error": str(e)}


@tool
def analyze_trends(
    config_id: int,
    campaign_id: Optional[str] = None,
    days: int = 30
) -> dict:
    """
    Analisa tendências históricas de métricas.

    Use esta ferramenta para identificar padrões e tendências
    ao longo do tempo (CPL, leads, spend, etc.).

    Args:
        config_id: ID da configuração Facebook Ads
        campaign_id: ID de uma campanha específica ou None para conta toda
        days: Período de análise em dias (padrão: 30)

    Returns:
        Análise de tendências com insights
    """
    try:
        since = datetime.utcnow() - timedelta(days=days)

        with get_db_session() as db:
            query = select(FacebookAdsInsight).where(
                FacebookAdsInsight.date >= since
            )

            if campaign_id:
                query = query.where(FacebookAdsInsight.campaign_id == campaign_id)

            query = query.order_by(FacebookAdsInsight.date)
            insights = db.execute(query).scalars().all()

            if not insights:
                return {
                    "found": False,
                    "message": "Sem dados históricos para análise."
                }

            # Agrupar por semana
            weekly_data = {}
            for i in insights:
                week_key = i.date.strftime("%Y-W%W")
                if week_key not in weekly_data:
                    weekly_data[week_key] = {
                        "spend": 0,
                        "leads": 0,
                        "impressions": 0,
                        "clicks": 0,
                    }
                weekly_data[week_key]["spend"] += float(i.spend or 0)
                weekly_data[week_key]["leads"] += int(i.leads or 0)
                weekly_data[week_key]["impressions"] += int(i.impressions or 0)
                weekly_data[week_key]["clicks"] += int(i.clicks or 0)

            # Calcular CPL por semana
            weekly_metrics = []
            for week, data in sorted(weekly_data.items()):
                cpl = data["spend"] / data["leads"] if data["leads"] > 0 else None
                weekly_metrics.append({
                    "week": week,
                    "spend": format_currency(data["spend"]),
                    "leads": data["leads"],
                    "cpl": format_currency(cpl) if cpl else "N/A",
                    "cpl_raw": cpl,
                })

            # Analisar tendências
            trends = _analyze_weekly_trends(weekly_metrics)

            return {
                "found": True,
                "period_days": days,
                "campaign_id": campaign_id or "Conta inteira",
                "weekly_data": weekly_metrics,
                "trends": trends,
                "summary": _build_trend_summary(trends)
            }
    except Exception as e:
        logger.error("Erro ao analisar tendências", error=str(e))
        return {"error": str(e)}


@tool
def get_account_summary(
    config_id: int,
    days: int = 7
) -> dict:
    """
    Retorna um resumo geral da conta de anúncios.

    Use esta ferramenta para obter uma visão macro da performance
    da conta, incluindo totais e médias.

    Args:
        config_id: ID da configuração Facebook Ads
        days: Período de análise em dias (padrão: 7)

    Returns:
        Resumo com métricas agregadas da conta
    """
    try:
        since = datetime.utcnow() - timedelta(days=days)

        with get_db_session() as db:
            # Contar campanhas por status
            campaigns = db.execute(
                select(FacebookAdsCampaign).where(
                    FacebookAdsCampaign.config_id == config_id
                )
            ).scalars().all()

            status_counts = {}
            for c in campaigns:
                status = c.status or "UNKNOWN"
                status_counts[status] = status_counts.get(status, 0) + 1

            # Buscar insights agregados
            insights = db.execute(
                select(FacebookAdsInsight).where(
                    FacebookAdsInsight.date >= since
                )
            ).scalars().all()

            # Calcular totais
            total_spend = sum(float(i.spend or 0) for i in insights)
            total_leads = sum(int(i.leads or 0) for i in insights)
            total_impressions = sum(int(i.impressions or 0) for i in insights)
            total_clicks = sum(int(i.clicks or 0) for i in insights)

            avg_cpl = total_spend / total_leads if total_leads > 0 else None
            avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else None
            avg_cpc = total_spend / total_clicks if total_clicks > 0 else None

            # Buscar classificações válidas
            classifications = db.execute(
                select(MLCampaignClassification).where(
                    MLCampaignClassification.config_id == config_id,
                    or_(
                        MLCampaignClassification.valid_until.is_(None),
                        MLCampaignClassification.valid_until > datetime.utcnow()
                    )
                )
            ).scalars().all()

            tier_counts = {tier.value: 0 for tier in CampaignTier}
            for c in classifications:
                tier_counts[c.tier.value] += 1

            active_campaigns = status_counts.get("ACTIVE", 0)
            high_performers = tier_counts.get("HIGH_PERFORMER", 0)

            return {
                "period_days": days,
                "campaigns": {
                    "total": len(campaigns),
                    "active": active_campaigns,
                    "paused": status_counts.get("PAUSED", 0),
                    "by_status": status_counts,
                },
                "performance": {
                    "total_spend": format_currency(total_spend),
                    "total_leads": total_leads,
                    "avg_cpl": format_currency(avg_cpl) if avg_cpl else "N/A",
                    "avg_ctr": format_percentage(avg_ctr) if avg_ctr else "N/A",
                    "avg_cpc": format_currency(avg_cpc) if avg_cpc else "N/A",
                    "total_impressions": format_number(total_impressions),
                    "total_clicks": format_number(total_clicks),
                },
                "classifications": {
                    "high_performers": high_performers,
                    "moderate": tier_counts.get("MODERATE", 0),
                    "low": tier_counts.get("LOW", 0),
                    "underperformers": tier_counts.get("UNDERPERFORMER", 0),
                },
                "summary": f"Nos últimos {days} dias: {total_leads} leads gerados, "
                          f"CPL médio {format_currency(avg_cpl) if avg_cpl else 'N/A'}, "
                          f"investimento total {format_currency(total_spend)}. "
                          f"{active_campaigns} campanhas ativas, sendo {high_performers} high performers."
            }
    except Exception as e:
        logger.error("Erro ao gerar resumo da conta", error=str(e))
        return {"error": str(e)}


@tool
def calculate_roi(
    config_id: int,
    campaign_id: Optional[str] = None,
    average_ticket: float = 150000.0,
    conversion_rate: float = 5.0,
    days: int = 30
) -> dict:
    """
    Calcula o ROI estimado das campanhas.

    Use esta ferramenta para estimar o retorno sobre investimento
    baseado em leads gerados, ticket médio e taxa de conversão.

    Args:
        config_id: ID da configuração Facebook Ads
        campaign_id: ID de uma campanha específica ou None para conta toda
        average_ticket: Valor médio de venda em R$ (padrão: R$ 150.000)
        conversion_rate: Taxa de conversão de lead para venda em % (padrão: 5%)
        days: Período de análise em dias (padrão: 30)

    Returns:
        Análise de ROI com projeção de receita
    """
    try:
        since = datetime.utcnow() - timedelta(days=days)

        with get_db_session() as db:
            query = select(FacebookAdsInsight).where(
                FacebookAdsInsight.date >= since
            )

            if campaign_id:
                query = query.where(FacebookAdsInsight.campaign_id == campaign_id)

            insights = db.execute(query).scalars().all()

            if not insights:
                return {
                    "found": False,
                    "message": "Sem dados para calcular ROI."
                }

            # Calcular totais
            total_spend = sum(float(i.spend or 0) for i in insights)
            total_leads = sum(int(i.leads or 0) for i in insights)

            if total_spend == 0:
                return {
                    "found": True,
                    "message": "Sem investimento no período.",
                    "total_spend": "R$ 0,00",
                    "total_leads": total_leads,
                }

            # Calcular projeções
            estimated_sales = total_leads * (conversion_rate / 100)
            estimated_revenue = estimated_sales * average_ticket
            roi = ((estimated_revenue - total_spend) / total_spend) * 100 if total_spend > 0 else 0
            cost_per_sale = total_spend / estimated_sales if estimated_sales > 0 else None

            # Breakeven
            breakeven_leads = total_spend / (average_ticket * (conversion_rate / 100))

            return {
                "found": True,
                "period_days": days,
                "campaign_id": campaign_id or "Conta inteira",
                "inputs": {
                    "average_ticket": format_currency(average_ticket),
                    "conversion_rate": format_percentage(conversion_rate),
                },
                "metrics": {
                    "total_investment": format_currency(total_spend),
                    "total_leads": total_leads,
                    "cpl": format_currency(total_spend / total_leads) if total_leads > 0 else "N/A",
                },
                "projections": {
                    "estimated_sales": round(estimated_sales, 1),
                    "estimated_revenue": format_currency(estimated_revenue),
                    "roi": format_percentage(roi),
                    "cost_per_sale": format_currency(cost_per_sale) if cost_per_sale else "N/A",
                },
                "breakeven": {
                    "leads_needed": round(breakeven_leads, 0),
                    "current_leads": total_leads,
                    "status": "✅ Acima do breakeven" if total_leads >= breakeven_leads else "⚠️ Abaixo do breakeven",
                },
                "summary": f"ROI estimado de {format_percentage(roi)} no período. "
                          f"Com {total_leads} leads e taxa de conversão de {format_percentage(conversion_rate)}, "
                          f"projeção de {round(estimated_sales, 1)} vendas e receita de {format_currency(estimated_revenue)}."
            }
    except Exception as e:
        logger.error("Erro ao calcular ROI", error=str(e))
        return {"error": str(e)}


@tool
def get_top_campaigns(
    config_id: int,
    metric: str = "leads",
    days: int = 30,
    limit: int = 10
) -> dict:
    """
    Retorna as campanhas ranqueadas por uma métrica específica.

    Use esta ferramenta para identificar as melhores ou piores campanhas
    baseado em diferentes métricas como leads, CPL, gasto, etc.

    Args:
        config_id: ID da configuração Facebook Ads
        metric: Métrica para ordenar - "leads", "cpl", "spend", "ctr", "impressions" (padrão: leads)
        days: Período de análise em dias (padrão: 30)
        limit: Número máximo de campanhas a retornar (padrão: 10)

    Returns:
        Lista de campanhas ranqueadas pela métrica escolhida
    """
    try:
        since = datetime.utcnow() - timedelta(days=days)

        with get_db_session() as db:
            # Buscar todas as campanhas
            campaigns = db.execute(
                select(FacebookAdsCampaign).where(
                    FacebookAdsCampaign.config_id == config_id
                )
            ).scalars().all()

            if not campaigns:
                return {
                    "found": False,
                    "message": "Nenhuma campanha encontrada."
                }

            # Calcular métricas para cada campanha
            campaign_metrics = []
            for campaign in campaigns:
                insights = db.execute(
                    select(FacebookAdsInsight).where(
                        FacebookAdsInsight.campaign_id == campaign.campaign_id,
                        FacebookAdsInsight.date >= since
                    )
                ).scalars().all()

                total_spend = sum(float(i.spend or 0) for i in insights)
                total_leads = sum(int(i.leads or 0) for i in insights)
                total_impressions = sum(int(i.impressions or 0) for i in insights)
                total_clicks = sum(int(i.clicks or 0) for i in insights)

                cpl = total_spend / total_leads if total_leads > 0 else None
                ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else None

                campaign_metrics.append({
                    "campaign_id": campaign.campaign_id,
                    "name": campaign.name,
                    "status": campaign.status,
                    "leads": total_leads,
                    "leads_formatted": format_number(total_leads),
                    "spend": total_spend,
                    "spend_formatted": format_currency(total_spend),
                    "cpl": cpl,
                    "cpl_formatted": format_currency(cpl) if cpl else "N/A",
                    "ctr": ctr,
                    "ctr_formatted": format_percentage(ctr) if ctr else "N/A",
                    "impressions": total_impressions,
                    "impressions_formatted": format_number(total_impressions),
                    "clicks": total_clicks,
                })

            # Ordenar por métrica
            reverse_order = True  # Maior é melhor para leads, spend, impressions
            if metric == "cpl":
                # Filtrar campanhas sem CPL e ordenar (menor CPL é melhor)
                campaign_metrics = [c for c in campaign_metrics if c["cpl"] is not None]
                reverse_order = False

            sort_key = {
                "leads": lambda x: x["leads"],
                "cpl": lambda x: x["cpl"] if x["cpl"] else float('inf'),
                "spend": lambda x: x["spend"],
                "ctr": lambda x: x["ctr"] if x["ctr"] else 0,
                "impressions": lambda x: x["impressions"],
            }.get(metric, lambda x: x["leads"])

            campaign_metrics.sort(key=sort_key, reverse=reverse_order)
            top_campaigns = campaign_metrics[:limit]

            # Construir resultado
            result = []
            for i, c in enumerate(top_campaigns, 1):
                result.append({
                    "rank": i,
                    "campaign_id": c["campaign_id"],
                    "name": c["name"],
                    "status": c["status"],
                    "leads": c["leads"],
                    "spend": c["spend_formatted"],
                    "cpl": c["cpl_formatted"],
                    "ctr": c["ctr_formatted"],
                })

            # Identificar a melhor
            best = top_campaigns[0] if top_campaigns else None
            metric_labels = {
                "leads": "leads",
                "cpl": "menor CPL",
                "spend": "maior investimento",
                "ctr": "melhor CTR",
                "impressions": "mais impressões",
            }

            summary = ""
            if best:
                if metric == "leads":
                    summary = f"🏆 Campanha com mais leads: '{best['name']}' com {best['leads']} leads (CPL {best['cpl_formatted']}, gasto {best['spend_formatted']})."
                elif metric == "cpl":
                    summary = f"🏆 Campanha com melhor CPL: '{best['name']}' com {best['cpl_formatted']} (gerou {best['leads']} leads)."
                elif metric == "spend":
                    summary = f"🏆 Campanha com maior investimento: '{best['name']}' com {best['spend_formatted']} (gerou {best['leads']} leads)."
                else:
                    summary = f"🏆 Top campanha por {metric_labels.get(metric, metric)}: '{best['name']}'."

            return {
                "found": True,
                "period_days": days,
                "metric": metric,
                "total_campaigns": len(campaigns),
                "campaigns_with_data": len(top_campaigns),
                "ranking": result,
                "best_campaign": {
                    "name": best["name"],
                    "campaign_id": best["campaign_id"],
                    metric: best.get(metric, best["leads"]),
                } if best else None,
                "summary": summary,
            }
    except Exception as e:
        logger.error("Erro ao buscar top campanhas", error=str(e))
        return {"error": str(e)}


def _analyze_weekly_trends(weekly_metrics: list) -> dict:
    """Analisa tendências semanais."""
    trends = {}

    if len(weekly_metrics) < 2:
        return {"cpl": "Dados insuficientes", "leads": "Dados insuficientes"}

    # Tendência de CPL
    cpl_values = [w.get("cpl_raw") for w in weekly_metrics if w.get("cpl_raw")]
    if len(cpl_values) >= 2:
        first_half_avg = sum(cpl_values[:len(cpl_values)//2]) / (len(cpl_values)//2)
        second_half_avg = sum(cpl_values[len(cpl_values)//2:]) / (len(cpl_values) - len(cpl_values)//2)

        if second_half_avg > first_half_avg * 1.1:
            trends["cpl"] = "📈 Em alta - CPL aumentando ao longo do período"
        elif second_half_avg < first_half_avg * 0.9:
            trends["cpl"] = "📉 Em queda - CPL melhorando ao longo do período"
        else:
            trends["cpl"] = "➡️ Estável - CPL mantendo consistência"
    else:
        trends["cpl"] = "Dados insuficientes"

    # Tendência de Leads
    leads_values = [w.get("leads", 0) for w in weekly_metrics]
    if len(leads_values) >= 2:
        first_half_sum = sum(leads_values[:len(leads_values)//2])
        second_half_sum = sum(leads_values[len(leads_values)//2:])

        if second_half_sum > first_half_sum * 1.1:
            trends["leads"] = "📈 Em alta - Volume de leads crescendo"
        elif second_half_sum < first_half_sum * 0.9:
            trends["leads"] = "📉 Em queda - Volume de leads diminuindo"
        else:
            trends["leads"] = "➡️ Estável - Volume de leads consistente"
    else:
        trends["leads"] = "Dados insuficientes"

    return trends


def _build_trend_summary(trends: dict) -> str:
    """Constrói resumo das tendências."""
    parts = []

    cpl_trend = trends.get("cpl", "")
    if "alta" in cpl_trend.lower():
        parts.append("⚠️ CPL em tendência de alta, atenção necessária")
    elif "queda" in cpl_trend.lower():
        parts.append("✅ CPL em tendência de queda, bom trabalho!")
    elif "estável" in cpl_trend.lower():
        parts.append("CPL se mantendo estável")

    leads_trend = trends.get("leads", "")
    if "alta" in leads_trend.lower():
        parts.append("✅ Volume de leads crescendo")
    elif "queda" in leads_trend.lower():
        parts.append("⚠️ Volume de leads em queda")
    elif "estável" in leads_trend.lower():
        parts.append("Volume de leads consistente")

    return " | ".join(parts) if parts else "Análise de tendências não disponível"
