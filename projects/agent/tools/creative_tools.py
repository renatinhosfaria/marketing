"""
Tools do Especialista em Criativos.

Combina DB direto (metadados de anuncios, analise de fadiga) e
Facebook API (preview URLs).
Retornam ToolResult padronizado.
"""

from typing import List, Optional

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from sqlalchemy import select, func, and_

from shared.db.session import async_session_maker
from shared.db.models import (
    SistemaFacebookAdsAds,
    SistemaFacebookAdsInsightsHistory,
)

from projects.agent.tools.result import ToolResult, tool_success, tool_error
from projects.agent.tools.config_resolver import resolve_config_id
from projects.agent.tools.ownership import _validate_entity_ownership


@tool
async def get_ad_creatives(
    campaign_id: Optional[str] = None,
    adset_id: Optional[str] = None,
    config: RunnableConfig = None,
) -> ToolResult:
    """Lista anuncios com metadados (formato, copy, thumbnail URL).

    Args:
        campaign_id: Filtrar por campanha (opcional).
        adset_id: Filtrar por adset (opcional).
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    try:
        async with async_session_maker() as session:
            query = select(
                SistemaFacebookAdsAds.ad_id,
                SistemaFacebookAdsAds.name,
                SistemaFacebookAdsAds.status,
                SistemaFacebookAdsAds.creative_id,
                SistemaFacebookAdsAds.adset_id,
                SistemaFacebookAdsAds.campaign_id,
            ).where(
                SistemaFacebookAdsAds.config_id == config_id,
            )

            if campaign_id:
                query = query.where(
                    SistemaFacebookAdsAds.campaign_id == campaign_id,
                )
            if adset_id:
                query = query.where(
                    SistemaFacebookAdsAds.adset_id == adset_id,
                )

            result = await session.execute(query.limit(50))
            rows = [dict(row._mapping) for row in result.all()]
            return tool_success(rows)
    except Exception as e:
        return tool_error("DB_ERROR", f"Erro ao buscar criativos: {e}")


@tool
async def detect_creative_fatigue(
    ad_ids: List[str],
    window_days: int = 14,
    config: RunnableConfig = None,
) -> ToolResult:
    """Detecta fadiga criativa: queda de CTR + aumento de frequency ao longo do tempo.

    Args:
        ad_ids: Lista de IDs dos anuncios a analisar.
        window_days: Janela de analise em dias (default: 14).
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    try:
        model = SistemaFacebookAdsInsightsHistory
        fatigued_ads = []
        healthy_ads = []

        async with async_session_maker() as session:
            for ad_id in ad_ids[:20]:  # Limitar a 20 anuncios
                # Validar ownership
                if not await _validate_entity_ownership(ad_id, config_id, "ad"):
                    continue

                result = await session.execute(
                    select(
                        model.date,
                        model.ctr,
                        model.frequency,
                        model.impressions,
                    )
                    .where(and_(
                        model.ad_id == ad_id,
                        model.config_id == config_id,
                        model.date >= func.current_date() - window_days,
                    ))
                    .order_by(model.date.asc())
                )
                rows = result.all()

                if len(rows) < 3:
                    continue

                # Analise de tendencia simples: CTR caindo + frequency subindo
                first_half = rows[:len(rows) // 2]
                second_half = rows[len(rows) // 2:]

                avg_ctr_first = sum(
                    float(r.ctr or 0) for r in first_half
                ) / len(first_half)
                avg_ctr_second = sum(
                    float(r.ctr or 0) for r in second_half
                ) / len(second_half)
                avg_freq_first = sum(
                    float(r.frequency or 0) for r in first_half
                ) / len(first_half)
                avg_freq_second = sum(
                    float(r.frequency or 0) for r in second_half
                ) / len(second_half)

                ctr_declining = avg_ctr_second < avg_ctr_first * 0.85
                freq_increasing = avg_freq_second > avg_freq_first * 1.2

                ad_info = {
                    "ad_id": ad_id,
                    "ctr_first_half": round(avg_ctr_first, 3),
                    "ctr_second_half": round(avg_ctr_second, 3),
                    "frequency_first_half": round(avg_freq_first, 2),
                    "frequency_second_half": round(avg_freq_second, 2),
                }

                if ctr_declining and freq_increasing:
                    ad_info["fatigue_score"] = "high"
                    fatigued_ads.append(ad_info)
                elif ctr_declining or freq_increasing:
                    ad_info["fatigue_score"] = "medium"
                    fatigued_ads.append(ad_info)
                else:
                    ad_info["fatigue_score"] = "none"
                    healthy_ads.append(ad_info)

        return tool_success({
            "fatigued_ads": fatigued_ads,
            "healthy_ads": healthy_ads,
            "total_analyzed": len(fatigued_ads) + len(healthy_ads),
        })
    except Exception as e:
        return tool_error("DB_ERROR", f"Erro ao detectar fadiga criativa: {e}")


@tool
async def compare_creatives(
    ad_ids: List[str],
    metric: str = "ctr",
    config: RunnableConfig = None,
) -> ToolResult:
    """Compara performance entre criativos do mesmo adset/campanha.

    Args:
        ad_ids: Lista de IDs dos anuncios a comparar.
        metric: Metrica principal para comparacao (default: ctr).
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    try:
        model = SistemaFacebookAdsInsightsHistory
        rankings = []

        async with async_session_maker() as session:
            for ad_id in ad_ids[:20]:
                if not await _validate_entity_ownership(ad_id, config_id, "ad"):
                    continue

                result = await session.execute(
                    select(
                        func.avg(model.ctr).label("avg_ctr"),
                        func.avg(model.cost_per_lead).label("avg_cpl"),
                        func.sum(model.spend).label("total_spend"),
                        func.sum(model.leads).label("total_leads"),
                        func.sum(model.impressions).label("total_impressions"),
                    )
                    .where(and_(
                        model.ad_id == ad_id,
                        model.config_id == config_id,
                        model.date >= func.current_date() - 14,
                    ))
                )
                row = result.one()
                rankings.append({
                    "ad_id": ad_id,
                    "avg_ctr": float(row.avg_ctr or 0),
                    "avg_cpl": float(row.avg_cpl or 0),
                    "total_spend": float(row.total_spend or 0),
                    "total_leads": int(row.total_leads or 0),
                    "total_impressions": int(row.total_impressions or 0),
                })

        # Ordenar por metrica principal (descendente para ctr, ascendente para cpl)
        reverse = metric != "cpl"
        rankings.sort(key=lambda x: x.get(f"avg_{metric}", 0), reverse=reverse)

        return tool_success({
            "rankings": rankings,
            "best": rankings[0] if rankings else None,
            "worst": rankings[-1] if rankings else None,
            "metric": metric,
        })
    except Exception as e:
        return tool_error("DB_ERROR", f"Erro ao comparar criativos: {e}")


@tool
async def get_ad_preview_url(
    ad_id: str,
    config: RunnableConfig = None,
) -> ToolResult:
    """Retorna URL de preview do anuncio para renderizar no frontend.

    Args:
        ad_id: ID do anuncio no Facebook.
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    if not await _validate_entity_ownership(ad_id, config_id, "ad"):
        return tool_error(
            "OWNERSHIP_ERROR",
            f"Anuncio {ad_id} nao encontrado nesta conta.",
        )

    try:
        # TODO: Implementar chamada direta a Facebook API para preview
        # from projects.facebook_ads.client import FacebookAdsClient
        # preview_url = await client.get_ad_preview(ad_id)
        return tool_success({
            "ad_id": ad_id,
            "preview_url": f"https://www.facebook.com/ads/archive/render_ad/?id={ad_id}",
            "note": "URL de preview gerada. Para preview completo, requer token.",
        })
    except Exception as e:
        return tool_error("FB_API_ERROR", f"Erro ao buscar preview: {e}")
