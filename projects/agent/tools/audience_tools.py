"""
Tools do Especialista em Audiencias.

Todas as tools usam DB direto via SQLAlchemy.
Retornam ToolResult padronizado.
"""

from typing import List, Optional

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from sqlalchemy import select, func, and_

from shared.db.session import async_session_maker
from shared.db.models import (
    SistemaFacebookAdsAdsets,
    SistemaFacebookAdsInsightsHistory,
)

from projects.agent.tools.result import ToolResult, tool_success, tool_error
from projects.agent.tools.config_resolver import resolve_config_id
from projects.agent.tools.ownership import _validate_entity_ownership


@tool
async def get_adset_audiences(
    campaign_id: Optional[str] = None,
    config: RunnableConfig = None,
) -> ToolResult:
    """Dados de segmentacao dos adsets (targeting, idade, genero, interesses).

    Args:
        campaign_id: Filtrar por campanha (opcional).
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    try:
        async with async_session_maker() as session:
            query = select(
                SistemaFacebookAdsAdsets.adset_id,
                SistemaFacebookAdsAdsets.name,
                SistemaFacebookAdsAdsets.status,
                SistemaFacebookAdsAdsets.targeting,
                SistemaFacebookAdsAdsets.daily_budget,
                SistemaFacebookAdsAdsets.campaign_id,
            ).where(
                SistemaFacebookAdsAdsets.config_id == config_id,
            )

            if campaign_id:
                if not await _validate_entity_ownership(
                    campaign_id, config_id, "campaign",
                ):
                    return tool_error(
                        "OWNERSHIP_ERROR",
                        f"Campanha {campaign_id} nao encontrada nesta conta.",
                    )
                query = query.where(
                    SistemaFacebookAdsAdsets.campaign_id == campaign_id,
                )

            result = await session.execute(query.limit(50))
            rows = [dict(row._mapping) for row in result.all()]
            return tool_success(rows)
    except Exception as e:
        return tool_error("DB_ERROR", f"Erro ao buscar audiencias: {e}")


@tool
async def detect_audience_saturation(
    adset_ids: List[str],
    window_days: int = 14,
    config: RunnableConfig = None,
) -> ToolResult:
    """Analisa saturacao: frequency crescente + CTR decrescente = publico esgotado.

    Args:
        adset_ids: Lista de IDs dos adsets a analisar.
        window_days: Janela de analise em dias (default: 14).
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    try:
        model = SistemaFacebookAdsInsightsHistory
        saturated = []
        healthy = []

        async with async_session_maker() as session:
            for adset_id in adset_ids[:20]:
                if not await _validate_entity_ownership(
                    adset_id, config_id, "adset",
                ):
                    continue

                result = await session.execute(
                    select(
                        model.date,
                        model.ctr,
                        model.frequency,
                        model.impressions,
                    )
                    .where(and_(
                        model.adset_id == adset_id,
                        model.config_id == config_id,
                        model.date >= func.current_date() - window_days,
                    ))
                    .order_by(model.date.asc())
                )
                rows = result.all()

                if len(rows) < 3:
                    continue

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

                adset_info = {
                    "adset_id": adset_id,
                    "ctr_trend": round(avg_ctr_second - avg_ctr_first, 3),
                    "frequency_trend": round(avg_freq_second - avg_freq_first, 2),
                    "current_frequency": round(avg_freq_second, 2),
                }

                # Saturacao: frequency subindo E CTR caindo
                if avg_freq_second > avg_freq_first * 1.2 and avg_ctr_second < avg_ctr_first * 0.85:
                    adset_info["saturation_level"] = "high"
                    saturated.append(adset_info)
                elif avg_freq_second > avg_freq_first * 1.1:
                    adset_info["saturation_level"] = "medium"
                    saturated.append(adset_info)
                else:
                    adset_info["saturation_level"] = "low"
                    healthy.append(adset_info)

        return tool_success({
            "saturated": saturated,
            "healthy": healthy,
            "total_analyzed": len(saturated) + len(healthy),
        })
    except Exception as e:
        return tool_error("DB_ERROR", f"Erro ao detectar saturacao: {e}")


@tool
async def get_audience_performance(
    adset_ids: List[str],
    config: RunnableConfig = None,
) -> ToolResult:
    """Performance por audiencia: CPL, CTR, Leads por adset.

    Args:
        adset_ids: Lista de IDs dos adsets.
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    try:
        model = SistemaFacebookAdsInsightsHistory
        results_list = []

        async with async_session_maker() as session:
            for adset_id in adset_ids[:20]:
                if not await _validate_entity_ownership(
                    adset_id, config_id, "adset",
                ):
                    continue

                result = await session.execute(
                    select(
                        func.avg(model.cost_per_lead).label("avg_cpl"),
                        func.avg(model.ctr).label("avg_ctr"),
                        func.sum(model.leads).label("total_leads"),
                        func.sum(model.spend).label("total_spend"),
                        func.sum(model.impressions).label("total_impressions"),
                    )
                    .where(and_(
                        model.adset_id == adset_id,
                        model.config_id == config_id,
                        model.date >= func.current_date() - 14,
                    ))
                )
                row = result.one()
                results_list.append({
                    "adset_id": adset_id,
                    "avg_cpl": float(row.avg_cpl or 0),
                    "avg_ctr": float(row.avg_ctr or 0),
                    "total_leads": int(row.total_leads or 0),
                    "total_spend": float(row.total_spend or 0),
                    "total_impressions": int(row.total_impressions or 0),
                })

        # Ordenar por CPL (menor = melhor)
        results_list.sort(key=lambda x: x["avg_cpl"])

        return tool_success(results_list)
    except Exception as e:
        return tool_error("DB_ERROR", f"Erro ao buscar performance: {e}")
