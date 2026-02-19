"""
Tools do Analista de Performance & Impacto.

Combina DB via SQLAlchemy (metricas, comparacoes) e ML API via HTTP (impacto causal).
Retornam ToolResult padronizado.
"""

from typing import List, Literal

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from sqlalchemy import select, func, and_

from shared.db.session import async_session_maker
from shared.db.models import SistemaFacebookAdsInsightsHistory

from projects.agent.tools.result import ToolResult, tool_success, tool_error
from projects.agent.tools.http_client import _ml_api_call
from projects.agent.tools.config_resolver import resolve_config_id
from projects.agent.tools.ownership import _validate_entity_ownership
from projects.agent.config import agent_settings


# Mapa entity_type -> coluna na tabela InsightsHistory
_ENTITY_COLUMN = {
    "campaign": "campaign_id",
    "adset": "adset_id",
    "ad": "ad_id",
}


@tool
async def get_campaign_insights(
    entity_type: Literal["campaign", "adset", "ad"],
    entity_id: str,
    date_start: str,
    date_end: str,
    config: RunnableConfig = None,
) -> ToolResult:
    """Busca metricas detalhadas (spend, leads, CPL, CTR, CPC, impressions) por periodo.

    Args:
        entity_type: Nivel de entidade (campaign, adset, ad).
        entity_id: ID da entidade.
        date_start: Data de inicio (YYYY-MM-DD).
        date_end: Data de fim (YYYY-MM-DD).
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    # Validar ownership
    if not await _validate_entity_ownership(entity_id, config_id, entity_type):
        return tool_error(
            "OWNERSHIP_ERROR",
            f"Entidade {entity_id} nao encontrada nesta conta.",
        )

    try:
        model = SistemaFacebookAdsInsightsHistory
        entity_col = getattr(model, _ENTITY_COLUMN[entity_type])
        async with async_session_maker() as session:
            result = await session.execute(
                select(
                    model.date,
                    model.spend,
                    model.leads,
                    model.cost_per_lead,
                    model.ctr,
                    model.cpc,
                    model.impressions,
                )
                .where(and_(
                    entity_col == entity_id,
                    model.config_id == config_id,
                    model.date >= date_start,
                    model.date <= date_end,
                ))
                .order_by(model.date.asc())
            )
            rows = [dict(row._mapping) for row in result.all()]
            return tool_success(rows)
    except Exception as e:
        return tool_error("DB_ERROR", f"Erro ao buscar insights: {e}")


@tool
async def compare_periods(
    entity_id: str,
    period_a_start: str,
    period_a_end: str,
    period_b_start: str,
    period_b_end: str,
    entity_type: Literal["campaign", "adset", "ad"] = "campaign",
    metrics: List[str] = None,
    config: RunnableConfig = None,
) -> ToolResult:
    """Compara metricas entre dois periodos (ex: semana atual vs anterior).

    Args:
        entity_id: ID da entidade.
        period_a_start: Inicio do periodo A (YYYY-MM-DD).
        period_a_end: Fim do periodo A (YYYY-MM-DD).
        period_b_start: Inicio do periodo B (YYYY-MM-DD).
        period_b_end: Fim do periodo B (YYYY-MM-DD).
        entity_type: Nivel de entidade (campaign, adset, ad).
        metrics: Lista de metricas a comparar (default: spend, leads, cost_per_lead, ctr).
    """
    if metrics is None:
        metrics = ["spend", "leads", "cost_per_lead", "ctr"]

    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    if not await _validate_entity_ownership(entity_id, config_id, entity_type):
        return tool_error(
            "OWNERSHIP_ERROR",
            f"Entidade {entity_id} nao encontrada nesta conta.",
        )

    try:
        model = SistemaFacebookAdsInsightsHistory
        entity_col = getattr(model, _ENTITY_COLUMN[entity_type])

        async def _aggregate_period(start: str, end: str) -> dict:
            async with async_session_maker() as session:
                result = await session.execute(
                    select(
                        func.sum(model.spend).label("spend"),
                        func.sum(model.leads).label("leads"),
                        func.avg(model.cost_per_lead).label("cost_per_lead"),
                        func.avg(model.ctr).label("ctr"),
                        func.avg(model.cpc).label("cpc"),
                        func.sum(model.impressions).label("impressions"),
                    )
                    .where(and_(
                        entity_col == entity_id,
                        model.config_id == config_id,
                        model.date >= start,
                        model.date <= end,
                    ))
                )
                row = result.one()
                return {
                    m: float(getattr(row, m) or 0)
                    for m in metrics
                    if hasattr(row, m)
                }

        period_a = await _aggregate_period(period_a_start, period_a_end)
        period_b = await _aggregate_period(period_b_start, period_b_end)

        diffs = {}
        for m in metrics:
            a_val = period_a.get(m, 0)
            b_val = period_b.get(m, 0)
            if a_val and a_val != 0:
                diffs[m] = {
                    "absolute": round(b_val - a_val, 2),
                    "pct": round(((b_val - a_val) / a_val) * 100, 1),
                }
            else:
                diffs[m] = {"absolute": round(b_val - a_val, 2), "pct": None}

        return tool_success({
            "period_a": period_a,
            "period_b": period_b,
            "diffs": diffs,
        })
    except Exception as e:
        return tool_error("DB_ERROR", f"Erro ao comparar periodos: {e}")


@tool
async def analyze_causal_impact(
    entity_id: str,
    change_date: str | None = None,
    change_type: Literal[
        "budget_change",
        "creative_change",
        "audience_change",
        "pause",
        "reactivate",
    ] = "budget_change",
    entity_type: Literal["campaign", "adset", "ad"] = "campaign",
    intervention_date: str | None = None,
    window_before: int = 7,
    window_after: int = 7,
    config: RunnableConfig = None,
) -> ToolResult:
    """Analisa impacto causal de uma mudanca na campanha.

    Args:
        entity_id: ID da entidade.
        change_date: Data da mudanca (ISO date/datetime).
        change_type: Tipo da mudanca analisada.
        entity_type: Nivel da entidade (campaign/adset/ad).
        intervention_date: Alias legado para change_date.
        window_before: Janela de dias antes da mudanca.
        window_after: Janela de dias depois da mudanca.
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    final_change_date = change_date or intervention_date
    if not final_change_date:
        return tool_error(
            "VALIDATION_ERROR",
            "change_date e obrigatorio para analise de impacto causal.",
        )

    if agent_settings.enable_ml_endpoint_fixes:
        path = "/api/v1/impact/analyze"
        payload = {
            "config_id": config_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "change_date": final_change_date,
            "change_type": change_type,
            "window_before": window_before,
            "window_after": window_after,
        }
    else:
        path = "/api/v1/impact"
        payload = {
            "config_id": config_id,
            "entity_id": entity_id,
            "intervention_date": final_change_date,
            "metric": "cpl",
        }

    result = await _ml_api_call(
        "post",
        path,
        json=payload,
        account_id=account_id,
        timeout=30,
    )

    if not result.get("ok"):
        return result

    payload = result.get("data")
    if agent_settings.enable_ml_endpoint_fixes:
        if not isinstance(payload, dict):
            return tool_error(
                "SCHEMA_MISMATCH",
                "Resposta de impacto causal em formato inesperado.",
            )
        # Compatibilidade: backend pode retornar schema novo (overall_impact)
        # ou legado (impact_pct/significant).
        if "overall_impact" not in payload and "impact_pct" not in payload:
            return tool_error(
                "SCHEMA_MISMATCH",
                "Resposta de impacto causal em formato inesperado.",
            )

    return result


@tool
async def get_insights_summary(
    config: RunnableConfig = None,
) -> ToolResult:
    """Resumo de KPIs agregados da conta (total spend, leads, CPL medio).

    Retorna metricas agregadas dos ultimos 7 dias.
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    try:
        model = SistemaFacebookAdsInsightsHistory
        async with async_session_maker() as session:
            result = await session.execute(
                select(
                    func.sum(model.spend).label("total_spend"),
                    func.sum(model.leads).label("total_leads"),
                    func.avg(model.cost_per_lead).label("avg_cpl"),
                    func.avg(model.ctr).label("avg_ctr"),
                    func.sum(model.impressions).label("total_impressions"),
                )
                .where(
                    model.config_id == config_id,
                    model.date >= func.current_date() - 7,
                )
            )
            row = result.one()
            return tool_success({
                "total_spend": float(row.total_spend or 0),
                "total_leads": int(row.total_leads or 0),
                "avg_cpl": float(row.avg_cpl or 0),
                "avg_ctr": float(row.avg_ctr or 0),
                "total_impressions": int(row.total_impressions or 0),
                "period": "ultimos 7 dias",
            })
    except Exception as e:
        return tool_error("DB_ERROR", f"Erro ao buscar resumo: {e}")
