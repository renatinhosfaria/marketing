"""
Tools do Cientista de Previsao.

Todas as tools usam ML API via HTTP (Prophet + Ensemble).
Retornam ToolResult padronizado.
"""

from typing import Literal, Optional

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from projects.agent.tools.result import ToolResult, tool_error
from projects.agent.tools.http_client import _ml_api_call
from projects.agent.tools.config_resolver import resolve_config_id
from projects.agent.tools.ownership import _validate_entity_ownership
from projects.agent.config import agent_settings


@tool
async def generate_forecast(
    entity_id: str,
    entity_type: Literal["campaign", "adset", "ad"] = "campaign",
    metric: str = "cpl",
    horizon_days: int = 7,
    config: RunnableConfig = None,
) -> ToolResult:
    """Gera previsao (Prophet + Ensemble) para N dias.

    Args:
        entity_id: ID da entidade (campanha, adset, ad).
        entity_type: Nivel de entidade (campaign, adset, ad).
        metric: Metrica a prever (cpl, leads).
        horizon_days: Horizonte de previsao em dias (default: 7).
    """
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

    supported_metrics = {"cpl", "leads"}
    if not agent_settings.enable_ml_endpoint_fixes:
        supported_metrics.add("spend")

    if metric not in supported_metrics:
        return tool_error(
            "NOT_SUPPORTED",
            f"Metrica '{metric}' nao suportada no endpoint atual de previsoes.",
        )

    endpoint = f"/api/v1/predictions/{metric}"
    return await _ml_api_call(
        "post",
        endpoint,
        json={
            "config_id": config_id,
            "entity_id": entity_id,
            "horizon_days": horizon_days,
        },
        account_id=account_id,
        timeout=30,
    )


@tool
async def get_forecast_history(
    entity_type: Literal["campaign", "adset", "ad"] = "campaign",
    entity_id: Optional[str] = None,
    config: RunnableConfig = None,
) -> ToolResult:
    """Previsoes anteriores com acuracia real (previsto vs realizado).

    Args:
        entity_type: Nivel de entidade (campaign, adset, ad).
        entity_id: ID da entidade.
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    if agent_settings.enable_ml_endpoint_fixes:
        path = "/api/v1/forecasts"
        params = {
            "config_id": config_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
        }
    else:
        path = f"/api/v1/predictions/series/{entity_type}/{entity_id}"
        params = {"config_id": config_id}

    return await _ml_api_call(
        "get",
        path,
        account_id=account_id,
        params=params,
    )


@tool
async def validate_forecast(
    forecast_id: str,
    config: RunnableConfig = None,
) -> ToolResult:
    """Compara previsao passada com resultado real (MAPE, MAE).

    Args:
        forecast_id: ID da previsao a validar.
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    if agent_settings.enable_ml_endpoint_fixes:
        return tool_error(
            "NOT_SUPPORTED",
            "Validacao de forecast ainda nao esta disponivel no backend ML.",
        )

    return await _ml_api_call(
        "get",
        f"/api/v1/forecasts/{forecast_id}/validate",
        account_id=account_id,
        params={"config_id": config_id},
    )
