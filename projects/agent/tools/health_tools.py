"""
Tools do Monitor de Saude & Anomalias.

Todas as tools usam ML API via HTTP (CPU-heavy isolado).
Retornam ToolResult padronizado.
"""

from typing import List, Literal, Optional

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from projects.agent.tools.result import ToolResult, tool_success, tool_error
from projects.agent.tools.http_client import _ml_api_call
from projects.agent.tools.config_resolver import resolve_config_id
from projects.agent.tools.ml_schemas import MLDetectResponse


@tool
async def detect_anomalies(
    entity_type: Literal["campaign", "adset", "ad"],
    entity_ids: Optional[List[str]] = None,
    days: int = 1,
    config: RunnableConfig = None,
) -> ToolResult:
    """Executa deteccao de anomalias (IsolationForest + Z-score + IQR).

    Args:
        entity_type: Nivel de entidade (campaign, adset, ad).
        entity_ids: IDs especificos para analisar (None = todas).
        days: Dias de lookback para deteccao (default: 1).
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    result = await _ml_api_call(
        "post",
        "/api/v1/anomalies/detect",
        account_id=account_id,
        json={
            "config_id": config_id,
            "entity_type": entity_type,
            "entity_ids": entity_ids,
            "days": days,
        },
        timeout=30,
    )

    if not result["ok"]:
        return result

    # Valida schema da resposta
    try:
        validated = MLDetectResponse.model_validate(result["data"])
        return tool_success(validated.model_dump())
    except ValidationError as e:
        return tool_error(
            "SCHEMA_MISMATCH",
            f"ML API retornou resposta incompativel: {e.error_count()} erros. "
            "Verifique compatibilidade de versao.",
        )


@tool
async def get_classifications(
    entity_type: Literal["campaign", "adset", "ad"],
    config: RunnableConfig = None,
) -> ToolResult:
    """Busca classificacoes atuais (HIGH_PERFORMER, MODERATE, LOW, UNDERPERFORMER).

    Args:
        entity_type: Nivel de entidade (campaign, adset, ad).
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    return await _ml_api_call(
        "get",
        "/api/v1/classifications",
        account_id=account_id,
        params={"config_id": config_id, "entity_type": entity_type},
    )


@tool
async def classify_entity(
    entity_type: Literal["campaign", "adset", "ad"],
    entity_ids: Optional[List[str]] = None,
    config: RunnableConfig = None,
) -> ToolResult:
    """Classifica entidades por performance em tempo real.

    Args:
        entity_type: Nivel de entidade (campaign, adset, ad).
        entity_ids: IDs especificos para classificar (None = todas).
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    return await _ml_api_call(
        "post",
        "/api/v1/classifications/classify",
        json={
            "config_id": config_id,
            "entity_type": entity_type,
            "entity_ids": entity_ids,
        },
        account_id=account_id,
        timeout=15,
    )


@tool
async def get_anomaly_history(
    entity_id: Optional[str] = None,
    days: int = 7,
    config: RunnableConfig = None,
) -> ToolResult:
    """Retorna historico de anomalias detectadas nos ultimos N dias.

    Args:
        entity_id: ID da entidade para filtrar (None = todas).
        days: Dias de historico (default: 7).
    """
    cfg = (config or {}).get("configurable", {})
    account_id = cfg.get("account_id")

    config_id = await resolve_config_id(account_id)
    if config_id is None:
        return tool_error("NOT_FOUND", f"Conta {account_id} nao encontrada.")

    return await _ml_api_call(
        "get",
        "/api/v1/anomalies",
        account_id=account_id,
        params={
            "config_id": config_id,
            "entity_id": entity_id,
            "days": days,
        },
    )
