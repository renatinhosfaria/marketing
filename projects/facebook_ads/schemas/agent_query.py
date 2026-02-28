"""Schemas para a ferramenta de consulta do agente FB Ads."""

from typing import Any, Optional

from pydantic import Field

from projects.facebook_ads.schemas.base import CamelCaseModel


class AgentQueryRequest(CamelCaseModel):
    """Entrada para consulta/operacao do agente via NL->SQL."""

    prompt: str = Field(..., description="Prompt em linguagem natural")
    sql: Optional[str] = Field(default=None, description="SQL opcional para override controlado")
    context: Optional[dict[str, Any]] = Field(default=None, description="Contexto opcional")


class AgentQueryResponse(CamelCaseModel):
    """Saida de execucao da query do agente."""

    operation_type: str
    sql_executed: str
    rows_affected: int
    rows: list[dict[str, Any]] = []
    duration_ms: int
