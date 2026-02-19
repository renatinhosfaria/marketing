"""
Schema de roteamento do Supervisor.

RoutingDecision: decisao estruturada do Supervisor sobre quais agentes ativar.
AnalysisScope: escopo da analise para reduzir custo e latencia.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Literal, List, Optional


AgentType = Literal[
    "health_monitor",
    "performance_analyst",
    "creative_specialist",
    "forecast_scientist",
    "operations_manager",
    "audience_specialist",
]


class AnalysisScope(BaseModel):
    """Define o escopo da analise para reduzir custo e latencia.

    Sem scope, cada agente busca TODOS os dados da conta — ineficiente.
    Com scope, a query e filtrada: menos I/O, menos tokens, mais rapido.

    Validacoes com limites maximos previnem que o LLM (ou prompt injection)
    force analises caras demais em custo/latencia.
    """
    entity_type: Literal["campaign", "adset", "ad"] = Field(
        default="campaign",
        description="Nivel de entidade a analisar",
    )
    entity_ids: Optional[List[str]] = Field(
        default=None,
        max_length=50,
        description="IDs especificos mencionados pelo usuario (None = top N por spend, max 50)",
    )
    lookback_days: int = Field(
        default=7,
        ge=1,
        le=90,
        description="Janela de analise em dias (1-90)",
    )
    top_n: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Quantas entidades analisar se entity_ids nao especificado (1-50)",
    )


class RoutingDecision(BaseModel):
    """Decisao estruturada do Supervisor."""
    reasoning: str = Field(description="Por que estes agentes foram escolhidos")
    selected_agents: List[AgentType] = Field(
        description="Lista de agentes a serem ativados em paralelo"
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="Nivel de urgencia percebido"
    )
    scope: AnalysisScope = Field(
        default_factory=AnalysisScope,
        description="Escopo da analise (entidades, periodo, top N)",
    )


class ActionDecision(BaseModel):
    """Decisao estruturada do Operations Manager sobre qual acao executar."""
    action_needed: bool = Field(description="Se alguma acao e necessaria")
    action_type: Optional[Literal["budget_change", "status_change"]] = Field(
        default=None, description="Tipo da acao")
    campaign_id: Optional[str] = Field(
        default=None, description="ID da campanha alvo no Facebook")
    new_value: Optional[str] = Field(
        default=None, description="Budget em reais (ex: '75.0') ou status ('ACTIVE'/'PAUSED')")
    reason: str = Field(description="Justificativa em portugues")

    @model_validator(mode="after")
    def validate_value_matches_type(self):
        if not self.action_needed:
            return self
        if self.action_type == "budget_change" and self.new_value is not None:
            try:
                float(self.new_value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"new_value deve ser numerico para budget_change, recebido: {self.new_value}"
                )
        if self.action_type == "status_change" and self.new_value is not None:
            if self.new_value not in ("ACTIVE", "PAUSED"):
                raise ValueError(
                    f"new_value deve ser ACTIVE ou PAUSED para status_change, recebido: {self.new_value}"
                )
        return self
