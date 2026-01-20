"""Subagentes especialistas do sistema multi-agente.

Este modulo contem os 6 subagentes que sao coordenados pelo Orchestrator:
- ClassificationAgent: Analise de tiers de performance
- AnomalyAgent: Deteccao de problemas e alertas
- ForecastAgent: Previsoes de CPL/Leads
- RecommendationAgent: Recomendacoes de acoes
- CampaignAgent: Dados de campanhas
- AnalysisAgent: Analises avancadas
"""
from typing import Type

from app.agent.subagents.state import (
    SubagentState,
    SubagentTask,
    AgentResult,
    create_initial_subagent_state
)
from app.agent.subagents.base import BaseSubagent

# Import agents
from app.agent.subagents.classification.agent import ClassificationAgent
from app.agent.subagents.anomaly.agent import AnomalyAgent
from app.agent.subagents.forecast.agent import ForecastAgent
from app.agent.subagents.recommendation.agent import RecommendationAgent
from app.agent.subagents.campaign.agent import CampaignAgent
from app.agent.subagents.analysis.agent import AnalysisAgent


# Registry of all subagents
SUBAGENT_REGISTRY: dict[str, Type[BaseSubagent]] = {
    "classification": ClassificationAgent,
    "anomaly": AnomalyAgent,
    "forecast": ForecastAgent,
    "recommendation": RecommendationAgent,
    "campaign": CampaignAgent,
    "analysis": AnalysisAgent,
}


def get_subagent(name: str) -> BaseSubagent:
    """Retorna instancia de subagente pelo nome.

    Args:
        name: Nome do subagente (classification, anomaly, etc.)

    Returns:
        Instancia do subagente

    Raises:
        ValueError: Se nome nao for valido
    """
    if name not in SUBAGENT_REGISTRY:
        valid = ", ".join(SUBAGENT_REGISTRY.keys())
        raise ValueError(f"Subagente '{name}' nao encontrado. Validos: {valid}")

    return SUBAGENT_REGISTRY[name]()


def get_all_subagents() -> list[BaseSubagent]:
    """Retorna lista com todos os subagentes instanciados.

    Returns:
        Lista de instancias de subagentes
    """
    return [cls() for cls in SUBAGENT_REGISTRY.values()]


__all__ = [
    # State
    "SubagentState",
    "SubagentTask",
    "AgentResult",
    "create_initial_subagent_state",
    # Base
    "BaseSubagent",
    # Agents
    "ClassificationAgent",
    "AnomalyAgent",
    "ForecastAgent",
    "RecommendationAgent",
    "CampaignAgent",
    "AnalysisAgent",
    # Registry
    "SUBAGENT_REGISTRY",
    "get_subagent",
    "get_all_subagents",
]
