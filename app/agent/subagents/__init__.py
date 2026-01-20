"""Subagentes especialistas do sistema multi-agente.

Este módulo contém os 6 subagentes que são coordenados pelo Orchestrator:
- ClassificationAgent: Análise de tiers de performance
- AnomalyAgent: Detecção de problemas e alertas
- ForecastAgent: Previsões de CPL/Leads
- RecommendationAgent: Recomendações de ações
- CampaignAgent: Dados de campanhas
- AnalysisAgent: Análises avançadas
"""
from app.agent.subagents.state import (
    SubagentState,
    SubagentTask,
    AgentResult,
    create_initial_subagent_state
)
from app.agent.subagents.base import BaseSubagent

__all__ = [
    # State
    "SubagentState",
    "SubagentTask",
    "AgentResult",
    "create_initial_subagent_state",
    # Base
    "BaseSubagent",
]
