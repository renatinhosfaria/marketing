"""
State schema do subgraph Analista de Performance & Impacto.

PerformanceSubgraphState: estado privado do subgraph.
PerformanceOutput: contrato de saida (apenas agent_reports).
"""

from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
import operator

from projects.agent.graph.state import AgentReport


class PerformanceSubgraphState(TypedDict):
    """Estado privado do subgraph Performance Analyst."""
    messages: Annotated[List[AnyMessage], add_messages]
    scope: Optional[dict]               # AnalysisScope do supervisor
    metrics_data: Optional[dict]        # Metricas agregadas
    metrics_error: Optional[str]       # Erro ao buscar metricas (None = sem erro)
    comparison: Optional[dict]          # Resultado de comparacao entre periodos
    impact_analysis: Optional[dict]     # Resultado de analise causal
    report: Optional[str]              # Relatorio final
    agent_reports: Annotated[List[AgentReport], operator.add]


class PerformanceOutput(TypedDict):
    """Contrato de saida do subgraph — apenas agent_reports."""
    agent_reports: List[AgentReport]
