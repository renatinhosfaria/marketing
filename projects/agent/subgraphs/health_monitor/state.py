"""
State schema do subgraph Monitor de Saude & Anomalias.

HealthSubgraphState: estado privado do subgraph.
HealthOutput: contrato de saida (apenas agent_reports).
"""

from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
import operator

from projects.agent.graph.state import AgentReport


class HealthSubgraphState(TypedDict):
    """Estado privado do subgraph Health Monitor.

    Dados brutos de metricas NUNCA entram no State — sao carregados
    dentro do no, processados, e apenas agregacoes sao persistidas.
    """
    messages: Annotated[List[AnyMessage], add_messages]
    scope: Optional[dict]               # AnalysisScope do supervisor (via AgentInput)
    metrics_ref: Optional[str]          # Key Redis: "metrics:{thread_id}:{agent_id}:{uuid}"
    anomaly_results: Optional[dict]     # Scores e anomalias detectadas
    anomaly_error: Optional[str]        # Erro ao buscar anomalias (None = sem erro)
    classifications: Optional[dict]     # Tiers das entidades analisadas
    classifications_error: Optional[str] # Erro ao buscar classificacoes (None = sem erro)
    diagnosis: Optional[str]            # Diagnostico final
    agent_reports: Annotated[List[AgentReport], operator.add]


class HealthOutput(TypedDict):
    """Contrato de saida do subgraph — apenas agent_reports."""
    agent_reports: List[AgentReport]
