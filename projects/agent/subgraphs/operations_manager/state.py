"""
State schema do subgraph Gerente de Operacoes.

OperationsSubgraphState: estado privado do subgraph.
OperationsOutput: contrato de saida (apenas agent_reports).
"""

from typing import Annotated, List, Optional, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
import operator

from projects.agent.graph.state import AgentReport


class OperationsSubgraphState(TypedDict):
    """Estado privado do subgraph Operations Manager."""
    messages: Annotated[List[AnyMessage], add_messages]
    scope: Optional[dict]               # AnalysisScope do supervisor
    proposed_action: Optional[dict]     # Acao proposta com dry_run
    approval_status: Optional[Literal["pending", "approved", "rejected"]]
    execution_result: Optional[dict]    # Resultado da API do Facebook
    agent_reports: Annotated[List[AgentReport], operator.add]


class OperationsOutput(TypedDict):
    """Contrato de saida do subgraph — apenas agent_reports."""
    agent_reports: List[AgentReport]
