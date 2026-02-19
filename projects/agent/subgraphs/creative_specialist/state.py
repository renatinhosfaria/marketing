"""
State schema do subgraph Especialista em Criativos.

CreativeSubgraphState: estado privado do subgraph.
CreativeOutput: contrato de saida (apenas agent_reports).
"""

from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
import operator

from projects.agent.graph.state import AgentReport


class CreativeSubgraphState(TypedDict):
    """Estado privado do subgraph Creative Specialist."""
    messages: Annotated[List[AnyMessage], add_messages]
    scope: Optional[dict]               # AnalysisScope do supervisor
    ad_creatives: List[dict]            # Metadados dos anuncios
    fatigue_analysis: Optional[dict]    # Resultado da analise de fadiga
    preview_urls: List[str]             # URLs de preview para o frontend
    recommendation: Optional[str]       # Recomendacao final
    agent_reports: Annotated[List[AgentReport], operator.add]


class CreativeOutput(TypedDict):
    """Contrato de saida do subgraph — apenas agent_reports."""
    agent_reports: List[AgentReport]
