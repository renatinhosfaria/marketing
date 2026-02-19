"""
State schemas do grafo principal (Supervisor).

SupervisorState: estado compartilhado do grafo principal.
AgentReport: contrato de retorno dos agentes para o Synthesizer.
UserContext: dados do usuario autenticado.
AgentInput: input padrao para todos os subgraphs (compativel com Send()).
"""

from typing import Annotated, List, Optional, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from langchain_core.messages import AnyMessage
import operator


class AgentReport(TypedDict):
    """Relatorio produzido por cada agente para o Synthesizer."""
    agent_id: str
    status: Literal["running", "completed", "error"]
    summary: str              # Resumo textual para o Synthesizer
    data: Optional[dict]      # Dados estruturados (metricas, scores)
    confidence: float         # 0.0 - 1.0


class UserContext(TypedDict):
    """Dados do usuario autenticado, injetados pelo endpoint."""
    user_id: str
    account_id: str
    account_name: str
    timezone: str             # Default: America/Sao_Paulo


class SupervisorState(TypedDict):
    """Estado global do grafo principal.

    - messages: historico de mensagens (reducer: add_messages)
    - agent_reports: acumula reports dos agentes via operator.add (fan-in)
    - remaining_steps: controle de recursao (managed pelo LangGraph)
    """
    messages: Annotated[List[AnyMessage], add_messages]
    user_context: UserContext
    routing_decision: Optional[dict]
    agent_reports: Annotated[List[AgentReport], operator.add]
    pending_actions: List[dict]
    synthesis: Optional[str]
    remaining_steps: RemainingSteps


class AgentInput(TypedDict):
    """Input padrao para TODOS os subgraphs.

    Compativel com o arg do Send() no supervisor.
    O scope e propagado para que cada agente filtre por entidades/periodo relevantes.
    """
    messages: List[AnyMessage]
    scope: Optional[dict]
