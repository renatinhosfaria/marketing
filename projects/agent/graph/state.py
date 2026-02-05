"""
Definição do estado do agente de tráfego pago.

⚠️ DEPRECATION WARNING:
Este módulo define o estado do agente monolítico LEGADO e está DEPRECADO desde 2026-01-21.
Use app/agent/orchestrator/state.py (OrchestratorState) ou app/agent/subagents/state.py (SubagentState).

Para habilitar o novo sistema: AGENT_MULTI_AGENT_ENABLED=true
"""

import warnings
from typing import TypedDict, Annotated, Sequence, Optional, Any

warnings.warn(
    "app.agent.graph.state está DEPRECADO. Use app.agent.orchestrator.state ou app.agent.subagents.state",
    DeprecationWarning,
    stacklevel=2
)
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Estado do agente de tráfego pago.

    Este estado é passado entre os nós do grafo LangGraph
    e mantém todo o contexto da conversa.
    """

    # ==========================================
    # Mensagens da conversa
    # ==========================================
    # Acumula mensagens usando add_messages reducer
    messages: Annotated[Sequence[dict], add_messages]

    # ==========================================
    # Contexto da sessão
    # ==========================================
    config_id: int          # ID da configuração Facebook Ads
    user_id: int            # ID do usuário autenticado
    thread_id: str          # ID da thread para persistência

    # ==========================================
    # Dados coletados durante análise
    # ==========================================
    classifications: Optional[list[dict]]    # Classificações de campanhas
    recommendations: Optional[list[dict]]    # Recomendações ativas
    anomalies: Optional[list[dict]]          # Anomalias detectadas
    forecasts: Optional[list[dict]]          # Previsões de métricas

    # ==========================================
    # Estado da análise atual
    # ==========================================
    current_intent: Optional[str]            # Intenção detectada do usuário
    selected_campaigns: list[str]            # IDs de campanhas selecionadas
    analysis_result: Optional[dict]          # Resultado da análise

    # ==========================================
    # Metadados de execução
    # ==========================================
    tool_calls_count: int                    # Contador de tool calls no turno
    last_error: Optional[str]                # Último erro ocorrido (se houver)


def create_initial_state(
    config_id: int,
    user_id: int,
    thread_id: str,
    initial_message: Optional[dict] = None
) -> AgentState:
    """
    Cria o estado inicial do agente.

    Args:
        config_id: ID da configuração Facebook Ads
        user_id: ID do usuário
        thread_id: ID da thread de conversa
        initial_message: Mensagem inicial opcional

    Returns:
        Estado inicial do agente
    """
    messages = []
    if initial_message:
        messages.append(initial_message)

    return AgentState(
        messages=messages,
        config_id=config_id,
        user_id=user_id,
        thread_id=thread_id,
        classifications=None,
        recommendations=None,
        anomalies=None,
        forecasts=None,
        current_intent=None,
        selected_campaigns=[],
        analysis_result=None,
        tool_calls_count=0,
        last_error=None,
    )
