"""
Arestas condicionais do grafo LangGraph para o agente de tráfego pago.

⚠️ DEPRECATION WARNING:
Este módulo faz parte do agente monolítico LEGADO e está DEPRECADO desde 2026-01-21.
Use o novo sistema multi-agente em app/agent/orchestrator/.

Para habilitar: AGENT_MULTI_AGENT_ENABLED=true
"""

import warnings
from typing import Literal

warnings.warn(
    "app.agent.graph.edges está DEPRECADO. Use app.agent.orchestrator",
    DeprecationWarning,
    stacklevel=2
)
from app.agent.graph.state import AgentState
from app.agent.config import get_agent_settings


settings = get_agent_settings()


def should_continue(state: AgentState) -> Literal["call_tools", "generate_response", "handle_error"]:
    """
    Determina se deve continuar executando tools ou gerar resposta.

    Retorna:
    - "call_tools": Se há tool calls pendentes
    - "generate_response": Se não há mais tool calls
    - "handle_error": Se houve erro
    """
    # Verificar se há erro
    if state.get("last_error"):
        return "handle_error"

    messages = state.get("messages", [])
    if not messages:
        return "generate_response"

    last_message = messages[-1]

    # Verificar se a última mensagem tem tool calls
    tool_calls = []
    if hasattr(last_message, "tool_calls"):
        tool_calls = last_message.tool_calls
    elif isinstance(last_message, dict):
        tool_calls = last_message.get("tool_calls", [])

    # Verificar limite de tool calls
    tool_calls_count = state.get("tool_calls_count", 0)
    if tool_calls_count >= settings.max_tool_calls:
        return "generate_response"

    if tool_calls:
        return "call_tools"

    return "generate_response"


def route_by_intent(state: AgentState) -> Literal["gather_data", "call_model"]:
    """
    Roteia baseado na intenção detectada.

    Para intenções que precisam de dados ML, vai para gather_data.
    Para conversas gerais, vai direto para o modelo.
    """
    intent = state.get("current_intent", "general")

    # Intenções que precisam de dados do ML
    data_intensive_intents = ["analyze", "compare", "recommend", "forecast", "troubleshoot"]

    if intent in data_intensive_intents:
        # Verificar se já temos dados carregados
        has_data = any([
            state.get("classifications"),
            state.get("recommendations"),
            state.get("anomalies"),
            state.get("forecasts"),
        ])

        if not has_data:
            return "gather_data"

    return "call_model"


def after_tools(state: AgentState) -> Literal["call_model", "generate_response"]:
    """
    Determina próximo passo após executar ferramentas.

    Geralmente volta para o modelo processar os resultados das tools.
    """
    # Verificar limite de tool calls
    tool_calls_count = state.get("tool_calls_count", 0)
    if tool_calls_count >= settings.max_tool_calls:
        return "generate_response"

    return "call_model"


def check_data_quality(state: AgentState) -> Literal["call_model", "handle_error"]:
    """
    Verifica qualidade dos dados coletados.

    Se houve erro na coleta, vai para tratamento de erro.
    Caso contrário, prossegue para o modelo.
    """
    if state.get("last_error"):
        return "handle_error"

    return "call_model"
