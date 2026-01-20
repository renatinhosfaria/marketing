"""Estado do Orchestrator Agent.

Define o estado central usado pelo orchestrator para coordenar
os subagentes especialistas.
"""
from typing import TypedDict, Annotated, Sequence, Optional, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# Re-definir AgentResult aqui para evitar import circular via app/agent/__init__.py
# A definicao canonica esta em app/agent/subagents/state.py
class AgentResult(TypedDict):
    """Resultado de execucao de um subagente.

    Nota: Esta e uma copia da definicao em subagents/state.py para evitar
    problemas de import circular. Manter sincronizado se houver mudancas.
    """
    agent_name: str
    success: bool
    data: Optional[dict[str, Any]]
    error: Optional[str]
    duration_ms: int
    tool_calls: list[str]


# Agentes válidos no sistema
VALID_AGENTS = frozenset([
    'classification',
    'anomaly',
    'forecast',
    'recommendation',
    'campaign',
    'analysis'
])

# Mapeamento de intenção do usuário para agentes necessários
INTENT_TO_AGENTS: dict[str, list[str]] = {
    "analyze_performance": ["classification", "campaign"],
    "find_problems": ["anomaly", "classification"],
    "get_recommendations": ["recommendation", "classification"],
    "predict_future": ["forecast"],
    "compare_campaigns": ["analysis", "classification"],
    "full_report": ["classification", "anomaly", "recommendation", "forecast"],
    "troubleshoot": ["anomaly", "recommendation", "campaign"],
    "general": ["classification"],  # fallback
}

# Prioridade de síntese (menor = maior prioridade)
PRIORITY_ORDER: dict[str, int] = {
    "anomaly": 1,         # Problemas primeiro
    "recommendation": 2,   # Ações a tomar
    "classification": 3,   # Contexto de performance
    "forecast": 4,         # Projeções futuras
    "campaign": 5,         # Dados específicos
    "analysis": 6,         # Análises complementares
}


class ExecutionPlan(TypedDict):
    """Plano de execução para subagentes."""
    agents: list[str]
    tasks: dict[str, dict[str, Any]]
    parallel: bool
    timeout: int


class OrchestratorState(TypedDict):
    """Estado central do Orchestrator Agent.

    Attributes:
        messages: Histórico de mensagens da conversa
        thread_id: ID da thread para persistência
        config_id: ID da configuração Facebook Ads
        user_id: ID do usuário autenticado
        user_intent: Intenção detectada do usuário
        required_agents: Lista de subagentes necessários
        execution_plan: Plano detalhado de execução
        agent_results: Resultados coletados dos subagentes
        synthesized_response: Resposta final sintetizada
        confidence_score: Score de confiança da resposta (0-1)
        error: Erro global, se houver
    """
    # Conversa
    messages: Annotated[Sequence[BaseMessage], add_messages]
    thread_id: str
    config_id: int
    user_id: int

    # Planejamento
    user_intent: Optional[str]
    required_agents: list[str]
    execution_plan: Optional[ExecutionPlan]

    # Resultados dos subagentes
    agent_results: dict[str, AgentResult]

    # Resposta final
    synthesized_response: Optional[str]
    confidence_score: float

    # Erro
    error: Optional[str]


def create_initial_orchestrator_state(
    config_id: int,
    user_id: int,
    thread_id: str,
    messages: Sequence[BaseMessage] = None
) -> OrchestratorState:
    """Cria estado inicial do orchestrator.

    Args:
        config_id: ID da configuração Facebook Ads
        user_id: ID do usuário
        thread_id: ID da thread
        messages: Mensagens iniciais (opcional)

    Returns:
        Estado inicial do orchestrator
    """
    return OrchestratorState(
        messages=messages or [],
        thread_id=thread_id,
        config_id=config_id,
        user_id=user_id,
        user_intent=None,
        required_agents=[],
        execution_plan=None,
        agent_results={},
        synthesized_response=None,
        confidence_score=0.0,
        error=None
    )


def get_agents_for_intent(intent: str) -> list[str]:
    """Retorna lista de agentes para uma intenção.

    Args:
        intent: Intenção do usuário

    Returns:
        Lista de nomes de agentes
    """
    return INTENT_TO_AGENTS.get(intent, INTENT_TO_AGENTS["general"])
