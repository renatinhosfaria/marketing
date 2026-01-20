"""Estado compartilhado dos subagentes.

Define os tipos de estado usados por todos os subagentes especialistas
do sistema multi-agente.
"""
from typing import TypedDict, Annotated, Sequence, Optional, Any
from datetime import datetime
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SubagentTask(TypedDict):
    """Tarefa delegada para um subagente."""
    description: str
    context: dict[str, Any]
    priority: int  # 1 = highest


class AgentResult(TypedDict):
    """Resultado de execucao de um subagente."""
    agent_name: str
    success: bool
    data: Optional[dict[str, Any]]
    error: Optional[str]
    duration_ms: int
    tool_calls: list[str]


class SubagentState(TypedDict):
    """Estado interno de um subagente durante execucao.

    Attributes:
        messages: Historico de mensagens (com reducer add_messages)
        task: Tarefa delegada pelo orchestrator
        config_id: ID da configuracao Facebook Ads
        user_id: ID do usuario
        thread_id: ID da thread para persistencia
        result: Resultado parcial/final da analise
        error: Erro ocorrido durante execucao
        tool_calls_count: Contador de chamadas de tools
        started_at: Timestamp de inicio
        completed_at: Timestamp de conclusao
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    task: SubagentTask
    config_id: int
    user_id: int
    thread_id: str
    result: Optional[dict[str, Any]]
    error: Optional[str]
    tool_calls_count: int
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


def create_initial_subagent_state(
    task: SubagentTask,
    config_id: int,
    user_id: int,
    thread_id: str,
    messages: Sequence[BaseMessage] = None
) -> SubagentState:
    """Cria estado inicial para um subagente.

    Args:
        task: Tarefa a executar
        config_id: ID da configuracao Facebook Ads
        user_id: ID do usuario
        thread_id: ID da thread
        messages: Mensagens iniciais (opcional)

    Returns:
        Estado inicial do subagente
    """
    return SubagentState(
        messages=messages or [],
        task=task,
        config_id=config_id,
        user_id=user_id,
        thread_id=thread_id,
        result=None,
        error=None,
        tool_calls_count=0,
        started_at=datetime.utcnow(),
        completed_at=None
    )
