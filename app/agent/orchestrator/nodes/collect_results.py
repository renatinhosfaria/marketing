"""No collect_results do Orchestrator.

Responsavel por coletar resultados dos subagentes apos execucao paralela.
"""
import os
import importlib.util
from typing import Any, Optional
from datetime import datetime


# Carregar state.py diretamente para evitar problemas de import circular
_state_path = os.path.join(
    os.path.dirname(__file__),
    '..', 'state.py'
)
_state_path = os.path.abspath(_state_path)

_spec = importlib.util.spec_from_file_location("orchestrator_state", _state_path)
_state_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_state_module)

OrchestratorState = _state_module.OrchestratorState
AgentResult = _state_module.AgentResult


# Carregar subagents/state.py diretamente
_subagent_state_path = os.path.join(
    os.path.dirname(__file__),
    '..', '..', 'subagents', 'state.py'
)
_subagent_state_path = os.path.abspath(_subagent_state_path)

_subagent_spec = importlib.util.spec_from_file_location("subagent_state", _subagent_state_path)
_subagent_module = importlib.util.module_from_spec(_subagent_spec)
_subagent_spec.loader.exec_module(_subagent_module)

SubagentState = _subagent_module.SubagentState


def _calculate_duration_ms(
    started_at: Optional[datetime],
    completed_at: Optional[datetime]
) -> int:
    """Calcula duracao em milissegundos entre dois timestamps.

    Args:
        started_at: Timestamp de inicio
        completed_at: Timestamp de conclusao

    Returns:
        Duracao em milissegundos, ou 0 se timestamps invalidos
    """
    if started_at is None or completed_at is None:
        return 0

    try:
        delta = completed_at - started_at
        # Converter timedelta para milissegundos
        return int(delta.total_seconds() * 1000)
    except Exception:
        return 0


def _create_tool_calls_list(tool_calls_count: int) -> list[str]:
    """Cria lista de tool_calls com base no contador.

    Nota: Como nao temos os nomes especificos das tools chamadas,
    criamos uma lista com placeholders baseados no count.

    Args:
        tool_calls_count: Numero de chamadas de tools

    Returns:
        Lista de strings representando as tool calls
    """
    if tool_calls_count <= 0:
        return []
    return [f"tool_call_{i+1}" for i in range(tool_calls_count)]


def convert_subagent_to_result(
    subagent_state: dict[str, Any],
    agent_name: str
) -> AgentResult:
    """Converte SubagentState para formato AgentResult.

    Args:
        subagent_state: Estado do subagente apos execucao
        agent_name: Nome do agente (ex: 'classification', 'anomaly')

    Returns:
        AgentResult formatado para o OrchestratorState
    """
    # Extrair campos com valores padrao para campos ausentes
    result_data = subagent_state.get("result")
    error = subagent_state.get("error")
    started_at = subagent_state.get("started_at")
    completed_at = subagent_state.get("completed_at")
    tool_calls_count = subagent_state.get("tool_calls_count", 0)

    # Determinar sucesso: tem resultado (mesmo vazio) e nao tem erro
    success = (result_data is not None) and (error is None)

    # Calcular duracao
    duration_ms = _calculate_duration_ms(started_at, completed_at)

    # Criar lista de tool_calls
    tool_calls = _create_tool_calls_list(tool_calls_count)

    return AgentResult(
        agent_name=agent_name,
        success=success,
        data=result_data,
        error=error,
        duration_ms=duration_ms,
        tool_calls=tool_calls
    )


def collect_results(
    state: OrchestratorState,
    subagent_states: Optional[list[dict[str, Any]]] = None
) -> dict[str, Any]:
    """No que coleta resultados dos subagentes apos execucao paralela.

    Este no e chamado depois que todos os subagentes terminaram sua execucao.
    Ele consolida os resultados no formato esperado pelo OrchestratorState.

    Args:
        state: Estado atual do orchestrator
        subagent_states: Lista de estados dos subagentes apos execucao.
            Cada estado deve conter 'agent_name' para identificacao.

    Returns:
        Dict com 'agent_results' atualizado para merge no estado
    """
    # Obter resultados existentes (preservar)
    existing_results = dict(state.get("agent_results", {}))

    # Se nao ha estados de subagentes, retornar estado atual
    if subagent_states is None or len(subagent_states) == 0:
        return {"agent_results": existing_results}

    # Processar cada estado de subagente
    for subagent_state in subagent_states:
        try:
            # Extrair nome do agente
            agent_name = subagent_state.get("agent_name")
            if agent_name is None:
                # Tentar extrair do task se disponivel
                task = subagent_state.get("task", {})
                agent_name = task.get("agent_name", "unknown")

            # Converter para AgentResult
            agent_result = convert_subagent_to_result(subagent_state, agent_name)

            # Adicionar/sobrescrever no dicionario de resultados
            existing_results[agent_name] = agent_result

        except Exception as e:
            # Log do erro mas continua processando outros resultados
            # Em producao, usar logger apropriado
            agent_name = subagent_state.get("agent_name", "unknown")
            existing_results[agent_name] = AgentResult(
                agent_name=agent_name,
                success=False,
                data=None,
                error=f"Error collecting result: {str(e)}",
                duration_ms=0,
                tool_calls=[]
            )

    return {"agent_results": existing_results}
