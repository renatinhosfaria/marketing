"""No plan_execution do Orchestrator.

Responsavel por criar o plano de execucao baseado na intencao detectada.
"""
import os
import sys
import importlib.util
from typing import Any


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
ExecutionPlan = _state_module.ExecutionPlan
get_agents_for_intent = _state_module.get_agents_for_intent


# Descricoes de tarefas por agente
AGENT_TASK_DESCRIPTIONS: dict[str, str] = {
    "classification": "Analisar classificacao de performance das campanhas por tier",
    "anomaly": "Identificar anomalias e problemas nas campanhas",
    "forecast": "Gerar previsoes de CPL e leads para os proximos dias",
    "recommendation": "Fornecer recomendacoes de otimizacao priorizadas",
    "campaign": "Coletar dados detalhados das campanhas",
    "analysis": "Realizar analises avancadas e comparacoes",
}

# Prioridades por agente (menor = maior prioridade)
AGENT_PRIORITIES: dict[str, int] = {
    "anomaly": 1,
    "classification": 2,
    "recommendation": 3,
    "forecast": 4,
    "campaign": 5,
    "analysis": 6,
}

# Timeouts por agente (em segundos)
AGENT_TIMEOUTS: dict[str, int] = {
    "classification": 30,
    "anomaly": 30,
    "forecast": 45,
    "recommendation": 30,
    "campaign": 20,
    "analysis": 45,
}


def create_execution_plan(
    intent: str,
    config_id: int,
    context: dict[str, Any] = None
) -> ExecutionPlan:
    """Cria plano de execucao para uma intencao.

    Args:
        intent: Intencao detectada do usuario
        config_id: ID da configuracao Facebook Ads
        context: Contexto adicional (opcional)

    Returns:
        ExecutionPlan com agentes, tasks, parallel e timeout
    """
    agents = get_agents_for_intent(intent)
    context = context or {}

    tasks: dict[str, dict[str, Any]] = {}
    for agent in agents:
        tasks[agent] = {
            "description": AGENT_TASK_DESCRIPTIONS.get(agent, f"Executar {agent}"),
            "context": {"config_id": config_id, **context},
            "priority": AGENT_PRIORITIES.get(agent, 10),
        }

    # Calcular timeout total (max dos timeouts dos agentes selecionados)
    timeout = max(AGENT_TIMEOUTS.get(a, 30) for a in agents) if agents else 60

    return {
        "agents": agents,
        "tasks": tasks,
        "parallel": len(agents) > 1,
        "timeout": timeout,
    }


def plan_execution(state: OrchestratorState) -> dict:
    """No que cria o plano de execucao.

    Args:
        state: Estado atual do orchestrator

    Returns:
        Dict com execution_plan para atualizar o estado
    """
    intent = state.get("intent", "general")
    config_id = state.get("config_id", 1)
    context = state.get("context", {})

    plan = create_execution_plan(intent, config_id, context)

    return {"execution_plan": plan}
