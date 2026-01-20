"""No dispatch_agents do Orchestrator.

Responsavel por disparar subagentes em paralelo usando Send().
"""
import os
import importlib.util
from typing import Any

from langgraph.types import Send


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


# Carregar subagents/state.py diretamente
_subagent_state_path = os.path.join(
    os.path.dirname(__file__),
    '..', '..', 'subagents', 'state.py'
)
_subagent_state_path = os.path.abspath(_subagent_state_path)

_subagent_spec = importlib.util.spec_from_file_location("subagent_state", _subagent_state_path)
_subagent_module = importlib.util.module_from_spec(_subagent_spec)
_subagent_spec.loader.exec_module(_subagent_module)

create_initial_subagent_state = _subagent_module.create_initial_subagent_state


def dispatch_agents(state: OrchestratorState) -> list[Send]:
    """No que dispara subagentes em paralelo.

    Usa Send() do LangGraph para executar multiplos subagentes
    simultaneamente.

    Args:
        state: Estado atual do orchestrator

    Returns:
        Lista de objetos Send para cada subagente
    """
    required_agents = state.get("required_agents", [])
    execution_plan = state.get("execution_plan")

    if not required_agents:
        return []

    if not execution_plan:
        return []

    sends = []
    tasks = execution_plan.get("tasks", {})

    for agent_name in required_agents:
        task = tasks.get(agent_name, {})

        # Criar estado inicial do subagente
        subagent_state = create_initial_subagent_state(
            task={
                "description": task.get("description", f"Execute {agent_name}"),
                "context": task.get("context", {}),
                "priority": task.get("priority", 10),
            },
            config_id=state.get("config_id", 1),
            user_id=state.get("user_id", 1),
            thread_id=state.get("thread_id", "default"),
            messages=state.get("messages", []),
        )

        # Criar Send para o no do subagente
        sends.append(
            Send(
                f"subagent_{agent_name}",
                subagent_state
            )
        )

    return sends
