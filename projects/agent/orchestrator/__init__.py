"""Orchestrator Agent do sistema multi-agente.

O Orchestrator é responsável por:
- Interpretar a intenção do usuário
- Selecionar subagentes necessários
- Disparar execução em paralelo
- Coletar e sintetizar resultados
"""
from projects.agent.orchestrator.state import (
    OrchestratorState,
    ExecutionPlan,
    INTENT_TO_AGENTS,
    VALID_AGENTS,
    PRIORITY_ORDER,
    create_initial_orchestrator_state,
    get_agents_for_intent
)
from projects.agent.orchestrator.graph import (
    build_orchestrator_graph,
    get_orchestrator,
    reset_orchestrator
)
from projects.agent.orchestrator.prompts import (
    get_orchestrator_prompt,
    get_synthesis_prompt
)

__all__ = [
    # State
    "OrchestratorState",
    "ExecutionPlan",
    # Mappings
    "INTENT_TO_AGENTS",
    "VALID_AGENTS",
    "PRIORITY_ORDER",
    # Functions
    "create_initial_orchestrator_state",
    "get_agents_for_intent",
    # Graph
    "build_orchestrator_graph",
    "get_orchestrator",
    "reset_orchestrator",
    # Prompts
    "get_orchestrator_prompt",
    "get_synthesis_prompt",
]
