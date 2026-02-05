"""
Módulo do grafo LangGraph para o agente de tráfego.
"""

from projects.agent.graph.state import AgentState, create_initial_state
from projects.agent.graph.builder import build_agent_graph, create_agent, create_agent_sync
from projects.agent.graph.nodes import (
    classify_intent,
    gather_data,
    call_model,
    call_tools,
    generate_response,
    handle_error,
)
from projects.agent.graph.edges import (
    should_continue,
    route_by_intent,
    after_tools,
    check_data_quality,
)

__all__ = [
    # State
    "AgentState",
    "create_initial_state",
    # Builder
    "build_agent_graph",
    "create_agent",
    "create_agent_sync",
    # Nodes
    "classify_intent",
    "gather_data",
    "call_model",
    "call_tools",
    "generate_response",
    "handle_error",
    # Edges
    "should_continue",
    "route_by_intent",
    "after_tools",
    "check_data_quality",
]
