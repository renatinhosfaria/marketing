"""Grafo principal do Orchestrator Agent.

Constroi o grafo LangGraph que coordena todos os subagentes.
"""
import threading

from langgraph.graph import StateGraph, START, END

from projects.agent.orchestrator.state import (
    OrchestratorState,
    VALID_AGENTS
)
from projects.agent.orchestrator.nodes import (
    load_memory,
    parse_request,
    plan_execution,
    dispatch_agents,
    create_subagent_node,
    collect_results,
    synthesize,
    persist_memory,
)
from shared.core.logging import get_logger

logger = get_logger("orchestrator.graph")

# Singleton do grafo
_orchestrator_graph = None


def build_orchestrator_graph(checkpointer=None) -> StateGraph:
    """Constroi o grafo do Orchestrator.

    O grafo segue o fluxo:
    1. parse_request - Detecta intencao
    2. plan_execution - Planeja quais agentes usar
    3. dispatch_agents - Dispara subagentes em paralelo (via Send)
    4. subagent_* - Nos dos subagentes (executam em paralelo)
    5. collect_results - Agrega resultados
    6. synthesize - Gera resposta final

    Args:
        checkpointer: Checkpointer para persistencia de estado (opcional)

    Returns:
        Grafo compilado
    """
    logger.info("Construindo grafo do Orchestrator")

    # Criar grafo (OrchestratorState já possui reducer em agent_results)
    graph = StateGraph(OrchestratorState)

    # Adicionar nos principais
    graph.add_node("load_memory", load_memory)
    graph.add_node("parse_request", parse_request)
    graph.add_node("plan_execution", plan_execution)
    graph.add_node("collect_results", collect_results)
    graph.add_node("synthesize", synthesize)
    graph.add_node("persist_memory", persist_memory)

    # Adicionar nos de subagentes
    for agent_name in VALID_AGENTS:
        node_name = f"subagent_{agent_name}"
        graph.add_node(node_name, create_subagent_node(agent_name))

    # Adicionar arestas sequenciais
    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "parse_request")
    graph.add_edge("parse_request", "plan_execution")

    # dispatch_agents retorna Send() que conecta aos subagentes
    graph.add_conditional_edges(
        "plan_execution",
        dispatch_agents,
        # Map de subagente -> no de coleta apos execucao
        {f"subagent_{name}": f"subagent_{name}" for name in VALID_AGENTS}
    )

    # Todos os subagentes convergem para collect_results
    for agent_name in VALID_AGENTS:
        node_name = f"subagent_{agent_name}"
        graph.add_edge(node_name, "collect_results")

    # Fluxo final
    graph.add_edge("collect_results", "synthesize")
    graph.add_edge("synthesize", "persist_memory")
    graph.add_edge("persist_memory", END)

    logger.info("Grafo do Orchestrator construido com sucesso")

    return graph.compile(checkpointer=checkpointer)


_graph_lock = threading.Lock()


def get_orchestrator(checkpointer=None) -> StateGraph:
    """Retorna instancia singleton do grafo do Orchestrator.

    Thread-safe com double-checked locking.

    Args:
        checkpointer: Checkpointer para persistencia de estado (opcional).
                      Se fornecido e o grafo ainda nao foi criado, sera usado
                      na construcao. Se o grafo ja existe, e ignorado.

    Returns:
        Grafo compilado do Orchestrator
    """
    global _orchestrator_graph

    if _orchestrator_graph is None:
        with _graph_lock:
            if _orchestrator_graph is None:
                _orchestrator_graph = build_orchestrator_graph(
                    checkpointer=checkpointer
                )

    return _orchestrator_graph


def reset_orchestrator():
    """Reseta o singleton do orchestrator (para testes)."""
    global _orchestrator_graph
    _orchestrator_graph = None
