"""
Compilacao do grafo principal (SuperGraph).

build_supervisor_graph(): constroi o StateGraph com Supervisor, 6 agentes e Synthesizer.
compile_graph(): compila com checkpointer e store para persistencia.
"""

import time

from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from langchain_core.runnables import RunnableConfig
import structlog
import asyncio

from projects.agent.observability.metrics import agent_subgraph_duration

from projects.agent.graph.state import SupervisorState
from projects.agent.graph.supervisor import supervisor_node
from projects.agent.graph.synthesizer import synthesizer_node
from projects.agent.subgraphs.health_monitor.graph import build_health_graph
from projects.agent.subgraphs.performance_analyst.graph import build_performance_graph
from projects.agent.subgraphs.creative_specialist.graph import build_creative_graph
from projects.agent.subgraphs.audience_specialist.node import audience_node
from projects.agent.subgraphs.forecast_scientist.node import forecast_node
from projects.agent.subgraphs.operations_manager.graph import build_operations_graph

logger = structlog.get_logger()

# Retry policy para nos que chamam LLM
LLM_RETRY = RetryPolicy(max_attempts=2, initial_interval=2.0)

# Retry para chamadas HTTP dentro dos subgraphs
HTTP_MAX_ATTEMPTS = 3


def _safe_agent_wrapper(agent_id: str, agent_runnable):
    """Wrapper que garante producao de AgentReport mesmo quando o subgraph falha.

    Sem este wrapper, uma excecao no subgraph mataria o Send() worker sem
    produzir output — o synthesizer nao saberia que o agente falhou.
    Com o wrapper, excecoes produzem AgentReport(status="error") no ultimo retry.

    Compativel com:
    - subgraph compilado (tem `.ainvoke()`)
    - no simples async (callable, sem `.ainvoke()`)

    Store: Quando agent_runnable e um subgraph compilado (.ainvoke()),
    o LangGraph usa o config para injetar o store nos nos internos que pedem
    *, store: BaseStore na assinatura. O wrapper nao precisa receber ou
    propagar store explicitamente.
    """
    async def wrapped(state: dict, config: RunnableConfig):
        async def _invoke_once():
            if hasattr(agent_runnable, "ainvoke"):
                return await agent_runnable.ainvoke(state, config=config)
            return await agent_runnable(state, config)

        start = time.monotonic()
        for attempt in range(1, HTTP_MAX_ATTEMPTS + 1):
            try:
                result = await _invoke_once()
                agent_subgraph_duration.labels(agent_id=agent_id).observe(
                    time.monotonic() - start,
                )
                return result
            except Exception as e:
                if attempt < HTTP_MAX_ATTEMPTS:
                    backoff_s = 2 ** (attempt - 1)  # 1s, 2s
                    logger.warning(
                        "agent.subgraph_retry",
                        agent_id=agent_id,
                        attempt=attempt,
                        backoff_s=backoff_s,
                        error=str(e),
                    )
                    await asyncio.sleep(backoff_s)
                    continue
                agent_subgraph_duration.labels(agent_id=agent_id).observe(
                    time.monotonic() - start,
                )
                logger.error(
                    "agent.subgraph_failed",
                    agent_id=agent_id,
                    error=str(e),
                )
                return {"agent_reports": [{
                    "agent_id": agent_id,
                    "status": "error",
                    "summary": f"Agente {agent_id} falhou: {type(e).__name__}: {e}",
                    "data": {"error_code": "SUBGRAPH_EXCEPTION", "retryable": False},
                    "confidence": 0.0,
                }]}

    wrapped.__name__ = f"{agent_id}_safe"
    return wrapped


def build_supervisor_graph():
    """Constroi o SuperGraph principal.

    Fluxo:
      START -> supervisor -> [Send() fan-out] -> agentes -> synthesizer -> END

    O supervisor_node retorna List[Send()] para fan-out paralelo.
    Send() bypassa o sistema de arestas — nao precisa de add_conditional_edges.
    Todos os nos despachados via Send() rodam no mesmo super-step (paralelo).
    Apos TODOS completarem, as arestas levam ao synthesizer (proximo super-step).
    """
    health_subgraph = build_health_graph()
    performance_subgraph = build_performance_graph()
    creative_subgraph = build_creative_graph()
    operations_subgraph = build_operations_graph()

    builder = StateGraph(SupervisorState)

    # Nos — subgraphs sao wrappados com _safe_agent_wrapper
    builder.add_node("supervisor", supervisor_node, retry=LLM_RETRY)
    builder.add_node(
        "health_monitor",
        _safe_agent_wrapper("health_monitor", health_subgraph),
    )
    builder.add_node(
        "performance_analyst",
        _safe_agent_wrapper("performance_analyst", performance_subgraph),
    )
    builder.add_node(
        "creative_specialist",
        _safe_agent_wrapper("creative_specialist", creative_subgraph),
    )
    builder.add_node(
        "audience_specialist",
        _safe_agent_wrapper("audience_specialist", audience_node),
    )
    builder.add_node(
        "forecast_scientist",
        _safe_agent_wrapper("forecast_scientist", forecast_node),
    )
    builder.add_node(
        "operations_manager",
        _safe_agent_wrapper("operations_manager", operations_subgraph),
    )
    builder.add_node("synthesizer", synthesizer_node, retry=LLM_RETRY)

    # Edges
    builder.add_edge(START, "supervisor")

    # Fan-in: todos os agentes -> Synthesizer
    for agent in [
        "health_monitor",
        "performance_analyst",
        "creative_specialist",
        "audience_specialist",
        "forecast_scientist",
        "operations_manager",
    ]:
        builder.add_edge(agent, "synthesizer")

    builder.add_edge("synthesizer", END)

    return builder


def compile_graph(checkpointer, store):
    """Compila o grafo com persistencia (checkpointer + store)."""
    builder = build_supervisor_graph()
    return builder.compile(checkpointer=checkpointer, store=store)
