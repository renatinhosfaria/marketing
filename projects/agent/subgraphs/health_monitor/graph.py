"""
Subgraph do Monitor de Saude & Anomalias.

Fluxo: START -> fetch_metrics -> detect_anomalies -> diagnose -> END

input_schema=AgentInput: recebe messages + scope (compativel com Send()).
output_schema=HealthOutput: retorna apenas agent_reports (fan-in).
"""

from langgraph.graph import StateGraph, START, END

from projects.agent.subgraphs.health_monitor.state import (
    HealthSubgraphState,
    HealthOutput,
)
from projects.agent.graph.state import AgentInput
from projects.agent.subgraphs.health_monitor.nodes import (
    fetch_metrics_node,
    anomaly_detection_node,
    diagnose_node,
)


def build_health_graph():
    """Constroi e compila o subgraph do Monitor de Saude.

    input_schema=AgentInput garante que o subgraph recebe messages + scope.
    output_schema=HealthOutput garante que retorna apenas agent_reports.
    """
    builder = StateGraph(
        HealthSubgraphState,
        input_schema=AgentInput,
        output_schema=HealthOutput,
    )

    builder.add_node("fetch_metrics", fetch_metrics_node)
    builder.add_node("detect_anomalies", anomaly_detection_node)
    builder.add_node("diagnose", diagnose_node)

    builder.add_edge(START, "fetch_metrics")
    builder.add_edge("fetch_metrics", "detect_anomalies")
    builder.add_edge("detect_anomalies", "diagnose")
    builder.add_edge("diagnose", END)

    return builder.compile()
