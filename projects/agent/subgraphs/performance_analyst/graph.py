"""
Subgraph do Analista de Performance & Impacto.

Fluxo: START -> analyze_metrics -> compare_periods -> generate_report -> END
"""

from langgraph.graph import StateGraph, START, END

from projects.agent.subgraphs.performance_analyst.state import (
    PerformanceSubgraphState,
    PerformanceOutput,
)
from projects.agent.graph.state import AgentInput
from projects.agent.subgraphs.performance_analyst.nodes import (
    analyze_metrics_node,
    compare_periods_node,
    generate_report_node,
)


def build_performance_graph():
    """Constroi e compila o subgraph do Analista de Performance."""
    builder = StateGraph(
        PerformanceSubgraphState,
        input_schema=AgentInput,
        output_schema=PerformanceOutput,
    )

    builder.add_node("analyze_metrics", analyze_metrics_node)
    builder.add_node("compare_periods", compare_periods_node)
    builder.add_node("generate_report", generate_report_node)

    builder.add_edge(START, "analyze_metrics")
    builder.add_edge("analyze_metrics", "compare_periods")
    builder.add_edge("compare_periods", "generate_report")
    builder.add_edge("generate_report", END)

    return builder.compile()
