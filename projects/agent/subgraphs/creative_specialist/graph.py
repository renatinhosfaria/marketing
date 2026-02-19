"""
Subgraph do Especialista em Criativos.

Fluxo: START -> fetch_ads -> analyze_fatigue -> recommend -> END
"""

from langgraph.graph import StateGraph, START, END

from projects.agent.subgraphs.creative_specialist.state import (
    CreativeSubgraphState,
    CreativeOutput,
)
from projects.agent.graph.state import AgentInput
from projects.agent.subgraphs.creative_specialist.nodes import (
    fetch_ads_node,
    analyze_fatigue_node,
    recommend_node,
)


def build_creative_graph():
    """Constroi e compila o subgraph do Especialista em Criativos."""
    builder = StateGraph(
        CreativeSubgraphState,
        input_schema=AgentInput,
        output_schema=CreativeOutput,
    )

    builder.add_node("fetch_ads", fetch_ads_node)
    builder.add_node("analyze_fatigue", analyze_fatigue_node)
    builder.add_node("recommend", recommend_node)

    builder.add_edge(START, "fetch_ads")
    builder.add_edge("fetch_ads", "analyze_fatigue")
    builder.add_edge("analyze_fatigue", "recommend")
    builder.add_edge("recommend", END)

    return builder.compile()
