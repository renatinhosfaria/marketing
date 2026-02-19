"""
Subgraph do Gerente de Operacoes.

Fluxo: START -> propose_action -> execute_action -> END

O propose_action_node contem interrupt() para aprovacao humana.
O execute_action_node executa a acao ou gera relatorio se cancelada.
"""

from langgraph.graph import StateGraph, START, END

from projects.agent.subgraphs.operations_manager.state import (
    OperationsSubgraphState,
    OperationsOutput,
)
from projects.agent.graph.state import AgentInput
from projects.agent.subgraphs.operations_manager.nodes import (
    propose_action_node,
    execute_action_node,
)


def build_operations_graph():
    """Constroi e compila o subgraph do Gerente de Operacoes.

    interrupt() no propose_action_node pausa o grafo para aprovacao.
    O Command(resume=...) do frontend retoma a execucao.
    """
    builder = StateGraph(
        OperationsSubgraphState,
        input_schema=AgentInput,
        output_schema=OperationsOutput,
    )

    builder.add_node("propose_action", propose_action_node)
    builder.add_node("execute_action", execute_action_node)

    builder.add_edge(START, "propose_action")
    builder.add_edge("propose_action", "execute_action")
    builder.add_edge("execute_action", END)

    return builder.compile()
