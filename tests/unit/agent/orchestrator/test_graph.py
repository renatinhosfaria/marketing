"""Testes para o grafo do Orchestrator."""
import pytest


class TestOrchestratorGraph:
    """Testes para o grafo do orchestrator."""

    def test_build_orchestrator_graph_import(self):
        """build_orchestrator_graph deve ser importavel."""
        from app.agent.orchestrator.graph import build_orchestrator_graph
        assert build_orchestrator_graph is not None

    def test_orchestrator_graph_builds(self):
        """Grafo deve ser construido sem erros."""
        from app.agent.orchestrator.graph import build_orchestrator_graph

        graph = build_orchestrator_graph()
        assert graph is not None

    def test_orchestrator_has_nodes(self):
        """Grafo deve ter nos obrigatorios."""
        from app.agent.orchestrator.graph import build_orchestrator_graph

        graph = build_orchestrator_graph()

        # Verificar que e um grafo compilado
        assert hasattr(graph, 'invoke') or hasattr(graph, 'ainvoke')

    def test_get_orchestrator_import(self):
        """get_orchestrator deve ser importavel."""
        from app.agent.orchestrator.graph import get_orchestrator
        assert get_orchestrator is not None

    def test_get_orchestrator_singleton(self):
        """get_orchestrator deve retornar mesma instancia."""
        from app.agent.orchestrator.graph import get_orchestrator

        g1 = get_orchestrator()
        g2 = get_orchestrator()

        assert g1 is g2

    def test_graph_has_subagent_nodes(self):
        """Grafo deve incluir nos de subagentes."""
        from app.agent.orchestrator.graph import build_orchestrator_graph
        from app.agent.orchestrator.state import VALID_AGENTS

        graph = build_orchestrator_graph()
        graph_data = graph.get_graph()
        node_names = set(graph_data.nodes.keys())

        for agent_name in VALID_AGENTS:
            assert f"subagent_{agent_name}" in node_names
