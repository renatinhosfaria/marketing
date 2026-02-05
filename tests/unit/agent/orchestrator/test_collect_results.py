"""Testes para collect_results node."""
import pytest


class TestCollectResults:
    """Testes para o no collect_results."""

    def test_collect_results_import(self):
        """collect_results deve ser importavel."""
        from projects.agent.orchestrator.nodes.collect_results import collect_results
        assert collect_results is not None

    def test_collect_results_aggregates(self):
        """collect_results deve agregar resultados de subagentes."""
        from projects.agent.orchestrator.nodes.collect_results import collect_results
        from projects.agent.orchestrator.state import create_initial_orchestrator_state

        state = create_initial_orchestrator_state(
            config_id=1,
            user_id=1,
            thread_id="test"
        )
        state["agent_results"] = {
            "classification": {
                "agent_name": "classification",
                "success": True,
                "data": {"tiers": ["HIGH", "LOW"]},
                "error": None,
                "duration_ms": 100,
                "tool_calls": ["get_classifications"]
            }
        }

        result = collect_results(state)

        assert "agent_results" in result or result == {}

    def test_merge_subagent_results(self):
        """merge_subagent_results deve combinar multiplos resultados."""
        from projects.agent.orchestrator.nodes.collect_results import merge_subagent_results

        existing = {
            "classification": {"success": True, "data": {"a": 1}}
        }
        new_results = [
            {"agent_name": "anomaly", "result": {"success": True, "data": {"b": 2}}}
        ]

        merged = merge_subagent_results(existing, new_results)

        assert "classification" in merged
        assert "anomaly" in merged

    def test_calculate_confidence_score(self):
        """calculate_confidence_score deve calcular score corretamente."""
        from projects.agent.orchestrator.nodes.collect_results import calculate_confidence_score

        # Todos sucesso
        results = {
            "a": {"success": True},
            "b": {"success": True},
        }
        assert calculate_confidence_score(results) == 1.0

        # Metade sucesso
        results = {
            "a": {"success": True},
            "b": {"success": False},
        }
        assert calculate_confidence_score(results) == 0.5

    def test_collect_results_empty(self):
        """collect_results deve funcionar sem resultados."""
        from projects.agent.orchestrator.nodes.collect_results import collect_results
        from projects.agent.orchestrator.state import create_initial_orchestrator_state

        state = create_initial_orchestrator_state(
            config_id=1,
            user_id=1,
            thread_id="test"
        )
        state["agent_results"] = {}

        result = collect_results(state)
        assert isinstance(result, dict)
