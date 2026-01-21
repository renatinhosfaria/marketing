"""Testes para dispatch node."""
import pytest
from unittest.mock import Mock


class TestDispatch:
    """Testes para o no dispatch."""

    def test_dispatch_import(self):
        """dispatch_agents deve ser importavel."""
        from app.agent.orchestrator.nodes.dispatch import dispatch_agents
        assert dispatch_agents is not None

    def test_dispatch_returns_send_list(self):
        """dispatch_agents deve retornar lista de Send."""
        from app.agent.orchestrator.nodes.dispatch import dispatch_agents
        from app.agent.orchestrator.state import create_initial_orchestrator_state

        state = create_initial_orchestrator_state(
            config_id=1,
            user_id=1,
            thread_id="test"
        )
        state["required_agents"] = ["classification", "anomaly"]
        state["execution_plan"] = {
            "agents": ["classification", "anomaly"],
            "tasks": {
                "classification": {"description": "Test", "context": {}, "priority": 1},
                "anomaly": {"description": "Test", "context": {}, "priority": 1},
            },
            "parallel": True,
            "timeout": 60
        }

        result = dispatch_agents(state)

        assert isinstance(result, list)
        assert len(result) == 2

    def test_dispatch_creates_correct_send_objects(self):
        """dispatch_agents deve criar objetos Send corretos."""
        from app.agent.orchestrator.nodes.dispatch import dispatch_agents
        from app.agent.orchestrator.state import create_initial_orchestrator_state
        from langgraph.constants import Send

        state = create_initial_orchestrator_state(
            config_id=1,
            user_id=1,
            thread_id="test"
        )
        state["required_agents"] = ["classification"]
        state["execution_plan"] = {
            "agents": ["classification"],
            "tasks": {
                "classification": {
                    "description": "Analyze classification",
                    "context": {"config_id": 1},
                    "priority": 1
                },
            },
            "parallel": True,
            "timeout": 60
        }

        result = dispatch_agents(state)

        assert len(result) == 1
        assert isinstance(result[0], Send)

    def test_dispatch_empty_agents(self):
        """dispatch_agents deve retornar lista vazia se nao houver agentes."""
        from app.agent.orchestrator.nodes.dispatch import dispatch_agents
        from app.agent.orchestrator.state import create_initial_orchestrator_state

        state = create_initial_orchestrator_state(
            config_id=1,
            user_id=1,
            thread_id="test"
        )
        state["required_agents"] = []
        state["execution_plan"] = None

        result = dispatch_agents(state)

        assert result == []

    def test_create_subagent_node_import(self):
        """create_subagent_node deve ser importavel."""
        from app.agent.orchestrator.nodes.dispatch import create_subagent_node

        assert callable(create_subagent_node)
