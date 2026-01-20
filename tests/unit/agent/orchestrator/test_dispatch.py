"""Testes para dispatch node."""
import sys
import os
import importlib.util

import pytest


# Carregar o modulo diretamente para evitar dependencias de __init__.py
def load_module_direct(module_name: str, relative_path: str):
    """Carrega modulo diretamente pelo path."""
    module_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..', '..', '..',
        relative_path
    )
    module_path = os.path.abspath(module_path)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDispatchImport:
    """Testes de importacao do dispatch."""

    def test_dispatch_agents_import(self):
        """dispatch_agents deve ser importavel."""
        dispatch_module = load_module_direct(
            "dispatch",
            "app/agent/orchestrator/nodes/dispatch.py"
        )
        assert dispatch_module.dispatch_agents is not None

    def test_dispatch_agents_is_callable(self):
        """dispatch_agents deve ser uma funcao."""
        dispatch_module = load_module_direct(
            "dispatch",
            "app/agent/orchestrator/nodes/dispatch.py"
        )
        assert callable(dispatch_module.dispatch_agents)


class TestDispatchReturnsList:
    """Testes para retorno do dispatch_agents."""

    def test_dispatch_returns_list(self):
        """dispatch_agents deve retornar lista."""
        dispatch_module = load_module_direct(
            "dispatch",
            "app/agent/orchestrator/nodes/dispatch.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
        )
        state["required_agents"] = ["classification", "anomaly"]
        state["execution_plan"] = {
            "agents": ["classification", "anomaly"],
            "tasks": {
                "classification": {"description": "Test classification", "context": {}, "priority": 1},
                "anomaly": {"description": "Test anomaly", "context": {}, "priority": 2},
            },
            "parallel": True,
            "timeout": 60
        }

        result = dispatch_module.dispatch_agents(state)
        assert isinstance(result, list)

    def test_dispatch_returns_correct_count(self):
        """dispatch_agents deve retornar um Send para cada agente."""
        dispatch_module = load_module_direct(
            "dispatch",
            "app/agent/orchestrator/nodes/dispatch.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
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

        result = dispatch_module.dispatch_agents(state)
        assert len(result) == 2


class TestDispatchEmptyAgents:
    """Testes para casos de lista vazia de agentes."""

    def test_dispatch_empty_agents_returns_empty_list(self):
        """dispatch_agents deve retornar lista vazia se nao houver agentes."""
        dispatch_module = load_module_direct(
            "dispatch",
            "app/agent/orchestrator/nodes/dispatch.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
        )
        state["required_agents"] = []
        state["execution_plan"] = None

        result = dispatch_module.dispatch_agents(state)
        assert result == []

    def test_dispatch_no_execution_plan_returns_empty_list(self):
        """dispatch_agents deve retornar lista vazia sem execution_plan."""
        dispatch_module = load_module_direct(
            "dispatch",
            "app/agent/orchestrator/nodes/dispatch.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
        )
        state["required_agents"] = ["classification"]
        state["execution_plan"] = None

        result = dispatch_module.dispatch_agents(state)
        assert result == []


class TestDispatchSendObjects:
    """Testes para objetos Send retornados."""

    def test_dispatch_returns_send_objects(self):
        """dispatch_agents deve retornar objetos Send."""
        from langgraph.types import Send

        dispatch_module = load_module_direct(
            "dispatch",
            "app/agent/orchestrator/nodes/dispatch.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
        )
        state["required_agents"] = ["classification"]
        state["execution_plan"] = {
            "agents": ["classification"],
            "tasks": {
                "classification": {"description": "Test", "context": {}, "priority": 1},
            },
            "parallel": False,
            "timeout": 30
        }

        result = dispatch_module.dispatch_agents(state)
        assert len(result) == 1
        assert isinstance(result[0], Send)

    def test_dispatch_send_has_correct_node_name(self):
        """Send deve ter nome de no correto (subagent_<name>)."""
        from langgraph.types import Send

        dispatch_module = load_module_direct(
            "dispatch",
            "app/agent/orchestrator/nodes/dispatch.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
        )
        state["required_agents"] = ["classification", "anomaly"]
        state["execution_plan"] = {
            "agents": ["classification", "anomaly"],
            "tasks": {
                "classification": {"description": "Test", "context": {}, "priority": 1},
                "anomaly": {"description": "Test", "context": {}, "priority": 2},
            },
            "parallel": True,
            "timeout": 60
        }

        result = dispatch_module.dispatch_agents(state)

        # Verificar nomes dos nos
        node_names = [send.node for send in result]
        assert "subagent_classification" in node_names
        assert "subagent_anomaly" in node_names

    def test_dispatch_send_has_subagent_state(self):
        """Send deve conter SubagentState valido."""
        dispatch_module = load_module_direct(
            "dispatch",
            "app/agent/orchestrator/nodes/dispatch.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test-thread"
        )
        state["required_agents"] = ["forecast"]
        state["execution_plan"] = {
            "agents": ["forecast"],
            "tasks": {
                "forecast": {
                    "description": "Forecast task",
                    "context": {"date_range": "7d"},
                    "priority": 5
                },
            },
            "parallel": False,
            "timeout": 45
        }

        result = dispatch_module.dispatch_agents(state)
        assert len(result) == 1

        send = result[0]
        subagent_state = send.arg

        # Verificar campos do SubagentState
        assert "task" in subagent_state
        assert "config_id" in subagent_state
        assert "user_id" in subagent_state
        assert "thread_id" in subagent_state
        assert "messages" in subagent_state

        # Verificar valores
        assert subagent_state["config_id"] == 1
        assert subagent_state["user_id"] == 1
        assert subagent_state["thread_id"] == "test-thread"

    def test_dispatch_task_contains_description(self):
        """Task no SubagentState deve conter descricao."""
        dispatch_module = load_module_direct(
            "dispatch",
            "app/agent/orchestrator/nodes/dispatch.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
        )
        state["required_agents"] = ["recommendation"]
        state["execution_plan"] = {
            "agents": ["recommendation"],
            "tasks": {
                "recommendation": {
                    "description": "Generate recommendations",
                    "context": {},
                    "priority": 3
                },
            },
            "parallel": False,
            "timeout": 30
        }

        result = dispatch_module.dispatch_agents(state)
        send = result[0]
        task = send.arg["task"]

        assert task["description"] == "Generate recommendations"
        assert task["priority"] == 3


class TestDispatchMultipleAgents:
    """Testes para dispatch de multiplos agentes."""

    def test_dispatch_all_valid_agents(self):
        """dispatch_agents deve funcionar com todos os agentes validos."""
        from langgraph.types import Send

        dispatch_module = load_module_direct(
            "dispatch",
            "app/agent/orchestrator/nodes/dispatch.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        all_agents = list(state_module.VALID_AGENTS)
        tasks = {
            agent: {"description": f"Task for {agent}", "context": {}, "priority": i}
            for i, agent in enumerate(all_agents, 1)
        }

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
        )
        state["required_agents"] = all_agents
        state["execution_plan"] = {
            "agents": all_agents,
            "tasks": tasks,
            "parallel": True,
            "timeout": 120
        }

        result = dispatch_module.dispatch_agents(state)

        assert len(result) == len(all_agents)
        for send in result:
            assert isinstance(send, Send)

    def test_dispatch_preserves_messages(self):
        """dispatch_agents deve preservar mensagens no SubagentState."""
        from langchain_core.messages import HumanMessage

        dispatch_module = load_module_direct(
            "dispatch",
            "app/agent/orchestrator/nodes/dispatch.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        messages = [HumanMessage(content="Test message")]
        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test", messages=messages
        )
        state["required_agents"] = ["classification"]
        state["execution_plan"] = {
            "agents": ["classification"],
            "tasks": {
                "classification": {"description": "Test", "context": {}, "priority": 1},
            },
            "parallel": False,
            "timeout": 30
        }

        result = dispatch_module.dispatch_agents(state)
        subagent_state = result[0].arg

        assert len(subagent_state["messages"]) == 1
        assert subagent_state["messages"][0].content == "Test message"
