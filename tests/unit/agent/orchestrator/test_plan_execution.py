"""Testes para plan_execution node."""
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


class TestPlanExecution:
    """Testes para o no plan_execution."""

    def test_plan_execution_import(self):
        """plan_execution deve ser importavel."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )
        assert plan_execution_module.plan_execution is not None

    def test_create_execution_plan_import(self):
        """create_execution_plan deve ser importavel."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )
        assert plan_execution_module.create_execution_plan is not None

    def test_agent_task_descriptions_exists(self):
        """AGENT_TASK_DESCRIPTIONS deve existir."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )
        assert hasattr(plan_execution_module, 'AGENT_TASK_DESCRIPTIONS')
        assert isinstance(plan_execution_module.AGENT_TASK_DESCRIPTIONS, dict)

    def test_agent_priorities_exists(self):
        """AGENT_PRIORITIES deve existir."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )
        assert hasattr(plan_execution_module, 'AGENT_PRIORITIES')
        assert isinstance(plan_execution_module.AGENT_PRIORITIES, dict)

    def test_agent_timeouts_exists(self):
        """AGENT_TIMEOUTS deve existir."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )
        assert hasattr(plan_execution_module, 'AGENT_TIMEOUTS')
        assert isinstance(plan_execution_module.AGENT_TIMEOUTS, dict)


class TestCreateExecutionPlan:
    """Testes para create_execution_plan."""

    def test_create_execution_plan_analyze(self):
        """Deve criar plano para analyze_performance."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )
        plan = plan_execution_module.create_execution_plan("analyze_performance", config_id=1)

        assert plan["parallel"] is True
        assert "classification" in plan["agents"]
        assert "campaign" in plan["agents"]
        assert len(plan["tasks"]) == len(plan["agents"])

    def test_create_execution_plan_full_report(self):
        """Deve criar plano completo para full_report."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )
        plan = plan_execution_module.create_execution_plan("full_report", config_id=1)

        assert len(plan["agents"]) == 4
        assert "classification" in plan["agents"]
        assert "anomaly" in plan["agents"]
        assert "recommendation" in plan["agents"]
        assert "forecast" in plan["agents"]

    def test_create_execution_plan_find_problems(self):
        """Deve criar plano para find_problems."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )
        plan = plan_execution_module.create_execution_plan("find_problems", config_id=1)

        assert "anomaly" in plan["agents"]
        assert "classification" in plan["agents"]

    def test_create_execution_plan_has_tasks(self):
        """Plano deve ter tasks para cada agente."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )
        plan = plan_execution_module.create_execution_plan("find_problems", config_id=1)

        for agent in plan["agents"]:
            assert agent in plan["tasks"]
            task = plan["tasks"][agent]
            assert "description" in task
            assert "context" in task
            assert "priority" in task

    def test_create_execution_plan_timeout(self):
        """Plano deve ter timeout baseado nos agentes."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )
        plan = plan_execution_module.create_execution_plan("analyze_performance", config_id=1)

        assert "timeout" in plan
        assert plan["timeout"] > 0

    def test_create_execution_plan_single_agent_not_parallel(self):
        """Plano com 1 agente nao deve ser paralelo."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )
        plan = plan_execution_module.create_execution_plan("predict_future", config_id=1)

        # predict_future usa apenas forecast
        assert len(plan["agents"]) == 1
        assert plan["parallel"] is False

    def test_create_execution_plan_with_context(self):
        """Plano deve incluir contexto adicional nas tasks."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )
        context = {"campaign_id": "123", "date_range": "7d"}
        plan = plan_execution_module.create_execution_plan(
            "analyze_performance",
            config_id=1,
            context=context
        )

        for agent in plan["agents"]:
            task = plan["tasks"][agent]
            assert task["context"]["config_id"] == 1
            assert task["context"]["campaign_id"] == "123"
            assert task["context"]["date_range"] == "7d"

    def test_create_execution_plan_unknown_intent_uses_general(self):
        """Intencao desconhecida deve usar fallback general."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )
        plan = plan_execution_module.create_execution_plan("unknown_intent_xyz", config_id=1)

        # general usa apenas classification
        assert "classification" in plan["agents"]


class TestPlanExecutionNode:
    """Testes para plan_execution node function."""

    def test_plan_execution_returns_dict(self):
        """plan_execution deve retornar dict."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )
        state = {
            "intent": "analyze_performance",
            "config_id": 1,
            "context": {}
        }
        result = plan_execution_module.plan_execution(state)

        assert isinstance(result, dict)
        assert "execution_plan" in result

    def test_plan_execution_with_state(self):
        """plan_execution deve usar valores do estado."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )
        state = {
            "intent": "full_report",
            "config_id": 5,
            "context": {"test": "value"}
        }
        result = plan_execution_module.plan_execution(state)

        plan = result["execution_plan"]
        assert len(plan["agents"]) == 4
        # Verifica que config_id foi passado para tasks
        for agent in plan["agents"]:
            assert plan["tasks"][agent]["context"]["config_id"] == 5

    def test_plan_execution_default_values(self):
        """plan_execution deve usar valores padrao quando ausentes."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )
        # Estado minimo, sem intent explicito
        state = {}
        result = plan_execution_module.plan_execution(state)

        plan = result["execution_plan"]
        # Deve usar "general" como fallback
        assert "classification" in plan["agents"]


class TestAgentDescriptions:
    """Testes para AGENT_TASK_DESCRIPTIONS."""

    def test_all_agents_have_descriptions(self):
        """Todos os agentes devem ter descricoes."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        for agent in state_module.VALID_AGENTS:
            assert agent in plan_execution_module.AGENT_TASK_DESCRIPTIONS, \
                f"Agente {agent} sem descricao"
            assert len(plan_execution_module.AGENT_TASK_DESCRIPTIONS[agent]) > 0

    def test_descriptions_are_strings(self):
        """Descricoes devem ser strings."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )

        for agent, desc in plan_execution_module.AGENT_TASK_DESCRIPTIONS.items():
            assert isinstance(desc, str), f"Descricao de {agent} deve ser string"


class TestAgentPriorities:
    """Testes para AGENT_PRIORITIES."""

    def test_all_agents_have_priorities(self):
        """Todos os agentes devem ter prioridades."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        for agent in state_module.VALID_AGENTS:
            assert agent in plan_execution_module.AGENT_PRIORITIES, \
                f"Agente {agent} sem prioridade"

    def test_priorities_are_positive_integers(self):
        """Prioridades devem ser inteiros positivos."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )

        for agent, priority in plan_execution_module.AGENT_PRIORITIES.items():
            assert isinstance(priority, int), f"Prioridade de {agent} deve ser int"
            assert priority >= 1, f"Prioridade de {agent} deve ser >= 1"


class TestAgentTimeouts:
    """Testes para AGENT_TIMEOUTS."""

    def test_all_agents_have_timeouts(self):
        """Todos os agentes devem ter timeouts."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        for agent in state_module.VALID_AGENTS:
            assert agent in plan_execution_module.AGENT_TIMEOUTS, \
                f"Agente {agent} sem timeout"

    def test_timeouts_are_positive_integers(self):
        """Timeouts devem ser inteiros positivos."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )

        for agent, timeout in plan_execution_module.AGENT_TIMEOUTS.items():
            assert isinstance(timeout, int), f"Timeout de {agent} deve ser int"
            assert timeout > 0, f"Timeout de {agent} deve ser > 0"

    def test_timeout_calculation_uses_max(self):
        """Timeout do plano deve ser o maximo dos timeouts dos agentes."""
        plan_execution_module = load_module_direct(
            "plan_execution",
            "app/agent/orchestrator/nodes/plan_execution.py"
        )

        # full_report usa 4 agentes
        plan = plan_execution_module.create_execution_plan("full_report", config_id=1)

        # Calcular o timeout esperado manualmente
        expected_timeout = max(
            plan_execution_module.AGENT_TIMEOUTS.get(a, 30)
            for a in plan["agents"]
        )

        assert plan["timeout"] == expected_timeout
