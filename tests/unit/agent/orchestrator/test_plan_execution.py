"""Testes para plan_execution node."""
import pytest


class TestPlanExecution:
    """Testes para o no plan_execution."""

    def test_plan_execution_import(self):
        """plan_execution deve ser importavel."""
        from projects.agent.orchestrator.nodes.plan_execution import plan_execution
        assert plan_execution is not None

    def test_create_execution_plan_analyze(self):
        """Deve criar plano para analyze_performance."""
        from projects.agent.orchestrator.nodes.plan_execution import create_execution_plan

        plan = create_execution_plan("analyze_performance", config_id=1)

        assert plan["parallel"] is True
        assert "classification" in plan["agents"]
        assert "campaign" in plan["agents"]
        assert len(plan["tasks"]) == len(plan["agents"])

    def test_create_execution_plan_full_report(self):
        """Deve criar plano completo para full_report."""
        from projects.agent.orchestrator.nodes.plan_execution import create_execution_plan

        plan = create_execution_plan("full_report", config_id=1)

        assert len(plan["agents"]) == 4
        assert "classification" in plan["agents"]
        assert "anomaly" in plan["agents"]
        assert "recommendation" in plan["agents"]
        assert "forecast" in plan["agents"]

    def test_create_execution_plan_has_tasks(self):
        """Plano deve ter tasks para cada agente."""
        from projects.agent.orchestrator.nodes.plan_execution import create_execution_plan

        plan = create_execution_plan("find_problems", config_id=1)

        for agent in plan["agents"]:
            assert agent in plan["tasks"]
            task = plan["tasks"][agent]
            assert "description" in task
            assert "context" in task
            assert "priority" in task

    def test_create_execution_plan_timeout(self):
        """Plano deve ter timeout baseado nos agentes."""
        from projects.agent.orchestrator.nodes.plan_execution import create_execution_plan

        plan = create_execution_plan("analyze_performance", config_id=1)

        assert "timeout" in plan
        assert plan["timeout"] > 0

    def test_plan_execution_sets_required_agents(self):
        """plan_execution deve retornar required_agents."""
        from projects.agent.orchestrator.nodes.plan_execution import plan_execution

        state = {
            "user_intent": "find_problems",
            "config_id": 1,
            "context": {},
        }

        result = plan_execution(state)

        assert result.get("required_agents") == ["anomaly", "classification"]
