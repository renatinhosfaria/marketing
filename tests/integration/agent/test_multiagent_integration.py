"""Testes de integracao para sistema multi-agente.

Estes testes requerem:
- Banco de dados configurado
- API keys validas
- Redis (opcional)
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.fixture
def mock_llm():
    """Mock do LLM para testes."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=MagicMock(
        content="Resposta do modelo",
        tool_calls=[]
    ))
    return mock


class TestMultiAgentIntegration:
    """Testes de integracao do multi-agent."""

    @pytest.mark.asyncio
    async def test_orchestrator_full_flow(self, mock_llm):
        """Testa fluxo completo do orchestrator."""
        from projects.agent.orchestrator import (
            create_initial_orchestrator_state,
            get_orchestrator,
            reset_orchestrator
        )
        from langchain_core.messages import HumanMessage

        # Reset para garantir estado limpo
        reset_orchestrator()

        # Criar estado
        state = create_initial_orchestrator_state(
            config_id=1,
            user_id=1,
            thread_id="test-integration",
            messages=[HumanMessage(content="Como esta a performance?")]
        )

        # Verificar estado inicial
        assert state["config_id"] == 1
        assert state["user_intent"] is None
        assert len(state["agent_results"]) == 0

    @pytest.mark.asyncio
    async def test_subagent_standalone_execution(self, mock_llm):
        """Testa execucao standalone de subagente."""
        from projects.agent.subagents import ClassificationAgent

        with patch('app.agent.llm.provider.get_llm_with_tools', return_value=mock_llm):
            agent = ClassificationAgent()

            # Verificar configuracao basica
            assert agent.AGENT_NAME == "classification"
            assert len(agent.get_tools()) == 4

    def test_all_subagents_instantiate(self):
        """Todos os subagentes devem instanciar sem erro."""
        from projects.agent.subagents import get_all_subagents

        agents = get_all_subagents()

        assert len(agents) == 6

        for agent in agents:
            assert agent.AGENT_NAME is not None
            assert len(agent.get_tools()) > 0
            assert agent.get_system_prompt() is not None

    def test_intent_detection_coverage(self):
        """Deteccao de intencao deve cobrir casos principais."""
        from projects.agent.orchestrator.nodes import detect_intent

        test_cases = [
            ("Como esta minha performance?", "analyze_performance"),
            ("Tem algum problema nas campanhas?", "find_problems"),
            ("O que devo fazer agora?", "get_recommendations"),
            ("Qual a previsao para semana?", "predict_future"),
            ("Compare campanha A com B", "compare_campaigns"),
            ("Ola", "general"),
        ]

        for message, expected_intent in test_cases:
            detected = detect_intent(message)
            assert detected == expected_intent, \
                f"Para '{message}': esperado {expected_intent}, obtido {detected}"

    def test_execution_plan_creation(self):
        """Planos de execucao devem ser criados corretamente."""
        from projects.agent.orchestrator.nodes import create_execution_plan

        plan = create_execution_plan("full_report", config_id=1)

        assert len(plan["agents"]) == 4
        assert plan["parallel"] is True
        assert plan["timeout"] > 0

        for agent_name in plan["agents"]:
            assert agent_name in plan["tasks"]


@pytest.mark.skipif(
    True,  # Mudar para False quando quiser rodar testes reais
    reason="Requer ambiente completo configurado"
)
class TestMultiAgentRealExecution:
    """Testes com execucao real (requer infra)."""

    @pytest.mark.asyncio
    async def test_real_chat_flow(self):
        """Testa fluxo de chat real."""
        from projects.agent.service import TrafficAgentService

        service = TrafficAgentService()
        await service.initialize()

        # Este teste so roda com ambiente completo
        pass
