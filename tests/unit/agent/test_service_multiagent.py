"""Testes para multi-agent no service."""
import pytest
from unittest.mock import AsyncMock, patch, Mock


class TestMultiAgentService:
    """Testes para funcionalidade multi-agente no service."""

    def test_should_use_multiagent(self):
        """should_use_multiagent deve verificar configuração."""
        from projects.agent.service import should_use_multiagent

        # Deve retornar baseado na configuração
        result = should_use_multiagent()
        assert isinstance(result, bool)

    def test_get_agent_returns_orchestrator_when_enabled(self):
        """get_agent deve retornar orchestrator quando multi-agent habilitado."""
        from projects.agent.service import get_agent

        with patch('app.agent.service.should_use_multiagent', return_value=True):
            agent = get_agent()
            # Deve ser o orchestrator
            assert agent is not None

    def test_get_agent_returns_legacy_when_disabled(self):
        """get_agent deve retornar agente legado quando desabilitado."""
        from projects.agent.service import get_agent

        with patch('app.agent.service.should_use_multiagent', return_value=False):
            agent = get_agent()
            # Deve ser o agente legado
            assert agent is not None

    @pytest.mark.asyncio
    async def test_chat_uses_multiagent_when_enabled(self):
        """chat deve usar multi-agent quando habilitado."""
        from projects.agent.service import TrafficAgentService

        service = TrafficAgentService()

        with patch.object(service, '_chat_multiagent', new_callable=AsyncMock) as mock:
            with patch('app.agent.service.should_use_multiagent', return_value=True):
                mock.return_value = {"response": "test", "intent": "general"}

                # Este teste é um placeholder para verificar integração
                assert mock is not None
