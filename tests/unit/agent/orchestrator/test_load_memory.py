"""Testes para o no load_memory."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from langchain_core.messages import HumanMessage, SystemMessage

from projects.agent.orchestrator.nodes.load_memory import load_memory


def _mock_settings(**overrides):
    defaults = {
        "summarization_enabled": True,
        "vector_store_enabled": False,
        "entity_memory_enabled": False,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


@pytest.mark.asyncio
class TestLoadMemory:
    async def test_loads_existing_summary(self):
        state = {
            "messages": [HumanMessage(content="oi")],
            "thread_id": "t-1",
            "user_id": 1,
            "config_id": 1,
            "conversation_summary": None,
        }

        mock_service = AsyncMock()
        mock_service.load_summary.return_value = "Resumo anterior da conversa."

        with patch(
            "projects.agent.memory.summarization.SummarizationService",
            return_value=mock_service,
        ):
            with patch(
                "projects.agent.orchestrator.nodes.load_memory.get_agent_settings",
                return_value=_mock_settings(),
            ):
                result = await load_memory(state)

        assert result["conversation_summary"] == "Resumo anterior da conversa."
        injected = [m for m in result["messages"] if isinstance(m, SystemMessage)]
        assert any("Resumo anterior" in m.content for m in injected)

    async def test_no_summary_when_disabled(self):
        state = {
            "messages": [HumanMessage(content="oi")],
            "thread_id": "t-1",
            "user_id": 1,
            "config_id": 1,
            "conversation_summary": None,
        }

        with patch(
            "projects.agent.orchestrator.nodes.load_memory.get_agent_settings",
            return_value=_mock_settings(summarization_enabled=False),
        ):
            result = await load_memory(state)

        assert result.get("conversation_summary") is None

    async def test_no_summary_when_none_exists(self):
        state = {
            "messages": [HumanMessage(content="oi")],
            "thread_id": "t-1",
            "user_id": 1,
            "config_id": 1,
            "conversation_summary": None,
        }

        mock_service = AsyncMock()
        mock_service.load_summary.return_value = None

        with patch(
            "projects.agent.memory.summarization.SummarizationService",
            return_value=mock_service,
        ):
            with patch(
                "projects.agent.orchestrator.nodes.load_memory.get_agent_settings",
                return_value=_mock_settings(),
            ):
                result = await load_memory(state)

        assert result.get("conversation_summary") is None

    async def test_loads_user_memory_when_enabled(self):
        state = {
            "messages": [HumanMessage(content="Qual o CPL?")],
            "thread_id": "t-1",
            "user_id": 1,
            "config_id": 1,
            "conversation_summary": None,
        }

        mock_user_memory = MagicMock()
        mock_user_memory.get_user_context = AsyncMock(return_value={
            "entities": [{"type": "preference", "key": "CPL max", "value": "R$30"}],
            "related_conversations": [],
        })
        mock_user_memory.format_context_for_injection.return_value = (
            "## Memoria do usuario\n\n### Conhecimento\n- [preference] CPL max: R$30"
        )

        with patch(
            "projects.agent.orchestrator.nodes.load_memory.get_agent_settings",
            return_value=_mock_settings(
                summarization_enabled=False,
                entity_memory_enabled=True,
            ),
        ):
            with patch(
                "projects.agent.memory.user_memory.UserMemoryService",
                return_value=mock_user_memory,
            ):
                result = await load_memory(state)

        assert result.get("user_entities") is not None
        assert len(result["user_entities"]) == 1
        msgs = [m for m in result.get("messages", []) if isinstance(m, SystemMessage)]
        assert any("CPL max" in m.content for m in msgs)
