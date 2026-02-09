"""Testes para o no persist_memory."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from langchain_core.messages import HumanMessage

from projects.agent.orchestrator.nodes.persist_memory import persist_memory


@pytest.mark.asyncio
class TestPersistMemory:
    async def test_summarizes_long_conversations(self):
        messages = [HumanMessage(content=f"msg {i}") for i in range(25)]

        state = {
            "messages": messages,
            "thread_id": "t-1",
            "user_id": 1,
            "config_id": 1,
            "synthesized_response": "Resposta final.",
            "conversation_summary": None,
        }

        mock_service = AsyncMock()
        mock_service.get_or_create_summary.return_value = ("Resumo novo.", messages[-10:])

        with patch(
            "projects.agent.memory.summarization.SummarizationService",
            return_value=mock_service,
        ):
            with patch(
                "projects.agent.orchestrator.nodes.persist_memory.get_agent_settings",
                return_value=MagicMock(
                    summarization_enabled=True,
                    summarization_threshold=20,
                    summarization_keep_recent=10,
                    vector_store_enabled=False,
                    entity_memory_enabled=False,
                ),
            ):
                result = await persist_memory(state)

        mock_service.get_or_create_summary.assert_called_once()

    async def test_skips_when_disabled(self):
        state = {
            "messages": [HumanMessage(content="oi")],
            "thread_id": "t-1",
            "user_id": 1,
            "config_id": 1,
            "synthesized_response": "Resposta.",
            "conversation_summary": None,
        }

        with patch(
            "projects.agent.orchestrator.nodes.persist_memory.get_agent_settings",
            return_value=MagicMock(
                summarization_enabled=False,
                vector_store_enabled=False,
                entity_memory_enabled=False,
            ),
        ):
            result = await persist_memory(state)

        assert result == {}

    async def test_stores_embedding_when_enabled(self):
        state = {
            "messages": [HumanMessage(content="oi")],
            "thread_id": "t-1",
            "user_id": 1,
            "config_id": 1,
            "synthesized_response": "Resposta sintetizada.",
            "conversation_summary": None,
            "user_intent": "analysis",
            "agent_results": {"analysis": {}},
        }

        mock_embedding_service = AsyncMock()
        mock_embedding_service.store_embedding = AsyncMock()

        with patch(
            "projects.agent.orchestrator.nodes.persist_memory.get_agent_settings",
            return_value=MagicMock(
                summarization_enabled=False,
                vector_store_enabled=True,
                entity_memory_enabled=False,
            ),
        ):
            with patch(
                "projects.agent.memory.embeddings.EmbeddingService",
                return_value=mock_embedding_service,
            ):
                result = await persist_memory(state)

        mock_embedding_service.store_embedding.assert_called_once()
        assert result == {}

    async def test_extracts_entities_when_enabled(self):
        state = {
            "messages": [HumanMessage(content="oi")],
            "thread_id": "t-1",
            "user_id": 1,
            "config_id": 1,
            "synthesized_response": "Campanha XYZ tem CPL de R$ 25,00.",
            "conversation_summary": None,
        }

        mock_entity_service = AsyncMock()
        mock_entity_service.extract_entities.return_value = [
            {"type": "campaign", "key": "Campanha XYZ", "value": "CPL R$ 25,00"}
        ]
        mock_entity_service.save_entities = AsyncMock()

        with patch(
            "projects.agent.orchestrator.nodes.persist_memory.get_agent_settings",
            return_value=MagicMock(
                summarization_enabled=False,
                vector_store_enabled=False,
                entity_memory_enabled=True,
            ),
        ):
            with patch(
                "projects.agent.memory.entities.EntityMemoryService",
                return_value=mock_entity_service,
            ):
                result = await persist_memory(state)

        mock_entity_service.extract_entities.assert_called_once()
        mock_entity_service.save_entities.assert_called_once()
        assert result == {}

    async def test_skips_without_thread_id(self):
        state = {
            "messages": [HumanMessage(content="oi")],
            "thread_id": None,
            "user_id": 1,
            "config_id": 1,
            "synthesized_response": "Resposta.",
            "conversation_summary": None,
        }

        result = await persist_memory(state)
        assert result == {}
