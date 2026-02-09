"""Testes para UserMemoryService."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from projects.agent.memory.user_memory import UserMemoryService


def _mock_settings(**overrides):
    defaults = {
        "entity_memory_enabled": True,
        "entity_max_per_user": 50,
        "vector_store_enabled": True,
        "cross_thread_enabled": True,
        "rag_top_k": 3,
        "rag_min_similarity": 0.75,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


@pytest.mark.asyncio
class TestUserMemoryService:
    async def test_get_user_context_combines_entities_and_rag(self):
        mock_entity_service = AsyncMock()
        mock_entity_service.load_user_entities.return_value = [
            {"type": "preference", "key": "CPL maximo", "value": "R$ 30,00", "confidence": 0.9, "mention_count": 3},
        ]

        mock_embedding_service = AsyncMock()
        mock_embedding_service.search_similar.return_value = [
            {"content": "Campanha XYZ com CPL baixo", "similarity": 0.88, "thread_id": "t-old"},
        ]

        service = UserMemoryService()

        with patch("projects.agent.memory.user_memory.get_agent_settings", return_value=_mock_settings()):
            with patch.object(service, "_entity_service", mock_entity_service):
                with patch.object(service, "_embedding_service", mock_embedding_service):
                    context = await service.get_user_context(
                        user_id=1,
                        current_query="Qual o CPL das campanhas?",
                        current_thread_id="t-new",
                    )

        assert context["entities"] is not None
        assert len(context["entities"]) == 1
        assert context["related_conversations"] is not None
        assert len(context["related_conversations"]) == 1

    async def test_returns_empty_when_no_history(self):
        mock_entity_service = AsyncMock()
        mock_entity_service.load_user_entities.return_value = []

        mock_embedding_service = AsyncMock()
        mock_embedding_service.search_similar.return_value = []

        service = UserMemoryService()

        with patch("projects.agent.memory.user_memory.get_agent_settings", return_value=_mock_settings()):
            with patch.object(service, "_entity_service", mock_entity_service):
                with patch.object(service, "_embedding_service", mock_embedding_service):
                    context = await service.get_user_context(
                        user_id=1,
                        current_query="Oi",
                        current_thread_id="t-1",
                    )

        assert context["entities"] == []
        assert context["related_conversations"] == []

    def test_format_context_with_entities_and_conversations(self):
        service = UserMemoryService()
        context = {
            "entities": [
                {"type": "preference", "key": "CPL maximo", "value": "R$ 30,00"},
            ],
            "related_conversations": [
                {"content": "Campanha XYZ com CPL baixo", "similarity": 0.88},
            ],
        }
        formatted = service.format_context_for_injection(context)
        assert formatted is not None
        assert "CPL maximo" in formatted
        assert "Campanha XYZ" in formatted

    def test_format_context_returns_none_when_empty(self):
        service = UserMemoryService()
        context = {"entities": [], "related_conversations": []}
        formatted = service.format_context_for_injection(context)
        assert formatted is None
